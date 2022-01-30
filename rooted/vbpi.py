import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import random
import numpy as np
from utils import namenum
from branchModel_new import BaseModel
from treePriors import ConstantCoalescent, SkyrideCoalescent
from clockModel import FixedRateModel, StrictModel
from vector_sbnModel import SBN
from phyloModel import PHY

import pdb


class VBPI(nn.Module):
    EPS = 1e-40
    CoalescentModel = {'constant': ConstantCoalescent,
                       'skyride': SkyrideCoalescent}
    ClockRateModel = {'fixed_rate': FixedRateModel,
                      'strict': StrictModel}
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden, subModel, emp_tree_freq=None,
                 sample_info=None, psp=True, mu_0=-0.0049, sigma_0=2.148, coalescent_type='constant',
                 root_height_offset=0.0, grid_size=100, cut_off=200, gamma_alpha=0.001, gamma_beta=0.001,
                 log_pop_size_offset=10.0, clock_rate=1.0, mu_1=0.0, sigma_1=3.0, clock_type='fixed_rate', feature_dim=2, use_ambiguity=False):
        super().__init__()
        self.taxa, self.emp_tree_freq = taxa, emp_tree_freq
        if emp_tree_freq:
            self.trees, self.emp_freqs = zip(*emp_tree_freq.items())
            self.emp_freqs = np.array(self.emp_freqs)
            self.negDataEnt = np.sum(self.emp_freqs * np.log(np.maximum(self.emp_freqs, self.EPS)))
        
        self.ntips = len(data)
        self.phylo_model = PHY(data, taxa, pden, subModel, use_ambiguity=use_ambiguity)
        
        self.tree_model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict)
        self.rs_embedding_map, self.ss_embedding_map = self.tree_model.rs_map, self.tree_model.ss_map
        
        self.branch_model = BaseModel(self.ntips, self.rs_embedding_map, self.ss_embedding_map, feature_dim=feature_dim,
                                      sample_info=sample_info, psp=psp, root_height_offset=root_height_offset)
        
        self.now = min(self.branch_model.sample_info)
        self.tree_prior_model = self.CoalescentModel[coalescent_type](self.ntips, mu_0=mu_0, sigma_0=sigma_0, gamma_alpha=gamma_alpha,
                                                                 gamma_beta=gamma_beta, now=self.now, log_pop_size_offset=log_pop_size_offset)

        self.clock_model = self.ClockRateModel[clock_type](init_clock_rate=clock_rate, mu=mu_1, sigma=sigma_1)
        
        torch.set_num_threads(1)
    
    def load_from(self, state_dict_path):
        self.load_state_dict(torch.load(state_dict_path))
        self.eval()
        self.tree_model.update_CPDs()

    def kl_div(self):
        kl_div = 0.0
        for tree, wt in self.emp_tree_freq.items():
            kl_div += wt * np.log(max(np.exp(self.tree_model.loglikelihood(tree)), self.EPS))
        kl_div = self.negDataEnt - kl_div
        return kl_div
    
    def logq_tree(self, tree):
        return self.tree_model(tree)
    
    def sample_pop_traj(self, cut_off=200., grid_size=100, n_traj=1000, n_particles=10):
        self.tree_prior_model.grid = torch.linspace(0., cut_off, grid_size+1)[1:] + self.now - cut_off/2./grid_size
        samp_pop_traj = []
        with torch.no_grad():
            for run in range(int(n_traj/n_particles)):
                samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
                [namenum(tree, self.taxa) for tree in samp_trees]
                samp_branch, logq_height, height, event_info = self.branch_model(samp_trees)
                self.tree_prior_model.get_pop_traj(height, event_info, samp_pop_traj)
                
        return torch.cat(samp_pop_traj)

    def sample_tree_height(self, n_rep=1000):
        samp_tree_height = []
        with torch.no_grad():
            for rep in range(n_rep):
                samp_tree = self.tree_model.sample_tree()
                namenum(samp_tree, self.taxa)
                samp_branch, logq_height, height, event_info = self.branch_model([samp_tree])
                samp_tree_height.append(height.squeeze()[0])
        
        return torch.stack(samp_tree_height)
        
    def sample_tree_loglikelihood(self, n_rep=1000):
        samp_tree_logll = []
        with torch.no_grad():
            for rep in range(n_rep):
                samp_tree = self.tree_model.sample_tree()
                namenum(samp_tree, self.taxa)
                samp_branch, logq_height, height, event_info = self.branch_model([samp_tree])
                log_clock_rate, logq_clock_rate = self.clock_model.sample(n_particles=1)
                samp_branch = samp_branch * log_clock_rate.exp()
                samp_tree_logll.append(self.phylo_model.loglikelihood(samp_branch[0], samp_tree))              
        
        return torch.stack(samp_tree_logll)
    
    def lower_bound(self, n_particles=1, n_runs=1000):
        lower_bounds = []
        with torch.no_grad():
            for run in range(n_runs):
                samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
                [namenum(tree, self.taxa) for tree in samp_trees]
                logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
                
                samp_branch, logq_height, height, event_info = self.branch_model(samp_trees)
                log_clock_rate, logq_clock_rate = self.clock_model.sample(n_particles=n_particles)
                samp_branch = samp_branch * log_clock_rate.exp()
                logll = torch.stack([self.phylo_model.loglikelihood(branch, tree) for branch, tree in zip(*[samp_branch, samp_trees])])
                
                self.tree_prior_model.update_batch(height, event_info)
                coalescent_param, logq_prior = self.tree_prior_model.sample_pop_size(n_particles=n_particles)
                logp_coalescent_prior, _ = self.tree_prior_model(coalescent_param, False)
      
                logp_clock_rate = self.clock_model(log_clock_rate)
                   
                lower_bounds.append(torch.logsumexp(logll + logp_coalescent_prior + logp_clock_rate - logq_tree - logq_height - logq_prior - logq_clock_rate - math.log(n_particles), 0))            
            
            lower_bound = torch.stack(lower_bounds).mean()
            
        return lower_bound.item()
    
    def vimco_lower_bound(self, inverse_temp=1.0, n_particles=10):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]
        logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
        
        samp_branch, logq_height, height, event_info = self.branch_model(samp_trees)
        log_clock_rate, logq_clock_rate = self.clock_model.sample(n_particles=n_particles)
        samp_branch = samp_branch * log_clock_rate.exp()
        logll = torch.stack([self.phylo_model.loglikelihood(branch, tree) for branch, tree in zip(*[samp_branch, samp_trees])])
        
        self.tree_prior_model.update_batch(height, event_info)
        coalescent_param, logq_prior = self.tree_prior_model.sample_pop_size(n_particles=n_particles)
        logp_coalescent_prior, samp_logp_coalescent_prior = self.tree_prior_model(coalescent_param)

        logp_clock_rate = self.clock_model(log_clock_rate)

        logp_joint = inverse_temp * logll + logp_coalescent_prior + logp_clock_rate        
        lower_bound = torch.logsumexp(logll + logp_coalescent_prior + logp_clock_rate - logq_tree - logq_height - logq_prior -logq_clock_rate - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree - logq_height - logq_prior - logq_clock_rate
        mean_exclude_signal = (torch.sum(l_signal) - l_signal) / (n_particles-1.)
        control_variates = torch.logsumexp(l_signal.view(-1,1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(n_particles), dim=0)
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        vimco_fake_term = torch.sum((temp_lower_bound - control_variates).detach() * logq_tree, dim=0)
        return temp_lower_bound, vimco_fake_term, lower_bound, logll[-1], samp_logp_coalescent_prior[-1], torch.mean(height[:,0])

    def rws_lower_bound(self, inverse_temp=1.0, n_particles=10):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]
        logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])

        samp_branch, logq_height, height, event_info = self.branch_model(samp_trees)
        log_clock_rate, logq_clock_rate = self.clock_model.sample(n_particles=n_particles)
        samp_branch = samp_branch * log_clock_rate.exp()
        logll = torch.stack([self.phylo_model.loglikelihood(branch, tree) for branch, tree in zip(*[samp_branch, samp_trees])])

        self.tree_prior_model.update_batch(height, event_info)
        coalescent_param, logq_prior = self.tree_prior_model.sample_pop_size(n_particles=n_particles)
        logp_coalescent_prior, samp_logp_coalescent_prior = self.tree_prior_model(coalescent_param)

        logp_clock_rate = self.clock_model(log_clock_rate)

        logp_joint = inverse_temp * logll + logp_coalescent_prior + logp_clock_rate
        lower_bound = torch.logsumexp(logll + logp_coalescent_prior + logp_clock_rate - logq_tree -logq_height -logq_prior - logq_clock_rate - math.log(n_particles), 0)

        l_signal = logp_joint - logq_tree.detach() -logq_height - logq_prior - logq_clock_rate
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        snis_wts = torch.softmax(l_signal, dim=0)
        rws_fake_term = torch.sum(snis_wts.detach() * logq_tree, dim=0)
        return temp_lower_bound, rws_fake_term, lower_bound, logll[-1], samp_logp_coalescent_prior[-1], torch.mean(height[:,0])
    
    def learn(self, stepsz, maxiter=100000, test_freq=1000, lb_test_freq=5000, anneal_freq=20000, anneal_rate=0.75, n_particles=10,
              init_inverse_temp=0.001, warm_start_interval=50000, method='vimco', save_to_path=None):
        lbs, lls, lps, root_ages = [], [], [], []
        test_kl_div, test_lb = [], []
        
        if not isinstance(stepsz, dict):
            stepsz = {'tree': stepsz, 'branch': stepsz, 'coalescent': stepsz, 'clock': stepsz}
        
        optimizer = torch.optim.Adam([
                    {'params': self.tree_model.parameters(), 'lr': stepsz['tree']},
                    {'params': self.branch_model.parameters(), 'lr': stepsz['branch']},
                    {'params': self.tree_prior_model.parameters(), 'lr': stepsz['coalescent']},
                    {'params': self.clock_model.parameters(), 'lr': stepsz['clock']}
                ])
        run_time = -time.time()
        for it in range(1, maxiter+1):
            inverse_temp = min(1., init_inverse_temp + it * 1.0/warm_start_interval)
            if method == 'vimco':
                temp_lower_bound, vimco_fake_term, lower_bound, logll, logprior, root_age = self.vimco_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - vimco_fake_term
            elif method == 'rws':
                temp_lower_bound, rws_fake_term, lower_bound, logll, logprior, root_age = self.rws_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - rws_fake_term
            else:
                raise NotImplementedError
            lbs.append(lower_bound.item())
            lls.append(logll.item())
            lps.append(logprior.item())
            root_ages.append(root_age.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.tree_model.update_CPDs()
            
            if it % test_freq == 0:
                run_time += time.time()
                if self.emp_tree_freq:
                    test_kl_div.append(self.kl_div())
                    print('Iter {}:({:.1f}s) Lower Bound: {:.4f} | Logll: {:.4f} | Root Age: {:.4f} | KL: {:.6f}'.format(it, run_time, np.mean(lbs), lls[-1], np.mean(root_ages), test_kl_div[-1]))
                else:
                    print('Iter {}:({:.1f}s) Lower Bound: {:.4f} | Logprior: {:.4f} | Logll: {:.4f} | Root Age: {:.4f}'.format(it, run_time, np.mean(lbs), lps[-1], lls[-1], np.mean(root_ages)))
                if it % lb_test_freq == 0:
                    run_time = -time.time()
                    test_lb.append(self.lower_bound(n_particles=1))
                    run_time += time.time()
                    print('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f}'.format(it, run_time, test_lb[-1]))
                    
                run_time = -time.time()
                lbs, lls, lps, root_ages = [], [], [], []
            
            if it % anneal_freq == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= anneal_rate
        
        if save_to_path is not None:
            torch.save(self.state_dict(), save_to_path)

        return test_lb, test_kl_div
