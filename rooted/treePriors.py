import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import pdb


class BaseCoalescent(nn.Module):
    def __init__(self, now=0.0):
        super().__init__()
        self.now = now

        
    def get_coalescent_info(self, height, event_info, grid=False):
        self.height = height
        if grid:
            eff_grid = self.grid[self.grid<self.height.max()]
        else:
            eff_grid = torch.tensor([])
        height_and_grid = torch.cat((self.height, eff_grid))
        event_info_updated = torch.cat((event_info, torch.zeros(eff_grid.size())))
        
        self.event_time, indices = torch.sort(height_and_grid, descending=True)
        self.event_info = torch.take(event_info_updated, indices)[:-1]
        lineages = 1 + torch.cumsum(self.event_info, dim=0)
        
        self.intervals = self.event_time[:-1] - self.event_time[1:]       
        self.coalescent_event_mask = torch.cat((self.event_info==1, torch.ByteTensor([0])))
        self.cf = lineages * (lineages - 1) / 2.
        
    
    def get_batch_coalescent_info(self, height, event_info, event_idxes=None, grid=False):
        self.height = height
        if grid:
            height_and_grid = torch.cat((self.height, self.grid.repeat(self.height.size(0), 1)), dim=-1)
            event_info_updated = torch.cat((event_info, torch.zeros((event_info.size(0), self.grid.size(0)))), dim=-1)
        else:
            height_and_grid = self.height
            event_info_updated = event_info
        
        self.event_time, indices = torch.sort(height_and_grid, descending=True, dim=-1)
        self.event_info = torch.gather(event_info_updated, 1, indices)[:, :-1]
        if event_idxes is not None:
            self.event_idxes = torch.gather(event_idxes, 1, indices)[:, :-1]
        lineages = 1 + torch.cumsum(self.event_info, dim=-1)
        
        self.intervals = self.event_time[:, :-1] - self.event_time[:, 1:]
        self.coalescent_event_mask = torch.cat((self.event_info==1, torch.zeros((self.height.size(0), 1), dtype=torch.uint8)), dim=-1)
        self.cf = lineages * (lineages - 1) / 2.
        
    def mean_std(self, *args):
        raise NotImplementedError
        
    def sample_pop_size(self, *args, n_particles=1):
        mean, std = self.mean_std(*args)
        sample_epsilon = torch.randn(n_particles, mean.size(-1))
        log_pop_size = mean + std.exp() * sample_epsilon
        logq_pop_size = torch.sum(-0.5*math.log(2*math.pi) - std - 0.5*sample_epsilon**2, dim=-1)
        
        return log_pop_size, logq_pop_size

    def mean_pop_size(self, *args):
        mean, std = self.mean_std(*args)
        return torch.exp(mean + std.exp()**2/2.)
    

class ConstantCoalescent(BaseCoalescent):
    def __init__(self, *args, mu_0=-0.0049, sigma_0=2.148, log_pop_size_offset=10., **kwargs):
        super().__init__()
        self.mu, self.sigma = mu_0, sigma_0
        self.log_pop_size_offset = log_pop_size_offset

        self.pop_size_param = nn.Parameter(torch.zeros(2,1))
    
    def update(self, height, event_info):
        self.get_coalescent_info(height, event_info)
        self.coalescent_count = self.coalescent_event_mask.sum().float()
        
    def update_batch(self, height, event_info):
        self.get_batch_coalescent_info(height, event_info)
        self.coalescent_count = self.coalescent_event_mask.sum(dim=-1).float()            

    def mean_std(self):
        # return self.pop_size_param[0], self.pop_size_param[1]
        return self.pop_size_param[0] + self.log_pop_size_offset, self.pop_size_param[1]
    
    def logprior(self, log_pop_size):
        return torch.sum(-0.5*math.log(2*math.pi) - 0.5*math.log(self.sigma**2) - 0.5*(log_pop_size-self.mu)**2/self.sigma**2, dim=-1)

    def loglikelihood(self, log_pop_size):
        log_pop_size = log_pop_size.squeeze()
        eff_pop_size = torch.exp(log_pop_size)
        loglikelihood = -log_pop_size*self.coalescent_count - torch.sum(self.intervals * self.cf, dim=-1)/eff_pop_size
        return loglikelihood
       
    def forward(self, log_pop_size, monitor_prior=True):
        log_coalescent_prior = self.loglikelihood(log_pop_size) + self.logprior(log_pop_size)
        if not monitor_prior:
            return log_coalescent_prior, None
        else:
            return log_coalescent_prior, (log_coalescent_prior - log_pop_size.squeeze()).detach()


class SkyrideCoalescent(BaseCoalescent):
    def __init__(self, ntips, gamma_alpha=0.001, gamma_beta=0.001, now=0.0, log_pop_size_offset=10., **kwargs):
        super().__init__(now)
        self.alpha, self.beta = gamma_alpha, gamma_beta
        self.log_pop_size_offset = log_pop_size_offset
        self.pop_size_param = nn.Parameter(torch.zeros(ntips-1, 2), requires_grad=True)
        
    def update_batch(self, height, event_info, event_idxes=None, grid=False):
        self.get_batch_coalescent_info(height, event_info, event_idxes=event_idxes, grid=grid)
        event_count = torch.cat((torch.nonzero(self.event_info==1)[:,1].view(height.size(0), -1), torch.ones((height.size(0), 1), dtype=torch.int64)*self.event_info.size(1)), dim=-1)
        if not grid:
            self.event_range = event_count[:, 1:] - event_count[:, :-1]
        else:
            self.event_range = torch.cat((event_count[:, 0].view(-1, 1), event_count[:, 1:] - event_count[:, :-1]), dim=-1)
        self.pop_size_mask = torch.from_numpy(np.repeat(np.tile(np.arange(self.event_range.size(1)), height.size(0)), self.event_range.numpy().flatten()).reshape(height.size(0), -1))
    
    def pad_param(self):
        self.param_padded = torch.cat((self.pop_size_param, torch.zeros(1, 2)), dim=0)

    def get_pop_traj(self, height, event_info, traj_list=[], samp_pop_size=None):
        self.update_batch(height, event_info, grid=True)
        with torch.no_grad():
            if samp_pop_size is None:
                samp_pop_size, _ = self.sample_pop_size(n_particles=height.size(0))
            samp_pop_size = torch.cat((samp_pop_size[:, :1], samp_pop_size), dim=1)
            log_eff_pop_size = torch.gather(samp_pop_size, -1, self.pop_size_mask)
            pop_size_traj = torch.masked_select(log_eff_pop_size, self.event_info==0).view(height.size(0), -1)
            
            traj_list.append(pop_size_traj)
    
    def mean_std(self):
        return self.pop_size_param[:, 0] + self.log_pop_size_offset, self.pop_size_param[:, 1] 
        
    def eff_pop_size(self, log_pop_size, log=False):
        log_eff_pop_size = torch.gather(log_pop_size, -1, self.pop_size_mask)
        if not log:
            return torch.exp(log_eff_pop_size)
        else:
            return log_eff_pop_size

    def sample_precision(self, log_pop_size):
        shape_value = 0.5*log_pop_size.size(1) + self.alpha
        rate_value = 0.5*torch.sum((log_pop_size[:, 1:] - log_pop_size[:, :-1])**2, dim=-1, keepdim=True) + self.beta
        return torch.from_numpy(np.random.gamma(shape_value, size=(log_pop_size.size(0), 1))).float() / rate_value 
        
    def logprior_batch(self, log_pop_size, log_tau):
        return torch.sum(-0.5*math.log(2*math.pi) + 0.5*log_tau - 0.5*log_tau.exp()*(log_pop_size[:, 1:] - log_pop_size[:, :-1])**2, dim=-1) + \
                         torch.sum(self.alpha*math.log(self.beta) - math.lgamma(self.alpha) + (self.alpha-1.)*log_tau - self.beta*log_tau.exp(), dim=-1)

    def logprior_marginal_batch(self, log_pop_size):
        half_pop_size_dim = 0.5*(log_pop_size.size(1)-1)
        return -half_pop_size_dim*math.log(2*math.pi) + self.alpha*math.log(self.beta) - math.lgamma(self.alpha) + math.lgamma(half_pop_size_dim+self.alpha) - (half_pop_size_dim+self.alpha)*torch.log(self.beta + 0.5*torch.sum((log_pop_size[:, 1:] - log_pop_size[:,:-1])**2, dim=-1))

    def loglikelihood_batch(self, log_pop_size):
        log_eff_pop_size = self.eff_pop_size(log_pop_size, log=True)
        inv_eff_pop_size = torch.exp(-log_eff_pop_size)
        loglikelihood = torch.sum(torch.masked_select(-log_eff_pop_size, self.coalescent_event_mask[:,:-1]).view(log_pop_size.size(0),-1), dim=-1) - \
                              torch.sum(inv_eff_pop_size * self.intervals * self.cf, dim=-1)
        return loglikelihood

    def forward(self, log_pop_size, monitor_precision=True):
        loglikelihood = self.loglikelihood_batch(log_pop_size)
        log_sampled_gmrf_prior = None
  
        if monitor_precision:
            log_tau = self.sample_precision(log_pop_size)
            log_sampled_gmrf_prior = loglikelihood + self.logprior_batch(log_pop_size, log_tau.log())

        return loglikelihood + self.logprior_marginal_batch(log_pop_size), log_sampled_gmrf_prior
        