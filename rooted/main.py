import argparse
import os

from dataManipulation import *
from utils import tree_summary, summary, summary_raw, get_support_info
from vbpi import VBPI
import time
import numpy as np
import datetime

parser = argparse.ArgumentParser()


########## Data arguments
parser.add_argument('--dataset', required=True, help=' HCV ')
parser.add_argument('--supportType', type=str, default='mcmc', help=' ufboot | mcmc ')
parser.add_argument('--burnin', type=int, default=0, help=' the number of samples to skip at first ')
parser.add_argument('--empFreq', default=False, action='store_true', help=' empirical frequency for KL computation ')

########## Model arguments
parser.add_argument('--coalescent_type', required=True, help=' constant | skyride ')
parser.add_argument('--root_height_offset', type=float, default=5.0, help=' constant shift of the root height')
parser.add_argument('--init_clock_rate', type=float, default=0.001, help=' initial rate for strict clock model ')
parser.add_argument('--clock_type', required=True, help=' fixed_rate | strict ')
parser.add_argument('--log_pop_size_offset', type=float, default=10.0, help=' constant shift of the log population size ')
parser.add_argument('--sample_info', default=False, action='store_true', help=' use tip sample date ')
parser.add_argument('--psp', default=False, action='store_true', help=' use psp parameterization ')

########## Optimizer arguments
parser.add_argument('--stepszTree', type=float, default=0.001, help=' step size for tree topology parameters')
parser.add_argument('--stepszBranch', type=float, default=0.001, help=' step size for branch length parameters ')
parser.add_argument('--stepszCoalescent', type=float, default=0.001, help=' step size for coalescent parameters ')
parser.add_argument('--stepszClock', type=float, default=0.001, help=' step size for clock rate parameters ')
parser.add_argument('--maxIter', type=int, default=200000, help=' number of iterations for training')
parser.add_argument('--invT0', type=float, default=0.001, help=' initial inverse temperature for annealing schedule ')
parser.add_argument('--nwarmStart', type=int, default=50000, help=' number of warm start iterations ')
parser.add_argument('--nParticle', type=int, default=10, help=' number of particles for variational objectives ')
parser.add_argument('--ar', type=float, default=0.75, help=' step size anneal rate ')
parser.add_argument('--af', type=int, default=20000, help=' step size anneal frequency ')
parser.add_argument('--tf', type=int, default=1000, help=' monitor frequency during training ')
parser.add_argument('--lbf', type=int, default=5000, help=' lower bound test frequency')
parser.add_argument('--gradMethod', type=str, default='vimco', help=' vimco | rws ')


args = parser.parse_args()

args.result_folder = 'results/' + args.dataset
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

args.save_to_path = args.result_folder + '/' + args.supportType + '_' + args.gradMethod + '_' + str(args.nParticle)
if args.psp:
    args.save_to_path = args.save_to_path + '_psp'
args.save_to_path = args.save_to_path + '_' + args.coalescent_type + '_' + args.clock_type + '_'  + str(datetime.datetime.now()) +'.pt'

print('Training with the following settings: {}'.format(args))

###### Load Data
print('\nLoading Data set: {} ......'.format(args.dataset))
run_time = -time.time()


mcmc_support_path = 'data/' + args.dataset + '/' + args.dataset + '_' + args.coalescent_type + '_support_short_run'
mcmc_support_trees_dict, mcmc_support_trees_wts = summary(mcmc_support_path, 'nexus', burnin=args.burnin)
data, taxa = loadData('data/' + args.dataset + '/' + args.dataset + '.nexus', 'nexus')

run_time += time.time()
print('Support loaded in {:.1f} seconds'.format(run_time))

emp_tree_freq = None

if args.empFreq:
    print('\nEstimating empirical tree frequency ......')
    run_time = -time.time()
    sampled_tree_dict, sampled_tree_wts, _  = tree_summary('data/DENV4/DENV4_constant_golden_run.trees', 'nexus', burnin=25001)
    emp_tree_freq = {sampled_tree_dict[tree_id]: tree_wts for tree_id, tree_wts in sorted(sampled_tree_wts.items(), key=lambda x:x[1], reverse=True)}
    run_time += time.time()
    print('Empirical frequency loaded in {:.1f} seconds'.format(run_time))
    del sampled_tree_dict, sampled_tree_wts

sample_info = None
if args.sample_info:
    sample_info = [1994.0 - float('19'+taxon[-2:]) for taxon in taxa]
rootsplit_supp_dict, subsplit_supp_dict = get_support_info(taxa, mcmc_support_trees_dict)
del mcmc_support_trees_dict, mcmc_support_trees_wts

model = VBPI(taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden=np.ones(4)/4., subModel=('JC', 1.0), sample_info=sample_info, clock_type=args.clock_type,
             emp_tree_freq=emp_tree_freq, psp=args.psp, coalescent_type=args.coalescent_type, root_height_offset=args.root_height_offset, log_pop_size_offset=args.log_pop_size_offset, clock_rate=args.init_clock_rate)

print('Parameter Info:')
for param in model.parameters():
    print(param.dtype, param.size())

print('\nVBPI running, results will be saved to: {}\n'.format(args.save_to_path))
test_lb, test_kl_div = model.learn({'tree':args.stepszTree, 'branch':args.stepszBranch, 'coalescent':args.stepszCoalescent, 'clock':args.stepszClock},
            maxiter=args.maxIter, test_freq=args.tf, n_particles=args.nParticle, anneal_freq=args.af, init_inverse_temp=args.invT0, warm_start_interval=args.nwarmStart,
            method=args.gradMethod, save_to_path=args.save_to_path)

np.save(args.save_to_path.replace('.pt', '_test_lb.npy'), test_lb)
if args.empFreq:
    np.save(args.save_to_path.replace('.pt', '_kl_div.npy'), test_kl_div)
