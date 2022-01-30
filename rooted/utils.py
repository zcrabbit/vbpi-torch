import torch
import numpy as np
from ete3 import Tree
from bitarray import bitarray
from Bio import Phylo
from io import StringIO
from treeManipulation import namenum
import dendropy
from collections import OrderedDict, defaultdict



class BitArray(object):
    def __init__(self, taxa):
        self.taxa = taxa
        self.ntaxa = len(taxa)
        self.map = {taxon: i for i, taxon in enumerate(taxa)}
        
    def combine(self, arrA, arrB):
        if arrA < arrB:
            return arrA + arrB
        else:
            return arrB + arrA 
        
    def merge(self, key):
        return bitarray(key[:self.ntaxa]) | bitarray(key[self.ntaxa:])
        
    def decomp_minor(self, key):
        return min(bitarray(key[:self.ntaxa]), bitarray(key[self.ntaxa:]))
        
    def minor(self, arrA):
        return min(arrA, ~arrA)
        
    def from_clade(self, clade):
        bit_list = ['0'] * self.ntaxa
        for taxon in clade:
            bit_list[self.map[taxon]] = '1'
        return bitarray(''.join(bit_list))
    

class ParamParser(object):
    def __init__(self):
        self.start_and_end = {}
        self.num_params = 0
        self.num_params_in_dicts = 0
        self.dict_name_list = []
        # self.dict_len = []
        
    def add_item(self, name):
        start = self.num_params
        self.num_params += 1
        self.start_and_end[name] = start
        
    def check_item(self, name):
        return name in self.start_and_end
        
    def add_dict(self, name, record_name=True):
        start = self.num_params_in_dicts
        self.num_params_in_dicts = self.num_params
        self.start_and_end[name] = (start, self.num_params)
        if record_name:
            self.dict_name_list.append(name)
        # self.dict_len.append(self.num_params - start)
    
    def get(self, tensor, name):
        start, end = self.start_and_end[name]
        return tensor[start:end]
        
    def get_index_or_slice(self, name):
        index_or_slice = self.start_and_end[name]
        if isinstance(index_or_slice, tuple):
            start, end = index_or_slice
            return list(range(start, end))
        else:
            return index_or_slice
                
    def get_index(self, name):
        return self.start_and_end[name]


def readTree(filename):
    with open(filename,'r') as readin_file:
        while True:
            line = readin_file.readline()
            if line == "":
                break
            newick = line.strip('\n').split('\t')[0]
            tree = Tree(newick,format=0)
        
    return tree
    

def read_mcmc_trees(filename, data_type, burnin=None, taxon=None):
    mcmc_samp_tree_stats = Phylo.parse(filename, data_type)
    mcmc_samp_tree_list = []
    num_processed_tree = 0
    for tree in mcmc_samp_tree_stats:
        num_processed_tree += 1
        if burnin and num_processed_tree <= burnin:
            continue
            
        handle = StringIO()
        Phylo.write(tree, handle, 'newick')
        mcmc_samp_tree = Tree(handle.getvalue().strip())
        if taxon:
            namenum(mcmc_samp_tree, taxon)
        
        handle.close()
        mcmc_samp_tree_list.append(mcmc_samp_tree)
    
    return mcmc_samp_tree_list


def saveTree(sampled_tree, filename, tree_format=5):
    if type(sampled_tree) is not list:
        sampled_tree = [sampled_tree]
        
    with open(filename,'w') as output_file:
        for tree in sampled_tree:
            tree_newick = tree.write(format=tree_format)
            output_file.write(tree_newick + '\n')


def parse_tree(tree, sample_info=None):
    height = []
    event_info = []
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            if sample_info is not None:
                node.height = sample_info[node.name]                
            else:
                node.height = 0.0
            height.append(node.height)
            event_info.append(-1)
        else:
            c1 = node.children[0]
            c2 = node.children[1]
            
#             pdb.set_trace()
            # print(node.name, branch[c1.name] + c1.height, branch[c2.name] + c2.height)
#             assert abs(branch[c1.name] + c1.height - branch[c2.name] - c2.height) < 1e-10
            assert abs(c1.dist + c1.height - c2.dist - c2.height) < 1e-10
            node.height = c1.dist + c1.height
            height.append(node.height)
            event_info.append(1)
            
    return torch.FloatTensor(height), torch.FloatTensor(event_info)


def tree_summary(path, schema, burnin=0):
    sampled_trees_dendropy = dendropy.TreeList.get(path=path, schema=schema)
    sampled_tree_dict = {}
    sampled_tree_counts = defaultdict(float)
    sampled_tree_ids = []
    for tree in sampled_trees_dendropy[burnin:]:
        sampled_tree = Tree(tree.as_string('newick', suppress_rooting=True))
        sampled_tree_id = sampled_tree.get_topology_id()
        if sampled_tree_id not in sampled_tree_dict:
            sampled_tree_dict[sampled_tree_id] = sampled_tree
            sampled_tree_ids.append(sampled_tree_id)
        sampled_tree_counts[sampled_tree_id] += 1.0
    
    num_of_sampled_trees = sum(sampled_tree_counts.values())
    for tree_id, counts in sampled_tree_counts.items():
        sampled_tree_counts[tree_id] = counts / num_of_sampled_trees
        
    return sampled_tree_dict, sampled_tree_counts, sampled_tree_ids


def summary(path, schema, burnin=0, n_rep=10):
    tree_dict_total = OrderedDict()
    tree_id_total = []
    for i in range(1, n_rep+1):
        tree_dict_rep, tree_counts_rep, tree_ids_rep = tree_summary(path + '_rep_{}'.format(i) + '.trees', schema, burnin)
        for tree_id in tree_ids_rep:
            if tree_id not in tree_id_total:
                tree_id_total.append(tree_id)
                tree_dict_total[tree_id] = tree_dict_rep[tree_id]

    return tree_dict_total, tree_id_total


def get_tree_list_raw(filename, burnin=0, truncate=None, hpd=0.95):
    tree_dict = {}
    tree_wts_dict = defaultdict(float)
    tree_names = []
    i, num_trees = 0, 0
    with open(filename, 'r') as input_file:
        while True:
            line = input_file.readline()
            if line == "":
                break
            num_trees += 1
            if num_trees < burnin:
                continue
            tree = Tree(line.strip())
            tree_id = tree.get_topology_id()
            if tree_id not in tree_wts_dict:
                tree_name = 'tree_{}'.format(i)
                tree_dict[tree_name] = tree
                tree_names.append(tree_name)
                i += 1
            tree_wts_dict[tree_id] += 1.0

            if truncate and num_trees == truncate + burnin:
                break
    tree_wts = [tree_wts_dict[tree_dict[tree_name].get_topology_id()]/(num_trees-burnin) for tree_name in tree_names]
    if hpd < 1.0:
        ordered_wts_idx = np.argsort(tree_wts)[::-1]
        cum_wts_arr = np.cumsum([tree_wts[k] for k in ordered_wts_idx])
        cut_at = next(x[0] for x in enumerate(cum_wts_arr) if x[1] > hpd)
        tree_wts = [tree_wts[k] for k in ordered_wts_idx[:cut_at]]
        tree_names = [tree_names[k] for k in ordered_wts_idx[:cut_at]]

    return tree_dict, tree_names, tree_wts


def summary_raw(file_path, truncate=None, hpd=0.95, n_rep=10):
    tree_dict_total = {}
    tree_id_set_total = set()
    tree_names_total = []
    n_samp_tree = 0

    for i in range(1, n_rep+1):
        tree_dict_rep, tree_names_rep, tree_wts_rep = get_tree_list_raw(file_path + '_rep_{}'.format(i), truncate=truncate, hpd=hpd)
        for j, name in enumerate(tree_names_rep):
            tree_id = tree_dict_rep[name].get_topology_id()
            if tree_id not in tree_id_set_total:
                n_samp_tree += 1
                tree_names_total.append('tree_{}'.format(n_samp_tree))
                tree_dict_total[tree_names_total[-1]] = tree_dict_rep[name]
                tree_id_set_total.add(tree_id)

    return tree_dict_total, tree_names_total
    

def get_support_info(taxa, sampled_tree_dict):
    rootsplit_supp_dict = OrderedDict()
    subsplit_supp_dict = OrderedDict()
    toBitArr = BitArray(taxa)
    for tree in sampled_tree_dict.values():
        nodetobitMap = {node: toBitArr.from_clade(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
        for node in tree.traverse('levelorder'):
            if node.is_root():
                rootsplit = min([nodetobitMap[child] for child in node.children]).to01()
                if rootsplit not in rootsplit_supp_dict:
                    rootsplit_supp_dict[rootsplit] = 0.0
                rootsplit_supp_dict[rootsplit] += 1.0
            elif not node.is_leaf():
                child_subsplit = min([nodetobitMap[child] for child in node.children]).to01()
                for sister in node.get_sisters():
                    parent_subsplit = (nodetobitMap[sister] + nodetobitMap[node]).to01()
                    if parent_subsplit not in subsplit_supp_dict:
                        subsplit_supp_dict[parent_subsplit] = OrderedDict()
                    if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                        subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                    subsplit_supp_dict[parent_subsplit][child_subsplit] += 1.0
    
    return rootsplit_supp_dict, subsplit_supp_dict
    

def beast_stats(filename):
    ss_stats = []
    with open(filename, 'r') as readin_file:
        while True:
            line = readin_file.readline()
            if line == "":
                break
            if line[0] == '#':
                continue
            
            if line[0] == 's':
                names = line.strip('\n').split('\t')
            else:
                stats = line.strip('\n').split('\t')
                ss_stats.append([float(stat) for stat in stats])
    return np.asarray(ss_stats), names
