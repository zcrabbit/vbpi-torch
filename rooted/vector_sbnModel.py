import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bitarray import bitarray
from ete3 import Tree
from utils import BitArray, ParamParser



class SBN(nn.Module):
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict):
        super().__init__()
        self.taxa, self.ntaxa = taxa, len(taxa)
        self.toBitArr = BitArray(taxa)
        self.rootsplit_supp_dict = rootsplit_supp_dict
        self.subsplit_supp_dict = subsplit_supp_dict
        
        self.CPDParser = ParamParser()
        for split in self.rootsplit_supp_dict:
            self.CPDParser.add_item(split)
        self.CPDParser.add_dict('rootsplit', record_name=False)
        self.rs_len = len(self.rootsplit_supp_dict)
        
        ss_mask, ss_max_len = [], 0
        for parent in self.subsplit_supp_dict:
            ss_len = len(self.subsplit_supp_dict[parent])
            if ss_len > 1:
                for child in self.subsplit_supp_dict[parent]:
                    self.CPDParser.add_item(parent+child)
                self.CPDParser.add_dict(parent)
                ss_mask.append(torch.ones(ss_len, dtype=torch.uint8))
                ss_max_len = max(ss_max_len, ss_len)
        
        self.ss_mask = torch.stack([F.pad(mask, (0, ss_max_len - mask.size(0)), 'constant', 0) for mask in ss_mask], dim=0)
        
        self.CPD_params = nn.Parameter(torch.zeros(self.CPDParser.num_params), requires_grad=True)
        self.idx_map = np.append(np.arange(self.CPDParser.num_params), [-2, -1])
        
        self.rs_CPDs = F.softmax(self.CPDParser.get(self.CPD_params, 'rootsplit'), 0)
        self.rs_map = {split: i for i, split in enumerate(self.rootsplit_supp_dict.keys())}
        self.rs_reverse_map = {i: split for i, split in enumerate(self.rootsplit_supp_dict.keys())}
        
        self.subsplit_parameter_set = set(self.CPDParser.dict_name_list)
        self.ss_name_map = {parent: i for i, parent in enumerate(self.CPDParser.dict_name_list)}
        
        self.ss_map = {}
        self.ss_reverse_map = {}
        self.ss_state = set()
        for parent in self.subsplit_supp_dict:
            self.ss_map[parent] = {child: i for i, child in enumerate(self.subsplit_supp_dict[parent].keys())}
            self.ss_reverse_map[parent] = {i: child for i, child in enumerate(self.subsplit_supp_dict[parent].keys())}
        
        ss_CPDs, self.ss_masked_CPDs = self.update_subsplit_CPDs()
        self.CPDs = torch.cat((self.rs_CPDs, ss_CPDs))
        self.one_tensor = torch.tensor([1.0])
        
    def update_rootsplit_CPDs(self):
        rs_params = self.CPDParser.get(self.CPD_params, 'rootsplit')
        self.rs_CPDs = F.softmax(rs_params, 0)
        if torch.isnan(self.rs_CPDs).any():
            raise Exception('Invalid rootsplit probability! Check rs_params:(max {:.4f}, min {:.4f})'.format(np.max(rs_params.detach().numpy()), np.min(rs_params.detach().numpy())))
    
    def update_subsplit_CPDs(self):
        temp_mat = torch.zeros(self.ss_mask.size())
        temp_mat.masked_scatter_(self.ss_mask, self.CPD_params[self.rs_len:])
        masked_temp_mat = temp_mat.masked_fill(1-self.ss_mask, -float('inf'))
        masked_CPDs = F.softmax(masked_temp_mat, dim=1)
        
        return masked_CPDs.masked_select(self.ss_mask), masked_CPDs
    
    def update_CPDs(self):
        self.update_rootsplit_CPDs()
        ss_CPDs, self.ss_masked_CPDs = self.update_subsplit_CPDs()
        self.CPDs = torch.cat((self.rs_CPDs, ss_CPDs))
        
    def check_parent_child(self, parent, child=None):
        if parent not in self.ss_map:
            return False
        else:
            if child and child not in self.ss_map[parent]:
                return False
        return True

    def subsplit_idxes_update(self, subsplit_idxes, ss_parent, ss_child):
        if not self.check_parent_child(ss_parent, ss_child):
            subsplit_idxes.append(-1)
        else:
            ss_name = ss_parent + ss_child
            if self.CPDParser.check_item(ss_name):
                subsplit_idxes.append(self.CPDParser.get_index(ss_name))
            else:
                subsplit_idxes.append(-2)
    
    def get_subsplit_CPDs(self, parent, child=None):
        if child:
            return self.CPDParser.get(self.CPDs, parent)[self.ss_map[parent][child]]
        else:
            if parent in self.subsplit_parameter_set:
                return self.CPDParser.get(self.CPDs, parent)
            else:
                return self.one_tensor
    
    def sample_tree(self, rooted=True):
        root = Tree()
        node_split_stack = [(root, '0'*self.ntaxa + '1'*self.ntaxa)]
        for i in range(self.ntaxa-1):
            node, split_bitarr = node_split_stack.pop()
            parent_clade_bitarr = bitarray(split_bitarr[self.ntaxa:])
            node.clade_bitarr = parent_clade_bitarr
            node.split_bitarr = min([parent_clade_bitarr, ~parent_clade_bitarr]).to01()
            if node.is_root():
                split_prob = self.rs_CPDs
                split = self.rs_reverse_map[torch.multinomial(split_prob, 1).item()]
            else:
                split_prob = self.get_subsplit_CPDs(split_bitarr)
                split = self.ss_reverse_map[split_bitarr][torch.multinomial(split_prob, 1).item()]
 
            comp_split = (parent_clade_bitarr ^ bitarray(split)).to01()
            
            c1 = node.add_child()
            c2 = node.add_child()
            if split.count('1') > 1:
                node_split_stack.append((c1, comp_split + split))
            else:
                c1.name = self.taxa[split.find('1')]
                c1.clade_bitarr = bitarray(split)
                c1.split_bitarr = min([c1.clade_bitarr, ~c1.clade_bitarr]).to01()
            if comp_split.count('1') > 1:
                node_split_stack.append((c2, split + comp_split))
            else:
                c2.name = self.taxa[comp_split.find('1')]
                c2.clade_bitarr = bitarray(comp_split)
                c2.split_bitarr = min([c2.clade_bitarr, ~c2.clade_bitarr]).to01()
        
        if not rooted:
            root.unroot()
        
        return root
    
    def grab_subsplit_idxes(self, tree):
        subsplit_idxes_list = []
        for node in tree.traverse("postorder"):
            if node.is_root():
#                 root_bipart_bitarr = node.children[0].split_bitarr
                root_bipart_bitarr = min(child.clade_bitarr for child in node.children)
                if root_bipart_bitarr.to01() not in self.rs_map:
                    subsplit_idxes_list.append(-1)
                else:
                    subsplit_idxes_list.append(self.CPDParser.get_index(root_bipart_bitarr.to01()))
            elif not node.is_leaf():
                comb_parent_bipart_bitarr = node.get_sisters()[0].clade_bitarr + node.clade_bitarr
                child_bipart_bitarr = min(child.clade_bitarr for child in node.children)
                ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), child_bipart_bitarr.to01()
                self.subsplit_idxes_update(subsplit_idxes_list, ss_parent, ss_child)
        
        return subsplit_idxes_list
    
    def loglikelihood(self, tree, no_clade_bitarr=True):
        copy_tree = tree.copy()
        if no_clade_bitarr:
            for node in copy_tree.traverse("postorder"):
                if not node.is_root():
                    node.clade_bitarr = self.toBitArr.from_clade(node.get_leaf_names())
        
        with torch.no_grad():
            logprob = self.forward(copy_tree)
        return logprob.item()
    
    def forward(self, tree, return_idxes_list=False, cached_subsplit_info=None):
        if cached_subsplit_info is None:
            subsplit_idxes_list = self.grab_subsplit_idxes(tree)
        else:
            subsplit_idxes_list = cached_subsplit_info
        
        CPDs = torch.cat((self.CPDs, torch.tensor([1.0, 0.0])))
        mapped_idxes_list = torch.LongTensor(subsplit_idxes_list)
        
        if not return_idxes_list:
            return CPDs[mapped_idxes_list].clamp(1e-06).log().sum()
        else:
            return CPDs[mapped_idxes_list].clamp(1e-06).log().sum(), subsplit_idxes_list             
