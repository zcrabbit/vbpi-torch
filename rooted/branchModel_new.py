import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaseModel(nn.Module):
    def __init__(self, ntips, rootsplit_embedding_map, subsplit_embedding_map, feature_dim=2, sample_info=None,
                 psp=True, root_height_offset=0.0):
        super().__init__()
        self.ntips = ntips
        self.embedding_map = rootsplit_embedding_map.copy()
        self.feature_dim = feature_dim
        if sample_info is not None:
            self.sample_info = torch.tensor(sample_info)
        else:
            self.sample_info = torch.zeros(self.ntips)
        self.psp = psp
        self.root_height_offset = root_height_offset

        self.embedding_dim = len(self.embedding_map)
        for parent in subsplit_embedding_map:
            parent_clade = parent[self.ntips:]
            if parent_clade not in self.embedding_map:
                self.embedding_map[parent_clade] = self.embedding_dim
                self.embedding_dim += 1
            if self.psp:
                for child, i in subsplit_embedding_map[parent].items():
                    if parent_clade + child not in self.embedding_map:
                        self.embedding_map[parent_clade+child] = self.embedding_dim
                        self.embedding_dim += 1
                    self.embedding_map[parent+child] = self.embedding_dim
                    self.embedding_dim += 1
        
        self.T_alpha = nn.Parameter(torch.zeros(self.embedding_dim, self.feature_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.T_alpha.data)
        self.padding_dim = -1
    
    def pad_feature(self):
        self.feature_padded = torch.cat((self.T_alpha, torch.zeros(1, self.feature_dim)), dim=0)
    
    def parse_sample_info(self, tree, leaf_only=False):
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.height_ = self.sample_info[node.name]
            elif not leaf_only:
                node.height_ = max(child.height_ for child in node.children)
    
    def grab_subsplit_idxes(self, tree):
        subsplit_idxes_list = []
        idx_list = []
        
        for node in tree.traverse("postorder"):
            if node.is_root():
                root_bipart_bitarr = min(child.clade_bitarr for child in node.children)
                if self.psp:    
                    subsplit_idxes_list.append([self.embedding_map[root_bipart_bitarr.to01()], self.padding_dim, self.padding_dim])
                else:
                    subsplit_idxes_list.append([self.embedding_map[root_bipart_bitarr.to01()]])
                idx_list.append(node.name)
            elif not node.is_leaf():
                comb_parent_bipart_bitarr = node.get_sisters()[0].clade_bitarr + node.clade_bitarr
                child_bipart_bitarr = min(child.clade_bitarr for child in node.children)
                if self.psp:
                    subsplit_idxes_list.append([self.embedding_map[node.clade_bitarr.to01()], self.embedding_map[node.clade_bitarr.to01() + child_bipart_bitarr.to01()], self.embedding_map[comb_parent_bipart_bitarr.to01() + child_bipart_bitarr.to01()]])
                else:
                    subsplit_idxes_list.append([self.embedding_map[node.clade_bitarr.to01()]])
                idx_list.append(node.name)
        
        return subsplit_idxes_list, idx_list
    
    def mean_std(self, tree, return_adj_matrix=False, cached_adj_info=None):
        if cached_adj_info is None:
            subsplit_idxes_list, idx_list = self.grab_subsplit_idxes(tree)
        else:
            subsplit_idxes_list, idx_list = cached_adj_info
        
        subsplit_idxes_list = torch.LongTensor(subsplit_idxes_list)
        branch_idx_map = torch.sort(torch.LongTensor(idx_list), dim=0, descending=False)[1]

        # mean_std = torch.index_select(self.T_alpha[subsplit_idxes_list].sum(1), 0, branch_idx_map)
        mean_std = torch.index_select(self.feature_padded[subsplit_idxes_list].sum(1), 0, branch_idx_map)
        
        if not return_adj_matrix:
            return mean_std[:, 0], mean_std[:, 1]
        else:
            return mean_std[:, 0], mean_std[:, 1], subsplit_idxes_list[branch_idx_map]
    
    def sample_T_alpha_base(self, n_particles):
        samp_log_T_alpha = torch.randn(n_particles, self.ntips-1)
        return samp_log_T_alpha, torch.sum(-0.5*math.log(2*math.pi) - 0.5*samp_log_T_alpha**2, -1)
    
    def sample_T_alpha(self, tree_list, return_adj_matrix=False):
        if not return_adj_matrix:
            mean, std = zip(*map(lambda x: self.mean_std(x), tree_list))
            mean, std = torch.stack(mean, dim=0), torch.stack(std, dim=0)
        else:
            mean, std, subsplit_idxes_list = zip(*map(lambda x: self.mean_std(x, return_adj_matrix=True), tree_list))
            mean, std, subsplit_idxes_list = torch.stack(mean, dim=0), torch.stack(std, dim=0), torch.stack(subsplit_idxes_list, dim=0)

        samp_log_T_alpha, logq_T_alpha = self.sample_T_alpha_base(len(tree_list))
        samp_log_T_alpha, logq_T_alpha = samp_log_T_alpha * std.exp() + mean, logq_T_alpha - torch.sum(std, -1)

        if not return_adj_matrix:
            return samp_log_T_alpha, logq_T_alpha, None
        else:
            return samp_log_T_alpha, logq_T_alpha, subsplit_idxes_list
    
    def branch_reparam_coalescent_info(self, tree, alpha, T, return_event_idx=False, sample_info=False):
        branch, height, event_info = [], [], []
        idx_list = []
        rescale_factor = []
        
        if not sample_info:
            self.parse_sample_info(tree)
        for node in tree.traverse("preorder"):
            if node.is_root():
                node.height = T + node.height_
                height.append(node.height)
                event_info.append(1)
            else:
                if not node.is_leaf():
                    rescale_factor.append(node.up.height - node.height_)
                    branch.append(alpha[node.name] * rescale_factor[-1])
                    node.height = node.up.height - branch[-1]
                    height.append(node.height)                   
                    event_info.append(1)
                else:
                    branch.append(node.up.height - node.height_)
                    height.append(node.height_)
                    event_info.append(-1)

                idx_list.append(node.name)
        
        branch = torch.stack(branch)
        branch_idx_map = torch.sort(torch.LongTensor(idx_list), dim=0, descending=False)[1]
        
        if not return_event_idx:
            return branch[branch_idx_map], torch.stack(rescale_factor), torch.stack(height), torch.FloatTensor(event_info)
        else:
            idx_list.insert(0, 2*self.ntips-2)
            return branch[branch_idx_map], torch.stack(rescale_factor), torch.stack(height), torch.FloatTensor(event_info), torch.LongTensor(idx_list)
    
    def forward(self, tree_list, return_idx=False):
        self.pad_feature()
        samp_log_T_alpha, logq_T_alpha, subsplit_idxes_list = self.sample_T_alpha(tree_list, return_idx)
        alpha_, log_T = samp_log_T_alpha[:, :-1] - 2.0, samp_log_T_alpha[:, -1] + self.root_height_offset
        alpha_vec = torch.sigmoid(alpha_)
        alpha = torch.cat((torch.zeros(len(tree_list), self.ntips), alpha_vec), dim=-1)
        T = log_T.exp()
        logq_T_alpha -= torch.sum(torch.log(alpha_vec*(1.-alpha_vec)), dim=-1) + log_T

        if not return_idx:
            raw_branch, rescale_factor, height, event_info = zip(*[self.branch_reparam_coalescent_info(tree_list[i], alpha[i], T[i]) for i in range(len(tree_list))])
            raw_branch, rescale_factor, height, event_info = torch.stack(raw_branch), torch.stack(rescale_factor), torch.stack(height), torch.stack(event_info)
        else:
            raw_branch, rescale_factor, height, event_info, event_idxes = zip(*[self.branch_reparam_coalescent_info(tree_list[i], alpha[i], T[i], True) for i in range(len(tree_list))])
            raw_branch, rescale_factor, height, event_info, event_idxes = torch.stack(raw_branch), torch.stack(rescale_factor), torch.stack(height), torch.stack(event_info), torch.stack(event_idxes)

        logq_height = logq_T_alpha - torch.sum(torch.log(rescale_factor), dim=-1)

        if not return_idx:
            return raw_branch, logq_height, height, event_info 
        else:
            return raw_branch, logq_height, height, event_info, event_idxes, subsplit_idxes_list
