import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FixedRateModel(nn.Module):
    def __init__(self, init_clock_rate=1.0, **kwargs):
        super().__init__()
        self.log_clock_rate = torch.tensor(math.log(init_clock_rate))
    
    def sample(self, **kwargs):
        return self.log_clock_rate, 0.0
    
    def forward(self, *args):
        return 0.0


class StrictModel(nn.Module):
    def __init__(self, init_clock_rate=1.0, mu=0.0, sigma=3.0, **kwargs):
        super().__init__()
        self.mu, self.sigma = mu, sigma
        self.log_clock_rate_offset = math.log(init_clock_rate)
        
        self.clock_rate_param = nn.Parameter(torch.zeros(2,1))
        
    def sample(self, n_particles=1, log_tree_height=0.0):
        mean, std = self.clock_rate_param[0] + self.log_clock_rate_offset, self.clock_rate_param[1]
        sample_epsilon = torch.randn(n_particles, mean.size(-1))
        log_clock_rate = mean + std.exp() * sample_epsilon - log_tree_height
        logq_clock_rate = torch.sum(-0.5*math.log(2*math.pi) - std - 0.5*sample_epsilon**2, dim=-1)

        return log_clock_rate, logq_clock_rate
        
    def forward(self, log_clock_rate):
        return torch.sum(-0.5*math.log(2*math.pi) - math.log(self.sigma) - 0.5*((log_clock_rate-self.mu)/self.sigma)**2, dim=-1)