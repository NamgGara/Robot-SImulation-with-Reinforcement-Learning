from xml.sax.handler import feature_external_ges
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import hyperparameters

# first test with VPG, not PPO
class VPG(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.layer_1 = nn.Linear(in_features= state_space, out_features= (state_space + 3))
        self.layer_2 = nn.Linear(in_features= (state_space + 3), out_features= (state_space + 4))
        self.layer_3 = nn.Linear(in_features= (state_space + 4), out_features= action_space)

    def forward(self,input):
        input = F.relu(self.layer_1(input))
        input = F.relu(self.layer_2(input))
        return self.layer_3(input)

# class PPO(nn.Module):
#     def __init__(self):
#         super().__init__()

VPG_mu = VPG(hyperparameters.feature_length, hyperparameters.action_space)
VPG_sigma = VPG(hyperparameters.feature_length, hyperparameters.action_space)

def policy_distribution_and_action(input, mu=VPG_mu, sigma=VPG_sigma):
    mean = mu(input)
    std = torch.exp(sigma(input))
    dist = torch.distributions.Normal(loc=mean, scale=std)

    print()
    print("gradient   ___", VPG_mu.layer_1.weight.grad)
    
    return dist, dist.sample()

# def return_and_backward(action, dist):
#     J_gradient = -1 * dist.log_prob(action)

return_and_backward = lambda action, dist: -1 * dist.log_prob(action)

