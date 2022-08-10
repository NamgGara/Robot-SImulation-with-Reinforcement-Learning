import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import hyperparameters

# first test with VPG, not PPO
class VPG(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.layer_1 = nn.Linear(in_features= state_space, out_features= (state_space -3))
        self.layer_2 = nn.Linear(in_features= (state_space -3), out_features= (state_space - 8))
        self.layer_3 = nn.Linear(in_features= (state_space - 8), out_features= 1)

    def forward(self,input):
        input = F.relu(self.layer_1(input))
        input = F.relu(self.layer_2(input))
        return self.layer_3(input)

# class PPO(nn.Module):
#     def __init__(self):
#         super().__init__()
from hyperparameters import feature_length

VPG_mu = VPG(feature_length)
VPG_sigma = VPG(feature_length)

