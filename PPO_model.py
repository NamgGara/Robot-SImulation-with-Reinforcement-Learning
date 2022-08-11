import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import hyperparameters
import os

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

VPG_mu = VPG(hyperparameters.feature_length, hyperparameters.action_space)
VPG_sigma = VPG(hyperparameters.feature_length, hyperparameters.action_space)

for i,j in zip(["mean_model.pt","st_dev_model.pt"],[VPG_mu, VPG_sigma]):
    if os.path.exists(i):
        j.load_state_dict(torch.load(i))

mu_optimizer, sigma_optimizer = [torch.optim.Adam(x.parameters(),lr=y) for x,y in
                                zip([VPG_mu,VPG_sigma],[hyperparameters.VPG_mu_learning_rate,hyperparameters.VPG_sigma_learning_rate])]

def save_model():
    torch.save(VPG_mu.state_dict(), "mean_model.pt")
    torch.save(VPG_sigma.state_dict(), "st_dev_model.pt")

def get_dist_and_action(input, mu=VPG_mu, sigma=VPG_sigma):
    mean = mu(input)
    std = torch.exp(sigma(input))
    dist = torch.distributions.Normal(loc=mean, scale=std)

    # print()
    # print("gradient   ___", VPG_mu.layer_1.weight.grad)
    
    return dist, dist.sample()

def log_prob_and_tau(action, dist):
    return -1 * dist.log_prob(action)
      #this is like returning an expectation of the tragetory and reward of tregetory

def training(batch,reward, mu_opt = mu_optimizer, sig_opt = sigma_optimizer):
    
    mu_opt.zero_grad()
    sig_opt.zero_grad()

    batch.mean().backward()

    mu_opt.step()
    sig_opt.step()



