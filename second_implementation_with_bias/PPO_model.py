import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import hyperparameters as param
import os

class Policy(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.layer_1 = nn.Linear(in_features= state_space, out_features= (state_space + 3))
        self.layer_2 = nn.Linear(in_features= (state_space + 3), out_features= (state_space + 4))
        self.layer_3 = nn.Linear(in_features=(state_space +4), out_features=36)
        self.action = nn.Linear(in_features=36, out_features=36)

    def forward(self,input):
        input = F.relu(self.layer_1(input))
        input = F.relu(self.layer_2(input))
        input = F.relu(self.layer_3(input))
        return self.action(input)

class A_Critic(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.layer_1 = nn.Linear(in_features= state_space + 1, out_features= (state_space - 3))
        self.layer_2 = nn.Linear(in_features= (state_space - 3), out_features= (state_space - 10))
        self.layer_3 = nn.Linear(in_features= (state_space - 10), out_features= 1)

    def forward(self,input):
        input = F.relu(self.layer_1(input))
        input = F.relu(self.layer_2(input))
        return self.layer_3(input)

VPG_mu = Policy(param.feature_length)
VPG_sigma = Policy(param.feature_length)
Critic = A_Critic(param.feature_length)
Velocity = Policy(param.feature_length)

mu_optimizer = torch.optim.Adam(VPG_mu.parameters(),lr=param.VPG_mu_learning_rate) 
sigma_optimizer = torch.optim.Adam(VPG_sigma.parameters(),lr=param.VPG_sigma_learning_rate)
critic_optimizer = torch.optim.Adam(Critic.parameters(),lr=param.Critic_lr)

list_of_file = ["mean_model.pt", "st_dev_model.pt", "critic.pt"]
list_of_load_files = [VPG_mu, VPG_sigma, Critic]

for i,j in zip(list_of_file,list_of_load_files):
    if os.path.exists(i):
        print("loaded")
        j.load_state_dict(torch.load(i))

def save_model():
    torch.save(VPG_mu.state_dict(), "mean_model.pt")
    torch.save(VPG_sigma.state_dict(), "st_dev_model.pt")
    torch.save(Critic.state_dict(), "critic.pt")

def get_dist_and_action(input, mu=VPG_mu, sigma=VPG_sigma, velocity = Velocity):
    mean = mu(input)
    speed = velocity(input)
    std = torch.exp(sigma(input)) 
    dist = torch.distributions.normal.Normal(loc=mean, scale=std)
    action = dist.sample()
    return dist, action, speed 

def log_prob_and_tau(action, dist):
    return -1 * dist.log_prob(action)

def training(batch_of_tregactory,state_value, mu_opt = mu_optimizer, sig_opt = sigma_optimizer, critic=critic_optimizer):
    mu_opt.zero_grad()
    sig_opt.zero_grad()
    critic.zero_grad()

    result = batch_of_tregactory + state_value
    result.backward(torch.tensor([1 for _ in range(0,36)]))
    print(f"gradient of VPG ", VPG_mu.layer_3.weight.grad[0])
    print(f"gradient of critic ", Critic.layer_3.weight.grad[0])

    mu_opt.step()
    sig_opt.step()
    critic.step()




