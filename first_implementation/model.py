import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim
import torch.utils.data
import os

class Model_HyperParameters:
    discount = 0.9
    t_reward = -1
    c_reward = 0
    reward = 2

class DQN(nn.Module):
    def __init__(self,feature_length):
        super().__init__()
        self.dense_1 = nn.Linear(in_features= feature_length, out_features= feature_length + 4)
        self.dense_2 = nn.Linear(in_features=feature_length + 4, out_features=feature_length-6)
        self.final_dense = nn.Linear(in_features=feature_length-6, out_features=1)
        
    def forward(self,states):
        result = nn.ReLU()(self.dense_1(states))
        result = nn.ReLU()(self.dense_2(result))
        return self.final_dense(result)
    
class ActorC(nn.Module):
    def __init__(self, feature_length):
        super().__init__()
        self.dense_1 = nn.Linear(in_features=feature_length, out_features=20)
        self.dense_2 = nn.Linear(in_features=20, out_features=18)
        self.final_mean = nn.Linear(in_features=18, out_features=feature_length)
        self.final_std = nn.Linear(in_features=18, out_features=feature_length)

    def forward(self,joint_list):

        result = nn.ReLU()(self.dense_1(joint_list))
        result = nn.ReLU()(self.dense_2(result))

        # residual blocks
        mean = self.final_mean(result) + joint_list
        std = self.final_std(result) + joint_list
        return torch.distributions.normal.Normal(mean,torch.log(std))

def reward(progress,threshold):
    if progress > threshold:
        return 2, progress
    else:
        return 0.0, threshold

def model_construction(value):

    with torch.cuda.device(0):
        actor_model = ActorC(feature_length=value)
        DQN_target = DQN(feature_length=value).requires_grad_(requires_grad=False)
        DQN_new = DQN(feature_length=value)
        critic = DQN(feature_length=value)

    return actor_model,DQN_target,DQN_new,critic

def model_loading(file_path, list_of_model, list_of_file_path):
    if os.path.exists(file_path):
        for nn_model,saving_path in zip(list_of_model,list_of_file_path):
            nn_model.load_state_dict(torch.load(saving_path))
            nn_model.train()
