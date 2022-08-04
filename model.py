import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim
import torch.utils.data

class Model_HyperParameters:
    discount = 0.9
    t_reward = -1
    c_reward = 0

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
        self.dense1 = nn.Linear(in_features=feature_length, out_features=40)
        self.dense2 = nn.Linear(in_features=40, out_features=30)
        self.final_mean = nn.Linear(in_features=30, out_features=feature_length)
        self.final_std = nn.Linear(in_features=30, out_features=feature_length)

    def forward(self,joint_list):

        result = nn.ReLU()(self.dense1(joint_list))
        result = nn.ReLU()(self.dense2(result))
        mean = self.final_mean(result)
        std = self.final_std(result)
        action = [torch.distributions.Normal(x,torch.abs(y)) for x,y in zip(mean,std)]
        return torch.tensor([act.sample() for act in action])

def reward(progress,threshold):
    if progress > threshold:
        return 10.0, (new_threshold:=progress)
    else:
        return 0.0, threshold


