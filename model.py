import torch 
from pybullet import getJointStates
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
    def __init__(self):
        super().__init__()
        

    """ 
    for actor critic, make a policy 
    take in states through a nn, produce a distribution (binomial) for each joint
    it can be two arrays of size joint x 1, 
    sample from binomial for action,

    """

    def forward(self,joint_list):


        return 

def reward(progress,threshold):
    if progress > threshold:
        return 10, (new_threshold:=progress)
    else:
        return 0, threshold


