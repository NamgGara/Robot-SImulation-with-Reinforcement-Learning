import torch 
from pybullet import getJointStates
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim
import torch.utils.data
from simulation import robot, joint_array

""" 
the model should be actor critic

to do:
what distribution to use for policy and how to select actions
recieves states of all joints and produces position for each joint

critic should be DQN
the network should produce value function of states
"""

joint_states= getJointStates(robot, joint_array)
feature_length = len(joint_array)


def reward_lock(progress,threshold):
    if progress > threshold:
        return 1, (new_threshold:=progress)

class ActorC:
    def __init__(self):
        pass

    def forward(self,joint_list):

        #logic

        return 

class DQN:
    def __init__(self):
        self.dense_1 = nn.Linear(in_features= feature_length, out_features= feature_length + 4)
        self.dense_2 = nn.Linear(in_features=feature_length + 4, out_features=feature_length-6)
        self.final_dense = nn.Linear(in_features=feature_length-6, out_features=1)
        
rl_model = ActorC()

for i in range(10):
    print(joint_states)

