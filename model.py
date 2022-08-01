import torch 
from pybullet import getJointStates
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim
import torch.utils.data
from simulation import robot, joint_array

joint_states= getJointStates(robot, joint_array)

class RL:
    def __init__(self):
        pass

    def forward(self,joint_list):

        #logic

        return 

class NN:
    def __init__(self) -> None:
        pass

rl_model = RL()
for i in range(10):
    print(joint_states)

