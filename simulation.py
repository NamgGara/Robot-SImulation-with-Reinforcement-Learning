import os
import pybullet as p
import pybullet_data
import random
from time import sleep
import torch
import model

import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
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
        result = F.relu(self.dense_1(states))
        result = F.relu(self.dense_2(result))
        # print(self.final_dense(result))
        return self.final_dense(result)
    

class ActorC(nn.Module):
    def __init__(self, feature_length):
        super().__init__()
        self.dense1 = nn.Linear(in_features=feature_length, out_features=40)
        self.dense2 = nn.Linear(in_features=40, out_features=30)
        self.final_mean = nn.Linear(in_features=30, out_features=feature_length)
        self.final_std = nn.Linear(in_features=30, out_features=feature_length)

    def forward(self,joint_list):

        result = F.relu((self.dense1(joint_list)))
        result = F.relu(self.dense2(result))
        mean = self.final_mean(result)
        std = self.final_std(result)
        action = torch.distributions.Normal(loc=mean, scale=std)
        return action.sample()

def reward(progress,threshold):
    if progress > threshold:
        return 10, (new_threshold:=progress)
    else:
        return 0, threshold
        

hyper_parameters = Model_HyperParameters()
robot_head = 2
gravity = [0,0,-9.9]
spawn_point = [0,0,3]
spawn_pitch = p.getQuaternionFromEuler([0,0,0])
urdf_model = "humanoid.urdf"
learning_rate_DQN = 0.05
save_path = "new_parameters.pt"

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*gravity)
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(urdf_model, spawn_point, spawn_pitch)
joint_array = range(p.getNumJoints(robot))
feature_length = len(joint_array)

rl_model = ActorC(feature_length=feature_length)
DQN_old = DQN(feature_length=feature_length)
DQN_new = DQN(feature_length=feature_length)

if os.path.exists(save_path):
    for i in [DQN_old, DQN_new]:
        i.load_state_dict(torch.load(save_path))
        i.train()

joint_states = p.getJointStates(robot, joint_array)
old_states_as_tensors = torch.tensor([joint[0] for joint in joint_states])
old_head_coord = p.getLinkState(robot,robot_head)[0]
threshold_cord = old_head_coord[2]

optimizer = torch.optim.SGD(DQN_new.parameters(),lr=learning_rate_DQN)
if __name__ == "__main__":
    for i in range(10000):
        optimizer.zero_grad()
        if i%100 == 0:
            DQN_old.load_state_dict(DQN_new.state_dict())
            torch.save(DQN_new.state_dict(),save_path)

        articulation = [1 for i in joint_array]
        
        p.stepSimulation()

        #motor control test 
        
        p.setJointMotorControlArray(robot, joint_array, p.POSITION_CONTROL, articulation, [1.0 for i in joint_array])
        
        joint_states = p.getJointStates(robot, joint_array)
        new_states_as_tensors = torch.tensor([joint[0] for joint in joint_states])
        new_head_coord = p.getLinkState(robot,robot_head)[0][2]

        with torch.no_grad():
            old_target = DQN_old(new_states_as_tensors)
        reward, threshold_cord = model.reward(new_head_coord, threshold_cord)
        hyper_parameters.c_reward += reward + hyper_parameters.t_reward

        delta = torch.nn.MSELoss()(torch.tensor(reward) + (hyper_parameters.discount * old_target), DQN_new(old_states_as_tensors))
        delta.backward()
        #ERROR 
        # print(DQN_new.dense_1.weight.grad)
        optimizer.step()
        old_states_as_tensors = new_states_as_tensors

        sleep(1./150.)
    p.disconnect()