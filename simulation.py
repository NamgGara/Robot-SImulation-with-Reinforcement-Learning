import os
import pybullet as p
import pybullet_data
import random
from time import sleep
import torch
import model

hyper_parameters = model.Model_HyperParameters()
robot_head = 2
gravity = [0,0,-9.9]
spawn_point = [0,0,1]
spawn_pitch = p.getQuaternionFromEuler([0,0,0])
urdf_model = "humanoid.urdf"
learning_rate_DQN = 0.1
save_path = "new_parameters.pt"
str_points = 1
simualtion_step = 1

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*gravity)
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(urdf_model, spawn_point, spawn_pitch)
joint_array = range(p.getNumJoints(robot))
feature_length = len(joint_array)
strength = [str_points for i in joint_array]

# from model.py
ActorC = model.ActorC(feature_length=feature_length)
DQN_old = model.DQN(feature_length=feature_length).requires_grad_(requires_grad=False)
DQN_new = model.DQN(feature_length=feature_length)

# if os.path.exists(save_path):
#     for i in [DQN_old, DQN_new]:
#         i.load_state_dict(torch.load(save_path))
#         i.train()

joint_states = p.getJointStates(robot, joint_array)
old_states_as_tensors = torch.tensor([joint[0] for joint in joint_states])
old_head_coord = p.getLinkState(robot,robot_head)[0]
threshold_cord = old_head_coord[2]

optimizer = torch.optim.SGD(DQN_new.parameters(),lr=learning_rate_DQN)

if __name__ == "__main__":
    for i in range(simualtion_step):
        optimizer.zero_grad()

        if i%100 == 0:
            DQN_old.load_state_dict(DQN_new.state_dict())
            torch.save(DQN_new.state_dict(),save_path)

        p.stepSimulation()

        #motor control 
        articulation = ActorC(old_states_as_tensors)        
        p.setJointMotorControlArray(robot, joint_array, p.POSITION_CONTROL, articulation, strength)
        
        joint_states = p.getJointStates(robot, joint_array)
        new_states_as_tensors = torch.tensor([joint[0] for joint in joint_states])
        new_head_coord = p.getLinkState(robot,robot_head)[0][2]

        old_target = DQN_old(new_states_as_tensors)
        
        reward, threshold_cord = model.reward(new_head_coord, threshold_cord)
        hyper_parameters.c_reward += reward + hyper_parameters.t_reward
        reward = torch.tensor(reward,requires_grad=False)

        delta = torch.nn.MSELoss()(reward + (hyper_parameters.discount * old_target), DQN_new(old_states_as_tensors))

        print("before step, the weights are =",DQN_new.final_dense.weight)

        optimizer.zero_grad()
        delta.backward()
        print("after backward, the grads",DQN_new.final_dense.weight.grad)

        optimizer.step()
        print("after step, the weights are = ",DQN_new.final_dense.weight)

        sleep(1./150.)
        old_states_as_tensors = new_states_as_tensors


    p.disconnect()