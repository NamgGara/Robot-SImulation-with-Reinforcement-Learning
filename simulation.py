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

# from model.py
rl_model = model.ActorC(feature_length=feature_length)
DQN_old = model.DQN(feature_length=feature_length)
DQN_new = model.DQN(feature_length=feature_length)

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