import pybullet as p
import pybullet_data
from time import sleep
import hyperparameters
import PPO_model
import torch

# pybullet boilerplate
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*hyperparameters.gravity)

plane = p.loadURDF(hyperparameters.plane)
robot = p.loadURDF(hyperparameters.urdf_model, hyperparameters.spawn_point,
                   p.getQuaternionFromEuler(hyperparameters.spawn_pitch))
joint_array, feature_length = list(range(p.getNumJoints(robot))),  len(range(p.getNumJoints(robot)))

def get_states_and_contact(robot_id=robot, plane_id=plane, joint_id=joint_array):
    raw_states = p.getJointStates(robot_id, jointIndices = joint_id)
    raw_contact = (p.getContactPoints(robot_id,plane_id, x) for x in joint_array)
    joint_states = [x[0] for x in raw_states]
    joint_contacts = [(1 if x!=() else 0) for x in raw_contact]
    return torch.tensor(joint_states + joint_contacts)

input = get_states_and_contact()

for i in range(hyperparameters.simualtion_step):

    p.stepSimulation()

    sleep(hyperparameters.simulation_speed)

    if i%200 ==0:
        dist, action = PPO_model.policy_distribution_and_action(input)
        print(dist)
        print(action)

    input = get_states_and_contact()
p.disconnect()
