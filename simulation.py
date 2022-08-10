import pybullet as p
import pybullet_data
from time import sleep
import hyperparameters
import PPO_model
import torch
import reward_tuning as rt

# pybullet boilerplate
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*hyperparameters.gravity)

plane = p.loadURDF(hyperparameters.plane)
robot = p.loadURDF(hyperparameters.urdf_model, hyperparameters.spawn_point,
                   p.getQuaternionFromEuler(hyperparameters.spawn_pitch),
                   flags= p.URDF_USE_SELF_COLLISION)
joint_array, feature_length = list(range(p.getNumJoints(robot))),  len(range(p.getNumJoints(robot)))

def get_states_and_contact(robot_id=robot, plane_id=plane, joint_id=joint_array):
    raw_states = p.getJointStates(robot_id, jointIndices = joint_id)
    raw_contact = (p.getContactPoints(robot_id,plane_id, x) for x in joint_array)
    joint_states = [x[0] for x in raw_states]
    joint_contacts = [(1 if x!=() else 0) for x in raw_contact]
    return torch.tensor(joint_states + joint_contacts)

def head_Z_coord():
    return p.getLinkState(robot,2)[0][2]

def reset_robot(robot):
    p.removeBody(robot)
    return p.loadURDF(hyperparameters.urdf_model, hyperparameters.spawn_point,
                   p.getQuaternionFromEuler(hyperparameters.spawn_pitch),
                   flags= p.URDF_USE_SELF_COLLISION)

input_tensor = get_states_and_contact()

batch = torch.tensor([])
rt.reward.set_threshold(head_Z_coord())
print(head_Z_coord())

for i in range(hyperparameters.epoch):
    for i in range(hyperparameters.simualtion_step):

        p.stepSimulation()

        dist, action = PPO_model.get_dist_and_action(input_tensor)

        p.setJointMotorControlArray(robot,joint_array,p.POSITION_CONTROL, action)

        reward = rt.reward(head_Z_coord()) + rt.overlapping_punishment(p.getContactPoints(robot,robot))
        batch = torch.cat((batch, PPO_model.log_prob_and_tau(action,dist,reward)), 0)

        input_tensor = get_states_and_contact()
        sleep(hyperparameters.simulation_speed)

        print(p.getContactPoints(robot,robot))
    
    PPO_model.training(batch)
    batch = torch.tensor([])
    robot = reset_robot(robot)
    PPO_model.save_model()
    rt.reward.reset()
    
p.disconnect()
