import pybullet as p
import pybullet_data
from time import sleep
import hyperparameters
import PPO_model
import torch
from reward_tuning import reward as rt

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
                   flags= p.URDF_USE_SELF_COLLISION) , 0

input_tensor = get_states_and_contact()

batch = torch.tensor([])

rt.set_threshold(head_Z_coord())
c_reward = 0
tau = torch.tensor([0.])
for a in range(hyperparameters.epoch):
    for b in range(hyperparameters.batch):
        for i in range(hyperparameters.simualtion_step):

            p.stepSimulation()

            dist, action = PPO_model.get_dist_and_action(input_tensor)

            p.setJointMotorControlArray(robot,joint_array,p.POSITION_CONTROL, action)

            c_reward += rt(head_Z_coord(), p.getContactPoints(robot,robot), i)

            batch = torch.cat((batch, PPO_model.log_prob_and_tau(action,dist)), 0)

            input_tensor = get_states_and_contact()
            sleep(hyperparameters.simulation_speed)

            if i%10==0:
                print(f"epoch=> {a}, and loop {i}")
        # change epoch back to 1000 
        tau += tau + (1/10) * (batch.sum() - tau)
        print(tau)
        batch = torch.tensor([])
        robot, c_reward = reset_robot(robot)
        rt.reset()

    print(f"progress was {rt.threshold}")
    PPO_model.training(tau, c_reward)
    tau = torch.tensor([0.])
    PPO_model.save_model()

p.disconnect()


