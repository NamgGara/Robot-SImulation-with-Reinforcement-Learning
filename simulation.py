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
robot = p.loadURDF(hyperparameters.urdf_model, hyperparameters.spawn_location,
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
    return p.loadURDF(hyperparameters.urdf_model, hyperparameters.spawn_location,
                   p.getQuaternionFromEuler(hyperparameters.spawn_pitch),
                   flags= p.URDF_USE_SELF_COLLISION) , 0

input_tensor = get_states_and_contact()
batch2=torch.zeros(size=(1,15))
rt.set_threshold(head_Z_coord())

C_reward_rtg = []
final_batch_summation = torch.tensor([],requires_grad=True)
state_value_batch = torch.tensor([],requires_grad=True)

def return_rtg_log_prob(C_reward_rtg):
    rtg = torch.zeros(size=(len(C_reward_rtg),))
    for i in C_reward_rtg[::-1]:
        rtg[i]= i + (rtg[i+1] if i+1<len(C_reward_rtg) else 0)

    return rtg.unsqueeze(1)

for a in range(hyperparameters.epoch):
    for b in range(hyperparameters.batch_size):
        for num, i in enumerate(range(hyperparameters.simulation_step_number)):

            p.stepSimulation()

            dist, action = PPO_model.get_dist_and_action(input_tensor)

            #trying critic
            state_value = PPO_model.Critic(input_tensor)
            state_value_batch = torch.cat((state_value_batch,state_value),0)

            p.setJointMotorControlArray(robot,joint_array,p.POSITION_CONTROL, action)

            #rtg
            C_reward_rtg.append(rt(head_Z_coord(), p.getContactPoints(robot,robot), i))

            batch2 = torch.cat((batch2, PPO_model.log_prob_and_tau(action,dist).unsqueeze(0)),0)
            input_tensor = get_states_and_contact()
            sleep(hyperparameters.simulation_speed)

            if i%100==0:
                print(f"epoch=> {a}, and loop {i}")

        #rtg attempt

        rtg = return_rtg_log_prob(C_reward_rtg) 
        advantage = rtg - state_value_batch
        rtg_log_prob = batch2[1:].sum(1) * advantage

        state_value_loss = torch.nn.MSELoss()(rtg.sum(),state_value_batch.sum())
        robot, c_reward = reset_robot(robot)
        batch2=torch.zeros(size=(1,15),requires_grad=True)

        C_reward_rtg = []
        state_value_batch = torch.tensor([],requires_grad=True)

        rt.reset()
    
    PPO_model.training2(rtg_log_prob.mean(), state_value_loss)

    PPO_model.save_model()

p.disconnect()


