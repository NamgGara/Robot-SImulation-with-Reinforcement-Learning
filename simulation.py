import pybullet as p
import pybullet_data
from time import sleep
import hyperparameters as param
import PPO_model
import torch
from reward_tuning import reward as rt
from graph import graph

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*param.gravity)

plane = p.loadURDF(param.plane)
robot = p.loadURDF(param.urdf_model, param.spawn_location,
                   p.getQuaternionFromEuler(param.spawn_pitch),flags= p.URDF_USE_SELF_COLLISION)
num = range(p.getNumJoints(robot))

joint_array,feature_length = list(num),len(num)

revolute = [4,7,10,13]
spherical = [1,2,3,6,9,11,12,14]

def get_states_and_contact(robot_id=robot, plane_id=plane, joint_id=joint_array):
    raw_states = p.getJointStates(robot_id, jointIndices = joint_id)
    joint_states = [x[0] for x in raw_states]
    raw_contact = (p.getContactPoints(robot_id,plane_id, x) for x in joint_array)
    joint_contacts = [(1 if x!=() else 0) for x in raw_contact]
    return torch.tensor(joint_states + joint_contacts)

def empty_list_and_tensor():
    return [], torch.tensor([],requires_grad=True),torch.tensor([],requires_grad=True)

def final_empyt_tensor():
    return torch.tensor([],requires_grad=True), torch.tensor([],requires_grad=True)

head_Z_coord = lambda: p.getLinkState(robot,2)[0][2]
rt.set_threshold(head_Z_coord())
input_tensor = get_states_and_contact()
final_state_value = torch.tensor([],requires_grad=True)

def reset_robot(robot):
    p.removeBody(robot)
    return p.loadURDF(param.urdf_model, param.spawn_location,p.getQuaternionFromEuler(param.spawn_pitch),
                   flags= p.URDF_USE_SELF_COLLISION)

def return_rtg(rtg_batch):
    rtg = torch.zeros(size=(len(rtg_batch),))

    for j,i in zip(rtg_batch[::-1],range(len(rtg_batch)-1,-1,-1)):
        rtg[i]= j + (rtg[i+1] if i+1<len(rtg_batch) else 0)
    return rtg

def cat_input_and_time_step(input, time_step):
    return torch.cat((input, torch.tensor([time_step])),0)

for a in range(param.epoch):
    final_state_value, final_policy = final_empyt_tensor()

    for b in range(param.batch_size):

        robot = reset_robot(robot)
        rt.reset()
        complete_reward, state_value_batch, policy_batch = empty_list_and_tensor()

        for i in range(param.simulation_step_number):
            
            if i%100==0:
                print(f"epoch=> {a}, and loop {i}")

            p.stepSimulation()
            
            dist, action, joint_speed = PPO_model.get_dist_and_action(input_tensor)

            p.setJointMotorControlArray(robot, revolute,p.POSITION_CONTROL, action[0:4], joint_speed[0:4])
            p.setJointMotorControlMultiDofArray(robot, spherical, p.POSITION_CONTROL, action[4:36].reshape(shape=(8,4)),  joint_speed[4:].reshape(shape=(8,4)))

            policy_batch = torch.cat((policy_batch, PPO_model.log_prob_and_tau(action,dist).unsqueeze(0)),0)
            
            state_value = PPO_model.Critic(cat_input_and_time_step(input_tensor, i))
            state_value_batch = torch.cat((state_value_batch,state_value),0)
            complete_reward.append(rt(head_Z_coord(), p.getContactPoints(robot,robot), i))
            
            input_tensor = get_states_and_contact()
            sleep(param.simulation_speed)
            
        rtg = return_rtg(complete_reward) 
        advantage = rtg - state_value_batch
        graph(rtg[0])
        
        rtg_log_prob = (advantage.unsqueeze(0) * policy_batch.T).T
        final_policy = torch.cat((final_policy,rtg_log_prob.sum(0).unsqueeze(0)),0)
        state_value_loss = torch.nn.MSELoss()(rtg,state_value_batch)
        final_state_value = torch.cat((final_state_value, state_value_loss.unsqueeze(0)),0)
    
    PPO_model.training(final_policy.mean(0), final_state_value.mean())
    PPO_model.save_model()
    graph.graph()
p.disconnect()


