import pybullet as p
import pybullet_data
from time import sleep
import hyperparameters as param
import PPO_model
import torch
from reward_tuning import reward as rt

physics_client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*param.gravity)

plane = p.loadURDF(param.plane)
robot = p.loadURDF(param.urdf_model, param.spawn_location,
                   p.getQuaternionFromEuler(param.spawn_pitch),flags= p.URDF_USE_SELF_COLLISION)

num = range(p.getNumJoints(robot))
joint_array, feature_length = list(num), len(num)

def get_states_and_contact(robot_id=robot, plane_id=plane, joint_id=joint_array):
    raw_states = p.getJointStates(robot_id, jointIndices = joint_id)
    joint_states = [x[0] for x in raw_states]
    raw_contact = (p.getContactPoints(robot_id,plane_id, x) for x in joint_array)
    joint_contacts = [(1 if x!=() else 0) for x in raw_contact]
    return torch.tensor(joint_states + joint_contacts)

def empty_list_and_tensor():
    return [], torch.tensor([],requires_grad=True),\
           torch.zeros(size=(1,15),requires_grad=True)

head_Z_coord = lambda: p.getLinkState(robot,2)[0][2]
rt.set_threshold(head_Z_coord())
input_tensor = get_states_and_contact()
non_rtg_batch, state_value_batch, batch = empty_list_and_tensor()
final_state_value = torch.tensor([],requires_grad=True)

def reset_robot(robot):
    p.removeBody(robot)
    return p.loadURDF(param.urdf_model, param.spawn_location,p.getQuaternionFromEuler(param.spawn_pitch),
                   flags= p.URDF_USE_SELF_COLLISION)

def return_rtg(rtg_batch):
    rtg = torch.zeros(size=(len(rtg_batch),))
    for i in rtg_batch[::-1]:
        rtg[i]= i + (rtg[i+1] if i+1<len(rtg_batch) else 0)
    return rtg.unsqueeze(1)

for a in range(param.epoch):
    for b in range(param.batch_size):
        for i in range(param.simulation_step_number):

            p.stepSimulation()

            dist, action = PPO_model.get_dist_and_action(input_tensor)
            state_value = PPO_model.Critic(input_tensor)
            state_value_batch = torch.cat((state_value_batch,state_value),0)

            p.setJointMotorControlArray(robot,joint_array,p.POSITION_CONTROL, action)

            non_rtg_batch.append(rt(head_Z_coord(), p.getContactPoints(robot,robot), i))

            batch = torch.cat((batch, PPO_model.log_prob_and_tau(action,dist).unsqueeze(0)),0)
            input_tensor = get_states_and_contact()
            sleep(param.simulation_speed)

            if i%100==0:
                print(f"epoch=> {a}, and loop {i}")

        rtg = return_rtg(non_rtg_batch) 
        advantage = rtg - state_value_batch
        rtg_log_prob = advantage * batch[1:].sum(1) 

        state_value_loss = torch.nn.MSELoss()(rtg.sum(),state_value_batch.sum())
        final_state_value = torch.cat((final_state_value, state_value_loss.unsqueeze(0)),0)

        robot = reset_robot(robot)
        rt.reset()
        non_rtg_batch, state_value_batch, batch = empty_list_and_tensor()

    PPO_model.training(rtg_log_prob.mean(), final_state_value.mean())
    PPO_model.save_model()
    final_state_value = torch.tensor([],requires_grad=True)


p.disconnect()


