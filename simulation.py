import pybullet as p
import pybullet_data
import random
from time import sleep
import model
from torch import save as _save

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)

plane = p.loadURDF("plane.urdf")

spawn_point = [0,0,3]
spawn_pitch = p.getQuaternionFromEuler([0,0,0])
robot = p.loadURDF("humanoid.urdf", spawn_point, spawn_pitch)

joint_array = range(p.getNumJoints(robot))
feature_length = len(joint_array)

# from model.py
rl_model =model.ActorC()
DQN_old = model.DQN()
DQN_new = model.DQN()
_save(DQN_old.state_dict(),"old_parameters")


if __name__ == "__main__":
    for i in range(10000):
        p.stepSimulation()

        #motor control test 

        articulation = [1 for i in joint_array]
        p.setJointMotorControlArray(robot, joint_array, p.POSITION_CONTROL, articulation, [1.0 for i in joint_array])

        joint_states = p.getJointStates(robot, joint_array)
        sleep(1./240.)

    p.disconnect()