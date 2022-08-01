import pybullet as p
import pybullet_data
import random
from time import sleep
import model

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)

plane = p.loadURDF("plane.urdf")

spawn_point = [0,0,3]
spawn_pitch = p.getQuaternionFromEuler([0,0,0])
robot = p.loadURDF("humanoid.urdf", spawn_point, spawn_pitch)
joint_array = range(p.getNumJoints(robot))

if __name__== "__main__":
    for i in range(10000):
        p.stepSimulation()

        #motor control test

        articulation = [random.random()*100 for i in joint_array]
        p.setJointMotorControlArray(robot, joint_array, p.POSITION_CONTROL, articulation, [1.0 for i in joint_array])

        joint_states = p.getJointState(robot, joint_array)
        
        sleep(1./240.)

    p.disconnect()