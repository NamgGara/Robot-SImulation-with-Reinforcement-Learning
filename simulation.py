import pybullet as p
import pybullet_data
from time import sleep

#hyper parameters
gravity = [0,0,-190.9]
spawn_point = [0,0,0]
spawn_pitch = p.getQuaternionFromEuler([0,1,0])
urdf_model = "..\\humanoid.urdf"
simualtion_step = 10000
simulation_speed = 1./300.

# pybullet boilerplate
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*gravity)
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(urdf_model, spawn_point, spawn_pitch)
joint_array = range(p.getNumJoints(robot))
feature_length = len(joint_array)


if __name__ == "__main__":

    for i in range(simualtion_step):

        p.stepSimulation()
        sleep(simulation_speed)
        
    p.disconnect()
