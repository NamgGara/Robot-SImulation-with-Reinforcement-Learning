import pybullet as p
import pybullet_data
from time import sleep
import hyperparameters

# pybullet boilerplate
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*hyperparameters.gravity)
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(hyperparameters.urdf_model, hyperparameters.spawn_point, p.getQuaternionFromEuler(hyperparameters.spawn_pitch))
joint_array = range(p.getNumJoints(robot))
feature_length = len(joint_array)


if __name__ == "__main__":

    for i in range(hyperparameters.simualtion_step):

        p.stepSimulation()
        sleep(hyperparameters.simulation_speed)
        
    p.disconnect()
