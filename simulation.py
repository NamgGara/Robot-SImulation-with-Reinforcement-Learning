import pybullet as p
import pybullet_data
from time import sleep
import hyperparameters

# pybullet boilerplate
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*hyperparameters.gravity)

plane = p.loadURDF(hyperparameters.plane)
robot = p.loadURDF(hyperparameters.urdf_model, hyperparameters.spawn_point,
                   p.getQuaternionFromEuler(hyperparameters.spawn_pitch))
joint_array, feature_length = range(p.getNumJoints(robot)), len(range(p.getNumJoints(robot)))

if __name__ == "__main__":

    for i in range(hyperparameters.simualtion_step):

        p.stepSimulation()
        sleep(hyperparameters.simulation_speed)
        
    p.disconnect()
