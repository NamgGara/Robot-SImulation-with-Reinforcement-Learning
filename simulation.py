import pybullet as p
import pybullet_data
from time import sleep

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)

plane = p.loadURDF("plane.urdf")

spawn_point = [0,0,3]
spawn_pitch = p.getQuaternionFromEuler([0,0,0])
robot = p.loadURDF("humanoid.urdf", spawn_point, spawn_pitch)

if __name__== "__main__":
    for i in range(10000):
        p.stepSimulation()
        sleep(1./240.)

    p.disconnect()