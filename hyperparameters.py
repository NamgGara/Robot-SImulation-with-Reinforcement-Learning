gravity = [0,0,-190.9]
spawn_location = [0,0,0]
spawn_pitch = [0,1,0]

urdf_model = "humanoid.urdf"
plane = "plane.urdf"

simulation_step_number = 1000
epoch = 50
batch_size = 3
feature_length = 15 + 15
action_space = 15
simulation_speed = 1./400.

VPG_mu_learning_rate = 0.00001
VPG_sigma_learning_rate = 0.00001
Critic_lr = 0.00001
speed_factor = 10
action_factor = 10
