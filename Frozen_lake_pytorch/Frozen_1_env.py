import gym
import matplotlib.pyplot as plt
from torch_Frozen_1_model import Model

env = gym.make("FrozenLake-v1")
old_obs = env.reset()
model = Model()

for i in range(100):
    env.render()
    action = model.action(old_obs)
    new_obs, reward, done, _ = env.step(action)
    model.Q_value(old_obs, new_obs, action, reward)
    if done:
        env.reset()

print(model.q_table)

