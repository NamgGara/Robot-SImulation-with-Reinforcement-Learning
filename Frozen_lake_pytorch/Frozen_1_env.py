import gym
from torch_Frozen_1_model import Model

env = gym.make("FrozenLake-v1")
model = Model(env.observation_space.n, env.action_space.n)
total = []
total_count = 0
episodes = 100
time_step = 100
for i in range(episodes):
    old_obs = env.reset()
    done = False

    for i in range(time_step):
        # env.render()
        action = model.action(old_obs)
        new_obs, reward, done, _ = env.step(action)
        model.Q_value(old_obs, new_obs, action, reward)
        old_obs = new_obs
        if done:
            total.append(reward)
            total_count+=reward
            break

model.reward_plot(total)
print(total_count)

