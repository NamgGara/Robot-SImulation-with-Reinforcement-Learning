import torch

class Model:
    def __init__(self, env_obs_space=16, env_action_space=4) -> None:
        self.q_table = torch.zeros(size=(env_obs_space, env_action_space))
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.01

    def Q_value(self,old_obs, new_obs, action, reward):
        Q = self.q_table
        Q[old_obs,action] = Q[old_obs,action] - (reward + self.gamma*torch.max(Q[new_obs]) - Q[old_obs,action])
        print(Q[0])

    def action(self,state):
        self.epsilon = max(0.1, self.epsilon - self.epsilon_decay)
        if torch.rand(size=(1,)) < self.epsilon:
            return torch.randint(0, 4, size=(1,)).item()
        return int(torch.max(self.q_table[state]).item())

