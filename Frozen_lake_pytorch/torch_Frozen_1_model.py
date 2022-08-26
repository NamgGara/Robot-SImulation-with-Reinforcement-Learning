import torch

class Model:
    def __init__(self, env_obs_space, env_action_space) -> None:
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.lr = 0.1
        self.q_table = torch.zeros(size=(env_obs_space, env_action_space))
        
        #testing
        self.epsilon = 0
        self.q_table = torch.tensor([[0.0613, 0.0605, 0.0605, 0.0587],
        [0.0412, 0.0452, 0.0447, 0.0576],
        [0.0745, 0.0679, 0.0679, 0.0581],
        [0.0388, 0.0394, 0.0324, 0.0539],
        [0.0822, 0.0692, 0.0608, 0.0492],
        [0.0000, 0.0000, 0.0000, 0.0000],
        [0.1090, 0.0924, 0.0999, 0.0212],
        [0.0000, 0.0000, 0.0000, 0.0000],
        [0.0716, 0.1020, 0.0983, 0.1321],
        [0.1368, 0.2359, 0.2254, 0.1298],
        [0.2827, 0.2606, 0.2153, 0.0967],
        [0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000],
        [0.1711, 0.2982, 0.3691, 0.2378],
        [0.3918, 0.6213, 0.5848, 0.5838],
        [0.0000, 0.0000, 0.0000, 0.0000]])

    def Q_value(self,old_obs, new_obs, action, reward):
        self.q_table[old_obs,action] = self.q_table[old_obs,action] + self.lr*(reward + self.gamma*torch.max(self.q_table[new_obs]) - self.q_table[old_obs,action])

    def action(self,state):
        # self.epsilon = max(0.1, self.epsilon*self.epsilon_decay)
        # if torch.rand(size=(1,)) < self.epsilon:
        #     return torch.randint(0, 4, size=(1,)).item()
        return int(torch.argmax(self.q_table[state]).item())

    @staticmethod
    def reward_plot(reward):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18,9))
        plt.plot(reward)
        plt.show()