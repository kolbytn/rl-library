from algorithms.rl_algorithm import RlAlgorithm
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from itertools import chain
import torch


class Dqn(RlAlgorithm):
    def __init__(self, env, value_network, experience, embeding_network=None, device=torch.device('cpu'), gamma=0.99,
                 episode_len=200, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, experiment_name='project'):
        super(Dqn, self).__init__(env, experience, experiment_name, gamma, episode_len, device)

        self._target_net = deepcopy(value_network).to(device)
        self._value_net = value_network.to(device)
        params = self._value_net.parameters()

        self._embedding_net = None
        if embeding_network:
            self._embedding_net = embeding_network
            params = chain(params, self._embedding_net.parameters())

        self._optim = optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self._crit = nn.MSELoss()

    def train(self, epochs, num_rollouts=10, num_samples=1000, batch_size=128, target_update=10):
        loop = tqdm(total=epochs, position=0, leave=False)
        for epoch in range(epochs):
            avg_reward = self.run_env(num_rollouts)
            dataset = self._experience.sample(num_samples)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            avg_loss = 0
            for state, action, action_data, reward, in data_loader:
                self._optim.zero_grad()
                target = reward + self.gamma * self._target_net(state)
                loss = self._crit(target[action], action_data[action])
                loss.backward()
                self._optim.step()

                avg_loss = (avg_loss + loss.item()) / 2
            loop.set_description('reward: %6.2f, value loss: %6.2f' % (avg_reward, avg_loss))

            if epoch % target_update == 0:
                self._target_net = deepcopy(self._value_net)
            print()
            loop.update(1)

    def _policy(self, state):
        action_val = self._value_net(state)
        return action_val, torch.argmax(action_val)
