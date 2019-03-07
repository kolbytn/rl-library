from algorithms.rl_algorithm import RlAlgorithm
from utils.dataset import RandomReplay
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from itertools import chain
import torch
import math
import random


class Dqn(RlAlgorithm):
    def __init__(self, env, value_network, embeding_network=None, device=torch.device('cpu'), gamma=0.999,
                 replay_capacity=100000, episode_len=200, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
                 eps_start=.9, eps_end=.05, eps_decay=200, experiment_name='project'):
        super(Dqn, self).__init__(env, experiment_name, gamma, episode_len, device)

        self._eps_start = eps_start
        self._eps_decay = eps_decay
        self._eps_end = eps_end
        self._steps = 0

        self._target_net = deepcopy(value_network).to(device)
        self._value_net = value_network.to(device)
        params = self._value_net.parameters()

        self._embedding_net = None
        if embeding_network:
            self._embedding_net = embeding_network
            params = chain(params, self._embedding_net.parameters())

        self._optim = optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self._crit = nn.functional.smooth_l1_loss

        self._experience = RandomReplay(replay_capacity, ('state', 'state_prime', 'action', 'reward'))

    def train(self, epochs, num_rollouts=50, num_samples=10000, batch_size=128, target_update=10):
        loop = tqdm(total=epochs, position=0, leave=False)
        for epoch in range(epochs):
            avg_reward = self.run_env(num_rollouts)
            dataset = self._experience.sample(num_samples)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            avg_loss = 0
            for state, state_prime, action, reward in data_loader:
                state = self._prepare_tensor(state)
                state_prime = self._prepare_tensor(state_prime)
                reward = self._prepare_tensor(reward)

                self._optim.zero_grad()
                values = self._value_net(state).squeeze(0).gather(1, action.unsqueeze(1)).squeeze(1)
                target = torch.max(self._target_net(state_prime)[0], 1)[0]
                target = reward + self.gamma * target
                loss = torch.mean(self._crit(target.squeeze(0), values))
                loss.backward()
                self._optim.step()

                avg_loss = (avg_loss + loss.item()) / 2
            loop.set_description('reward: %6.2f, value loss: %6.2f' % (avg_reward, avg_loss))

            if epoch % target_update == 0:
                self._target_net = deepcopy(self._value_net)
            print()
            loop.update(1)

    def _policy(self, state):
        eps = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1 * self._steps / self._eps_decay)
        self._steps += 1
        action_val = self._value_net(state)
        action = random.randint(0, self.action_dim - 1)
        if random.random() > eps:
            action = torch.argmax(action_val).item()
        return action_val, action
