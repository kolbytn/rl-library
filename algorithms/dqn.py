from algorithms.rl_algorithm import RlAlgorithm
from helpers.functions import calculate_returns
from helpers.dataset import RandomReplay
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from itertools import chain
import torch
import numpy as np


class Dqn(RlAlgorithm):
    def __init__(self, env, value_network, embeding_network=None, experience=RandomReplay,
                 device=torch.device('cpu'), gamma=0.99, episode_len=200, lr=1e-3, betas=(0.9, 0.999),
                 weight_decay=0.01, experiment_name='project'):
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

        self._policy = self._policy()

    def train(self, epochs, num_rollouts=100, episode_length=200, batch_size=256):
        pass

    def _policy(self, state):
        action_dist = self._value_net(state)
        return action_dist, torch.argmax(action_dist)
