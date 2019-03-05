from environments.gym import Cartpole
from utils.networks import PolicyNetwork, ValueNetwork, MLP
from algorithms.ppo import Ppo
from algorithms.dqn import Dqn
from utils.dataset import RandomReplay
import torch
import torch.nn as nn
from itertools import chain


def cartpole_dqn():
    env = Cartpole()
    value = MLP(env.state_dim, env.action_dim, hidden_size=10)
    experience = RandomReplay(100000, ('state', 'action_data', 'action', 'reward'))

    dqn = Dqn(env, value, experience, experiment_name='cartpole_dqn')
    result_data = dqn.train(1000)

    return result_data


def cartpole_ppo():
    env = Cartpole()
    policy = PolicyNetwork(4, 2)
    value = ValueNetwork(4)

    optim = torch.optim.Adam(chain(policy.parameters(), value.parameters()), lr=1e-3, betas=(0.9, 0.999))
    value_objective = nn.MSELoss()

    # Hyperparameters
    epochs = 1
    env_samples = 100
    episode_length = 200
    gamma = 0.9
    policy_epochs = 5
    batch_size = 256
    epsilon = 0.2
    c_value = 1
    c_policy = 1

    ppo = Ppo()
    returns = ppo.train(env, policy, value, optim, value_objective, epochs, env_samples,
                     episode_length, gamma, policy_epochs, batch_size, epsilon,
                     c_value, c_policy)

    return returns
