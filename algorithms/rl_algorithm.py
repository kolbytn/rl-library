import numpy as np
import torch
import random


class RlAlgorithm:
    def __init__(self, env, experiment_name, gamma, episode_len, device):
        self.experiment_name = experiment_name
        self.action_dim = env.action_dim
        self.gamma = gamma
        self.episode_len = episode_len
        self.device = device

        self._env = env
        self._experience = None  # Must be set by child

    def run_env(self, num_rollouts):
        rollouts = []
        avg_reward = 0
        for _ in range(num_rollouts):
            rollout = []
            state = self._env.reset()
            episode_reward = 0

            for _ in range(self.episode_len):
                s = self._prepare_tensor(state)

                a_data, a = self._policy(s)
                s_prime, r, t = self._env.step(a)

                rollout.append({'state': state,
                                'action': a,
                                'action_data': a_data.cpu().detach().numpy(),
                                'reward': r,
                                'terminal': t})
                if t:
                    break
                state = s_prime
                episode_reward += r

            rollouts.append(rollout)
            avg_reward = (avg_reward + episode_reward) / 2
        self._calc_returns(rollouts)
        self._experience.extend(rollouts)
        return avg_reward

    def _calc_returns(self, rollouts):
        for rollout in rollouts:
            discounted = 0
            for i in reversed(range(len(rollout))):
                discounted = self.gamma * discounted + rollout[i]["reward"]
                rollout[i]["return"] = discounted

    def _prepare_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().unsqueeze(0).to(self.device)
        return array.float().unsqueeze(0).to(self.device)

    def _policy(self, state):
        return np.full(self.action_dim, 1 / self.action_dim), random.randint(0, self.action_dim)
