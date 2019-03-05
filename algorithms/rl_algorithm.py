import numpy as np
import torch


class RlAlgorithm:
    def __init__(self, env, experience, experiment_name, gamma, episode_len, device):
        self.experiment_name = experiment_name
        self.gamma = gamma
        self.episode_len = episode_len
        self.device = device
        self._env = env
        self._experience = experience
        self._policy = None  # Implement random policy

    def run_env(self, num_rollouts):
        rollouts = []
        for _ in range(num_rollouts):
            rollout = []
            state = self._env.reset()

            for _ in range(self.episode_len):
                s = self._from_numpy(state)

                a_dist, a = self._policy(state)
                s_prime, r, t = self._env.step(a)

                rollout.append({'state': s, 'action': a, 'action_dist': a_dist, 'reward': r, 'terminal': t})
                if t:
                    break
                state = s_prime

            rollouts.append(rollout)
        self._calc_returns(rollouts)
        self._experience.add(rollouts)

    def _calc_returns(self, rollouts):
        for rollout in rollouts:
            discounted = 0
            for i in reversed(range(len(rollout))):
                discounted = self.gamma * discounted + rollout[i]["reward"]
                rollout[i]["return"] = discounted

    def _from_numpy(self, array):
        return torch.from_numpy(array).float().unsqueeze(0).to(self.device)
