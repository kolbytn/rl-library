from algorithms.rl_algorithm import RlAlgorithm
from helpers.functions import calculate_returns
from helpers.dataset import PpoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np


class Ppo(RlAlgorithm):
    def __init__(self):
        super(Ppo, self).__init__()

    def train(self, env, policy, value, optim, value_objective, epochs, env_samples,
              episode_length, gamma, policy_epochs, batch_size, epsilon,
              c_value, c_policy):
        returns = []
        loop = tqdm(total=epochs, position=0, leave=False)
        for epoch in range(epochs):
            # generate rollouts
            rollouts = []
            frames = []
            standing_len = 0
            for episode in range(env_samples):
                state = env.reset()
                done = False
                rollout = []
                total_reward = 0

                for step in range(1, episode_length + 1):
                    step = {}
                    state_torch = torch.FloatTensor(state).unsqueeze(0)
                    action_dist = policy(state_torch)
                    action_dist = action_dist[0].detach().numpy()
                    action = np.argmax(np.random.multinomial(1, action_dist))
                    state_p, r, done = env.step(action)

                    step["state"] = state
                    step["reward"] = r
                    step["action_dist"] = action_dist
                    step["action"] = action

                    total_reward += 1
                    rollout.append(step)
                    if done:
                        break

                    state = state_p

                    rollouts.append(rollout)
                    if total_reward == 200:
                        print("episodes: ", epoch * env_samples + episode)

            calculate_returns(rollouts, gamma)
            returns.append(sum([sum([s["return"] for s in r]) / len(r)
                           for r in rollouts]) / len(rollouts))

            # Approximate the value and policy
            dataset = PpoDataset(rollouts)
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, pin_memory=True)
            for _ in range(policy_epochs):
                for step in loader:
                    optim.zero_grad()

                    state = step["state"].float()
                    ret = step["return"].float().unsqueeze(1)
                    est_return = value(state)
                    advantage = ret - est_return
                    value_loss = c_value * value_objective(est_return, ret)

                    advantage.detach_()
                    action_dist = policy(state)
                    cur = action_dist[range(action_dist.shape[0]),
                                            step["action"]].unsqueeze(1)
                    old = step["action_dist"][range(
                        step["action_dist"].shape[0]), step["action"]].unsqueeze(1)
                    ratio = cur / old
                    left = ratio * advantage
                    right = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
                    policy_loss = c_policy * -torch.mean(torch.min(left, right))

                    loss = value_loss + policy_loss
                    loss.backward()
                    optim.step()

                loop.update(1)
                loop.set_description('Epoch:{}, Return:{:.4f}, Value Loss:{:.4f}, Policy Loss:{:.4f}'
                    .format(epoch, returns[-1], value_loss, policy_loss))
        return returns
