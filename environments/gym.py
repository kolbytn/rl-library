from environments.rl_environment import RlEnvironment
import gym


class Gym(RlEnvironment):
    def __init__(self):
        super(Gym, self).__init__()


class Cartpole(Gym):
    def __init__(self):
        super(Cartpole, self).__init__()
        self._env = gym.make('CartPole-v0')
        self.action_dim = 2
        self.state_dim = 4

    def step(self, action):
        s, r, t, _ = self._env.step(action)
        return s, r, t

    def reset(self):
        return self._env.reset()
