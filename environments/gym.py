import rl_environment

class Gym(RlEnvironment):
    __init__(self):
        super(Gym, self).__init__()


class Cartpole(Gym):
    __init__(self):
        super(Cartpole, self).__init__()
        self._env = gym.make('CartPole-v0')

    def step(self, action):
        s, r, t, _ = self._env.step(action.item())
        return s, r, t

    def reset(self):
        return self._env.reset()
