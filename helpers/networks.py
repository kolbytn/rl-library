import torch.nn as nn


class PolicyNetwork(nn.Module):
  def __init__(self, state_size=4, action_size=2):
    super(PolicyNetwork, self).__init__()
    self.state_size = state_size
    self.action_size = action_size

    self.net = nn.Sequential(nn.Linear(self.state_size, 10),
                             nn.ReLU(),
                             nn.Linear(10, 10),
                             nn.ReLU(),
                             nn.Linear(10, 10),
                             nn.ReLU(),
                             nn.Linear(10, self.action_size),
                             nn.Softmax(dim=1))

  def forward(self, x):
    return self.net(x)


class ValueNetwork(nn.Module):
  def __init__(self, state_size=4):
    super(ValueNetwork, self).__init__()
    self.state_size = state_size

    self.net = nn.Sequential(nn.Linear(self.state_size, 10),
                             nn.ReLU(),
                             nn.Linear(10, 10),
                             nn.ReLU(),
                             nn.Linear(10, 10),
                             nn.ReLU(),
                             nn.Linear(10, 1))

  def forward(self, x):
    return self.net(x)
