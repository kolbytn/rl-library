from torch.utils.data import Dataset
import numpy as np
import random
from collections import deque
from copy import deepcopy


class ReplayBuffer:
    def __init__(self):
        pass

    def extend(self, rollouts):
        pass

    def sample(self, n):
        pass


class RandomReplay(ReplayBuffer):
    def __init__(self, capacity, keys):
        super(RandomReplay, self).__init__()
        self.experience = deque(maxlen=capacity)
        self.keys = keys

    def extend(self, rollouts):
        for rollout in rollouts:
            self.experience.extend(rollout)

    def sample(self, n):
        sample = []
        exp = deepcopy(self.experience)
        random.shuffle(exp)
        for i in range(n):
            if len(exp) < 1:
                break
            data = exp.pop()
            sample.append(data)
        return ExperienceDataset(sample, self.keys)


class PrioritizedReplay:
    def __init__(self, capacity, keys, cramer_classes, epsilon=.01, alpha=.6):
        self._tree = SumTree(capacity)
        self.keys = keys
        self.cramer_classes = cramer_classes
        self.e = epsilon
        self.a = alpha
        self.max_error = 5

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def extend(self, rollout):
        for step in rollout:
            self._tree.add(self.max_error, step)

    def sample(self, n):
        sample = []
        n = min(self._tree.length, n)
        segment = self._tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self._tree.get(s)
            data["idx"] = idx
            sample.append(data)
        return ExperienceDataset(sample, self.keys, self.cramer_classes)

    def update(self, idx, error):
        self.max_error = max(self.max_error, error)
        p = self._get_priority(error)
        self._tree.update(idx, p)


class PpoDataset(Dataset):
    def __init__(self, data):
        super(PpoDataset, self).__init__()
        self.data = []
        for d in data:
            self.data.extend(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ExperienceDataset(Dataset):
    def __init__(self, experience, keys):
        super(ExperienceDataset, self).__init__()
        self._keys = keys
        self._exp = []
        for x in experience:
            self._exp.append(x)

    def __getitem__(self, index):
        chosen_exp = self._exp[index]
        return tuple(chosen_exp[k] for k in self._keys)

    def __len__(self):
        return len(self._exp)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.length = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
