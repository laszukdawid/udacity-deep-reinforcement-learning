import numpy as np
import random
import torch

from collections import deque, namedtuple

device = DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DirectedReplayBuffer:
    def __init__(self, batch_size: int, size: int=10000, agents_number: int=2, p_batch_size=0, n_batch_size=0, device=None):
        self.batch_size = batch_size
        self.agents_number = agents_number
        self.p_batch_size = p_batch_size
        self.n_batch_size = n_batch_size
        self.size: int = size
        self.neutral_memory = deque(maxlen=size)
        self.positive_memory = [deque(maxlen=size//agents_number) for agent in range(agents_number)]
        self.negative_memory = [deque(maxlen=size//agents_number) for agent in range(agents_number)]
        self.experiance = namedtuple("exp", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.device = device if device is not None else DEVICE

    def add(self, state, action, reward, next_state, done):
        exp = self.experiance(state, action, reward, next_state, done)
        self.neutral_memory.append(exp)
        for idx, r in enumerate(reward):
            if r == 0:
                continue
            if r > 0:
                self.positive_memory[idx].append(exp)
            else:
                self.negative_memory[idx].append(exp)

    def sample(self):
        experiences = [exp for exp in random.sample(self.neutral_memory, k=self.batch_size) if exp is not None]
        if self.p_batch_size > 0:
            for p_memory in self.positive_memory:
                experiences+= [exp for exp in random.sample(p_memory, k=min(self.p_batch_size, len(p_memory))) if exp is not None]
        if self.n_batch_size > 0:
            for n_memory in self.negative_memory:
                experiences+= [exp for exp in random.sample(n_memory, k=min(self.n_batch_size, len(n_memory))) if exp is not None]

        states = torch.tensor([e.state for e in experiences]).float().to(self.device).detach()
        actions = torch.tensor([e.action for e in experiences]).float().to(self.device).detach()
        rewards = torch.tensor([e.reward for e in experiences]).float().to(self.device).detach()
        next_states = torch.tensor([e.next_state for e in experiences]).float().to(self.device).detach()
        dones = torch.tensor([e.done for e in experiences]).float().to(self.device).detach()

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.neutral_memory)  # only careing about this since there's about 12:1:1 ratio

class ReplayBuffer:
    def __init__(self, batch_size: int, size: int=10000, p_batch_size=0, n_batch_size=0, device=None):
        self.batch_size = batch_size
        self.p_batch_size = p_batch_size
        self.n_batch_size = n_batch_size
        self.size: int = size
        self.memory = deque(maxlen=size)
        self.positive_memory = deque(maxlen=size//2)
        self.negative_memory = deque(maxlen=size//2)
        self.experiance = namedtuple("exp", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.device = device if device is not None else DEVICE

    def add(self, state, action, reward, next_state, done):
        exp = self.experiance(state, action, reward, next_state, done)
        self.memory.append(exp)
        if any(r>0 for r in reward):
            self.positive_memory.append(exp)
        elif any(r<0 for r in reward):
            self.negative_memory.append(exp)

    def sample(self):
        experiences = [exp for exp in random.sample(self.memory, k=self.batch_size) if exp is not None]
        if self.p_batch_size > 0:
            experiences+= [exp for exp in random.sample(self.positive_memory, k=min(self.p_batch_size, len(self.positive_memory))) if exp is not None]
        if self.n_batch_size > 0:
            experiences+= [exp for exp in random.sample(self.negative_memory, k=min(self.n_batch_size, len(self.negative_memory))) if exp is not None]

        states = torch.tensor([e.state for e in experiences]).float().to(self.device).detach()
        actions = torch.tensor([e.action for e in experiences]).float().to(self.device).detach()
        rewards = torch.tensor([e.reward for e in experiences]).float().to(self.device).detach()
        next_states = torch.tensor([e.next_state for e in experiences]).float().to(self.device).detach()
        dones = torch.tensor([e.done for e in experiences]).float().to(self.device).detach()

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.memory)

class Memory:
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

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
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
