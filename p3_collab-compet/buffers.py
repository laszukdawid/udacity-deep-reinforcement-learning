import numpy as np
import random
import torch

from collections import deque, namedtuple
from typing import List, Union

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, batch_size: int, size=10000, device=None):
        self.batch_size = batch_size
        self.size: int = size
        self.memory = deque(maxlen=size)
        self.experiance = namedtuple("exp", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.device = device if device is not None else DEVICE

    def add(self, state, action, reward, next_state, done):
        exp = self.experiance(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        experiences = [exp for exp in random.sample(self.memory, k=self.batch_size) if exp is not None]

        states = torch.tensor([e.state for e in experiences]).float().to(self.device).detach()
        actions = torch.tensor([e.action for e in experiences]).float().to(self.device).detach()
        rewards = torch.tensor([e.reward for e in experiences]).float().to(self.device).detach()
        next_states = torch.tensor([e.next_state for e in experiences]).float().to(self.device).detach()
        dones = torch.tensor([e.done for e in experiences]).float().to(self.device).detach()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Storage:

    def __init__(self, size: int, keys: Union[None, List[str]]=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data) -> None:
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self) -> None:
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys: List[str]):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
