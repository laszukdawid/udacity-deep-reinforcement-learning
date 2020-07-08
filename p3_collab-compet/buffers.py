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
    def __init__(self, batch_size: int, buffer_size: int=10000, p_batch_size: int=0, n_batch_size: int=0, device=None):
        """
            Enhanced Replay Experience Buffer.
            In addition to normal buffer, this class contains two smaller that are used for positive/negative rewards only.
            Positive and negative experiences are retrived when p_batch_size (positive) or n_batch_size (negative)
            are non zero (by default both are set to zero).
        """
        self.batch_size = batch_size
        self.p_batch_size = p_batch_size
        self.n_batch_size = n_batch_size
        self.buffer_size: int = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.positive_memory = deque(maxlen=buffer_size//2)
        self.negative_memory = deque(maxlen=buffer_size//2)
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
