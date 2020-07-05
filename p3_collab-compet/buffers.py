import random
import torch

from collections import deque, namedtuple

device = DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    def __len__(self) -> int:
        return len(self.memory)
