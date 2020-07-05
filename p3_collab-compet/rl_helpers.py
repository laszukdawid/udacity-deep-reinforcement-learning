import numpy as np
import torch

from typing import Union


class RescaleNormalizer:
    def __init__(self, coef: Union[int, float]=1.0):
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return x * self.coef

class CountNormalizer:
    def __init__(self, coef: Union[int, float]=1.0):
        self.coef = coef
    
    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return (x != 0) * self.coef


class LinearSchedule:
    def __init__(self, start, end=None, steps=1):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val

class OrnsteinUhlenbeckProcess():
    def __init__(self, size, std=None, theta: float=.15, dt: Union[int, float]=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std if std is not None else LinearSchedule(0.2)
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
                        + self.std() * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

class Replay:
    def __init__(self, memory_size: int, batch_size: int, drop_prob: float=0, to_np: bool=True):
        self.memory_size: int = memory_size
        self.batch_size: int = batch_size
        self.data = []
        self.pos = 0
        self.drop_prob = drop_prob
        self.to_np = to_np

    def feed(self, experience):
        if np.random.rand() < self.drop_prob:
            return
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if self.empty():
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        sampled_data = zip(*sampled_data)
        if self.to_np:
            sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return sampled_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def shuffle(self):
        np.random.shuffle(self.data)

    def clear(self):
        self.data = []
        self.pos = 0
