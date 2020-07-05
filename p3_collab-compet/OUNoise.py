import numpy as np
import torch

from typing import Union


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, scale=0.2, mu=0.0, theta=0.15, sigma=0.2, device=None):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset_states()

        self.device = device if device is not None else torch.device('cpu')

    def reset_states(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float().to(self.device)

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
    def __init__(self, size, std=None, theta: float=.15, dt: Union[int, float]=1e-2, x0=None, device=None):
        self.theta = theta
        self.mu = 0
        self.std = std if std is not None else LinearSchedule(0.2)
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.device = device if device is not None else torch.device('cuda')
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
                        + self.std() * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return torch.tensor(x).to(self.device).float()

    def reset_states(self) -> None:
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)