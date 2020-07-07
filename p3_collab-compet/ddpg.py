import numpy as np
import torch

from networks import ActorBody, CriticBody
from torch.optim import Adam
from OUNoise import OUNoise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, state_dim, action_dim, agents=2, hidden_layers=(128, 128), actor_lr=1e-3, actor_lr_decay=0, critic_lr=2e-3, critic_lr_decay=0, noise_scale=0.2, noise_sigma=0.1, clip=(-1, 1), device=None):
        super(DDPGAgent, self).__init__()
        self.device = device if device is not None else DEVICE

        self.actor = ActorBody(agents*state_dim, action_dim, hidden_layers=hidden_layers).to(self.device)
        self.critic = CriticBody(agents*state_dim, agents*action_dim, hidden_layers=hidden_layers).to(self.device)
        self.target_actor = ActorBody(agents*state_dim, action_dim, hidden_layers=hidden_layers).to(self.device)
        self.target_critic = CriticBody(agents*state_dim, agents*action_dim,  hidden_layers=hidden_layers).to(self.device)
        self.reset_agent()

        # noise_theta = 0.15
        # self.noise = OUNoise((action_dim), scale=noise_scale, theta=noise_theta, sigma=noise_sigma, device=self.device)
        self.noise = GaussianNoise(dim=(action_dim,), mu=1e-8, sigma=noise_sigma, scale=noise_scale, device=device)

        # initialize targets same as original networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_lr_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_lr_decay)
        self.action_min = clip[0]
        self.action_max = clip[1]
    
    def reset_noise(self):
        self.noise.reset_states()

    def reset_agent(self):
        self.actor.reset_parameters()
        self.critic.reset_parameters()
        self.target_actor.reset_parameters()
        self.target_critic.reset_parameters()

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def act(self, obs, noise: float=0.0):
        obs = obs.to(self.device)
        action = self.actor(obs) + noise*self.noise.sample()
        return torch.clamp(action, self.action_min, self.action_max)

    def target_act(self, obs, noise: float=0.0):
        obs = obs.to(self.device)
        action = self.target_actor(obs) + noise*self.noise.sample()
        return torch.clamp(action, self.action_min, self.action_max)

class GaussianNoise:
    def __init__(self, dim, mu=0, sigma=1, scale=1, device=None):
        self.dim = dim
        self.mu = mu
        self.sigma = sigma
        self.scale = scale
        self.device = device if device is not None else DEVICE

    def reset_states(self):
        pass

    def sample(self):
        return torch.tensor(self.scale*np.random.normal(self.mu, self.sigma, self.dim)).to(self.device)