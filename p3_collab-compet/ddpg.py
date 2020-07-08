from typing import Iterable
import torch
from torch.optim import Adam

from networks import ActorBody, CriticBody
from noises import GaussianNoise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:

    def __init__(self, state_dim: int, action_dim: int, agents: int=2, hidden_layers: Iterable[int]=(128, 128),
                        actor_lr=1e-3, actor_lr_decay=0, critic_lr=2e-3, critic_lr_decay=0,
                        noise_scale=0.2, noise_sigma=0.1, clip=(-1, 1), device=None):
        super(DDPGAgent, self).__init__()
        self.device = device if device is not None else DEVICE

        # Reason sequence initiation.
        self.actor = ActorBody(state_dim, action_dim, hidden_layers=hidden_layers).to(self.device)
        self.critic = CriticBody(agents*state_dim, agents*action_dim, hidden_layers=hidden_layers).to(self.device)
        self.target_actor = ActorBody(state_dim, action_dim, hidden_layers=hidden_layers).to(self.device)
        self.target_critic = CriticBody(agents*state_dim, agents*action_dim,  hidden_layers=hidden_layers).to(self.device)

        # Noise sequence initiation
        self.noise = GaussianNoise(dim=(action_dim,), mu=1e-8, sigma=noise_sigma, scale=noise_scale, device=device)

        # Target sequence initiation
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        # Optimization sequence initiation.
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_lr_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_lr_decay)
        self.action_min = clip[0]
        self.action_max = clip[1]

        # Breath, my child.
        self.reset_agent()
    
    def reset_agent(self) -> None:
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
