# individual network settings for each actor + critic pair
# see networkforall for details

from networks import ActorBody, CriticBody
from torch.optim import Adam
import torch


# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_layers=(128, 128)):
        super(DDPGAgent, self).__init__()

        self.actor = ActorBody(state_dim, action_dim, hidden_layers=hidden_layers).to(device)
        self.critic = CriticBody(state_dim, action_dim, hidden_layers=hidden_layers).to(device)
        self.target_actor = ActorBody(state_dim, action_dim, hidden_layers=hidden_layers).to(device)
        self.target_critic = CriticBody(state_dim, action_dim,  hidden_layers=hidden_layers).to(device)

        self.noise = OUNoise(action_dim, scale=1.0, device=device)

        # initialize targets same as original networks
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=5e-3)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-3)

    def _hard_update(self, target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs) + noise*self.noise.noise()
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action

