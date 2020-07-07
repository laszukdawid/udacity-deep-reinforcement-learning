import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Union

from buffers import ReplayBuffer, DirectedReplayBuffer
from ddpg import DDPGAgent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, env, state_dim: int, action_dim: int, config: Dict, device=None, writer=None):
        self.logger = logging.getLogger("MADDPG")
        self.device = device if device is not None else DEVICE
        self.writer = writer

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agents_number = config['agents_number']

        hidden_layers = config.get('hidden_layers', (400, 300))
        noise_scale = config.get('noise_scale', 0.2)
        noise_sigma = config.get('noise_sigma', 0.1)
        actor_lr = config.get('actor_lr', 1e-3)
        actor_lr_decay = config.get('actor_lr_decay', 0)
        critic_lr = config.get('critic_lr', 1e-3)
        critic_lr_decay = config.get('critic_lr_decay', 0)
        self.actor_tau = config.get('actor_tau', 0.002)
        self.critic_tau = config.get('critic_tau', 0.002)
        create_agent = lambda: DDPGAgent(
                                    state_dim, action_dim, agents=self.agents_number,
                                    hidden_layers=hidden_layers, actor_lr=actor_lr, actor_lr_decay=actor_lr_decay, critic_lr=critic_lr, critic_lr_decay=critic_lr_decay,
                                    noise_scale=noise_scale, noise_sigma=noise_sigma,
                                    device=self.device)
        self.maddpg_agent = [create_agent() for _ in range(self.agents_number)]
        
        self.discount = 0.99 if 'discount' not in config else config['discount']
        self.gradient_clip = 1.0 if 'gradient_clip' not in config else config['gradient_clip']

        self.warm_up = 1e3 if 'warm_up' not in config else config['warm_up']
        self.buffer_size = int(1e6) if 'buffer_size' not in config else config['buffer_size']
        self.batch_size = config.get('batch_size', 128)
        self.p_batch_size = config.get('p_batch_size', int(self.batch_size // 2))
        self.n_batch_size = config.get('n_batch_size', int(self.batch_size // 4))
        self.buffer = ReplayBuffer(self.batch_size, self.buffer_size)
        # self.buffer = DirectedReplayBuffer(self.batch_size, self.buffer_size, p_batch_size=self.p_batch_size, n_batch_size=self.n_batch_size, device=self.device)

        self.update_every_iterations = config.get('update_every_iterations', 2)
        self.number_updates = config.get('number_updates', 2)

        self.reset()

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def reset(self):
        self.iteration = 0
        self.reset_agents()
        self.reset_noise()

    def reset_agents(self):
        for agent in self.maddpg_agent:
            agent.reset_agent()
    
    def reset_noise(self):
        for agent in self.maddpg_agent:
            agent.reset_noise()

    def step(self, state, action, reward, next_state, done) -> None:
        if np.isnan(state).any() or np.isnan(next_state).any():
            print("State contains NaN. Skipping.")
            return

        self.iteration += 1
        self.buffer.add(state, action, reward, next_state, done)

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_every_iterations) == 0:
            self.evok_learning()

    def filter_batch(self, batch, agent_number):
        states, actions, rewards, next_states, dones = batch
        agent_states = states[:, agent_number*self.state_dim:(agent_number+1)*self.state_dim].clone()
        agent_next_states = next_states[:, agent_number*self.state_dim:(agent_number+1)*self.state_dim].clone()
        agent_rewards = rewards.select(1, agent_number).view(-1, 1).clone()
        agent_dones = dones.select(1, agent_number).view(-1, 1).clone()
        return (agent_states, states, actions, agent_rewards, agent_next_states, next_states, agent_dones)

    def evok_learning(self):
        for _ in range(self.number_updates):
            for agent_number in range(self.agents_number):
                batch = self.filter_batch(self.buffer.sample(), agent_number)
                self.learn(batch, agent_number)
                # self.update_targets()

    def act(self, states, noise: Union[None, List]=None):
        """get actions from all agents in the MADDPG object"""

        noise = [0]*self.agents_number if noise is None else noise

        # tensor_states = torch.tensor(states).view(-1, self.agents_number, self.state_dim)
        tensor_states = torch.tensor(states)
        with torch.no_grad():
            actions = []
            for agent_number, agent in enumerate(self.maddpg_agent):
                agent.actor.eval()
                # actions += agent.act(tensor_states.select(1, agent_number), noise[agent_number])
                actions += agent.act(tensor_states, noise[agent_number])
                agent.actor.train()

        return torch.stack(actions)

    def learn(self, samples, agent_number: int) -> None:
        """update the critics and actors of all the agents """

        action_offset = agent_number*self.action_dim
        flatten_actions = lambda a: a.view(-1, self.agents_number*self.action_dim)

        # No need to flip since there are no paralle agents
        agent_states, states, actions, rewards, agent_next_states, next_states, dones = samples

        agent = self.maddpg_agent[agent_number]

        next_actions = actions.clone()
        # next_actions.data[:, action_offset:action_offset+self.action_dim] = agent.target_actor(agent_next_states)
        next_actions[:, action_offset:action_offset+self.action_dim] = agent.target_actor(next_states)

        # critic loss
        Q_target_next = agent.target_critic(next_states, flatten_actions(next_actions))
        Q_target = rewards + (self.discount * Q_target_next * (1 - dones))
        Q_expected = agent.critic(states, actions)
        critic_loss = F.smooth_l1_loss(Q_expected, Q_target)
        # critic_loss = F.mse_loss(Q_expected, Q_target)

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), self.gradient_clip)
        agent.critic_optimizer.step()

        # Compute actor loss
        pred_actions = actions.clone()
        # pred_actions.data[:, action_offset:action_offset+self.action_dim] = agent.actor(agent_states)
        pred_actions[:, action_offset:action_offset+self.action_dim] = agent.actor(states)

        actor_loss = -agent.critic(states, flatten_actions(pred_actions)).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        if self.writer:
            self.writer.add_scalar(f'agent{agent_number}/critic_loss', critic_loss.item(), self.iteration)
            self.writer.add_scalar(f'agent{agent_number}/actor_loss', abs(actor_loss.item()), self.iteration)
        
        self._soft_update(agent.target_actor, agent.actor, self.actor_tau)
        self._soft_update(agent.target_critic, agent.critic, self.critic_tau)

    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            self._soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.actor_tau)
            self._soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.critic_tau)

    def _soft_update(self, target: nn.Module, source: nn.Module, tau) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
