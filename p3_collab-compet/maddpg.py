# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

import logging
import numpy as np

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
from rl_helpers import RescaleNormalizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'



class MADDPG(object):
    def __init__(self, state_dim, action_dim, agents_number: int, discount=0.95, tau=0.005):
        # super().all()

        self.maddpg_agent = [DDPGAgent(state_dim, action_dim, hidden_layers=(100, 100)) for _ in range(agents_number)]
        
        self.agents_number = agents_number
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau

        self.reward_normalizer = RescaleNormalizer(10)
        self.logger = logging.getLogger("MADDPG")


    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, states_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(torch.tensor(states_all_agents[num]), noise) for num, agent in enumerate(self.maddpg_agent)]
        return actions

    def target_act(self, states_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.target_act(states_all_agents.select(1, num), noise) for num, agent in enumerate(self.maddpg_agent)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        # No need to flip since there are no paralle agents
        state, action, reward, next_state, done = samples
        reward = self.reward_normalizer(reward)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_state)
        target_actions = torch.cat(target_actions, dim=1).view(-1, self.agents_number, self.action_dim)

        agent_state = state.select(1, agent_number).detach()
        agent_action = action.select(1, agent_number).detach()
        agent_target_actions = target_actions.select(1, agent_number).detach()
        
        # Make sure this works
        with torch.no_grad():
            q_next = agent.target_critic(agent_state, agent_target_actions)
        
        y = reward.select(1, agent_number) + self.discount * q_next * (1 - done.select(1, agent_number))

        q = agent.critic(agent_state, agent_action)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ agent.actor(state.select(1, num)) if num == agent_number \
                   else agent.actor(state.select(1, num)).detach()
                   for num, agent in enumerate(self.maddpg_agent)]
                
        q_input = torch.cat(q_input, dim=1).view(-1, self.agents_number, self.action_dim)
        
        # get the policy gradient
        actor_loss = -agent.critic(state, q_input).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        # al = actor_loss.cpu().detach().item()
        # cl = critic_loss.cpu().detach().item()
        # self.logger.info(f'{self.iter}  | agent{agent_number}/losses  | critic loss: {cl}  | actor_loss: {al}')

    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            self._soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            self._soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)

    def _soft_update(self, target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
