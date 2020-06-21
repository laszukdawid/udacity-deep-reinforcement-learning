import numpy as np
import torch

from typing import Any, Dict, Union

from utils import tensor, to_np
from networks import ActorBody, CriticBody, DeterministicActorCriticNet
from rl_helpers import CountNormalizer, OrnsteinUhlenbeckProcess, RescaleNormalizer, Replay


class Config(object):
    discount: float

    replay_fn: Any
    random_process_fn: Any
    warm_up: int
    target_network_mix: float
    state_normalizer: Union[CountNormalizer, RescaleNormalizer]
    reward_normalizer: Union[CountNormalizer, RescaleNormalizer]
    batch_size: int


class DDPGAgent(object):
    def __init__(self, d_config: Dict):

        self.task = d_config['task']
        self.state_dim = d_config['state_dim']
        self.action_dim = d_config['action_dim']

        self.actor_opt_fn = lambda params: torch.optim.Adam(params, 1e-4)
        self.critic_opt_fn = lambda params: torch.optim.Adam(params, 1e-4)

        actor_body = ActorBody(self.state_dim, hidden_units=(400, 300), gate=torch.relu, out_gate=torch.tanh)
        critic_body = CriticBody(self.state_dim, self.action_dim, hidden_units=(300, 300), gate=torch.tanh)

        # Network and TargetNetwork have the same architecture
        self.network = DeterministicActorCriticNet(
            self.state_dim, self.action_dim,
            actor_opt_fn=self.actor_opt_fn, critic_opt_fn=self.critic_opt_fn,
            actor_body=actor_body, critic_body=critic_body,
        )
        self.target_network = DeterministicActorCriticNet(
            self.state_dim, self.action_dim,
            actor_opt_fn=self.actor_opt_fn, critic_opt_fn=self.critic_opt_fn,
            actor_body=actor_body, critic_body=critic_body,
        )
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.total_steps = 0
        self.state = None

        self.config = Config()
        self.config.discount = 0.99
        self.config.batch_size = 64
        self.config.warm_up = int(1e4)
        self.config.target_network_mix = 1e-3
        self.config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=self.config.batch_size)
        self.config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=(self.action_dim,), dt=1)
        self.config.state_normalizer = RescaleNormalizer()
        self.config.reward_normalizer = CountNormalizer(0.1)

        self.replay = self.config.replay_fn()
        self.random_process = self.config.random_process_fn()


    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            # target_param.detach_()
            target_param.data.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)


    def eval_step(self, state):
        state = self.config.state_normalizer(state)
        action = self.network(state)
        return to_np(action)


    def step(self):
        all_reward = []
        not_finished = True
        self.random_process.reset_states()
        self.state = self.task.reset()
        self.state = self.config.state_normalizer(self.state)
        while(not_finished):
            reward, not_finished = self._step()
            all_reward += [reward]
        return all_reward

    def _step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = to_np(self.task.action_space.sample())
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, int(self.task.action_space.low), int(self.task.action_space.high))
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        reward = norm_reward = self.config.reward_normalizer(reward)

        experiences = list(zip(self.state, action, norm_reward, next_state, done))
        self.replay.feed_batch(experiences)
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if (self.replay.size() >= config.warm_up):
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = self.config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)

        return reward, not all(done)
