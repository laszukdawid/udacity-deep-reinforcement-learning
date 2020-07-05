# Project report

## Introduction

Some tasks require that there are more than one agent acting with the environment. This can be cooperative, like with building a building, or antagonistic, futuristic military skirmishes.
The difficulty in such tasks comes from the agent respodning not only to the environment which has its own laws, but also to other agents who have their own agendas, rarely being a stationary.

In this project we enjoyed a tennis environment where two virtual agents were having fun. The goal was sustain their enjoyment for as long as possible, or until enough fun was obtained.

## Experiment

Two agents have volunteered to participate in our ball bouncing study for indefinite amount of time, divided into episodes for ease of reference.
Their goal was to maximise accumulated reward; points were given for each succesull bounce to opponents side (+0.1 for the bouncer) and reduced when a ball a ground (-0.01 for ground owner).
Such scoring gives an incentive to continue bouncing ball back-and-forth.

The environment is considered solved when over 100 episodes the sum of best rewards per episode are at least 50 (average 0.5).

## Method

We used the Multi Agent Deep Deterministic Policy Gradient (MADDPG) to control our agents. Details of the method are described in a [paper by OpenAI team with collaborators](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). The method uses Deep Deterministic Policy Gradient (DDPG) to control individual agents, however, the critic networks are shared between all agents. Sharing and training is done through sampling on a replay buffer.

In addition to the MADDPG we have applied [Parameter Noise](https://openai.com/blog/better-exploration-with-parameter-noise/) modeled as Ornstein-Unhlenbeck process. The purpose of such augmentation was to improve state-space exploration with disrubting much the policy and value networks.

## Hyperparameters

It's been empirically verified that the task completion is heavily relayent on a proper hyperparmeter tunning. Our final solution have been configured with the following parmeters

| Name | Value | Descrption |
|-------|--------|-------------|
|Batch size | 512 | Number of samples retrieved from the replay buffer at training session.
|Buffer size| 1e6 | Number of last samples stored in the buffer and available for retrival.
|Noise scale|  XX | The amplitude of the noise generated.
|Noise theta|  XX | Value related to how quickly the noise process converges to the mean.
|Warm up period| XX | Number of samples since the initiation when the model wasn't training.
|Gamma | 0.95 | Discount factor used in the reward scaling.
|Tau  | 0.002 | The amount used in the soft update.


## Conclusion

The main challange of the task was to fine tune hyperparameters, especially the amount of noise. With too much noise agents would return extreme actions and end up on either side of their ground. With too little noise agents would quickly stop exploring and, given the rarity of the reward, they would give up on the task completly. We have found that a periodic restart of the noise, not necessarily related to the episode count, have helped the agents as a gentle reminder to not give up.

During the manual hyperparameters tunning a common observation was a gradual increases in averaged reward followed by a sudden drop. This might happen on a couple occasion within a single (about 1000 episodes). 
