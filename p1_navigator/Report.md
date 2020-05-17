# Report

## Introduction

The goal of the experiment was to create and train an agent so that it is able to manoeuvre a bounded 2D space whilst maximising the score. Rewards were given when the agent went through an object; a yellow banana increases score by 1 and a blue banana decreases it by 1. A complete state of the agent is a 37 dimensional vector which consists of agent's velocity and ray-based object perception at a various angles.

## Methodology

Given the intention of the task, the Q-learning was selected as a learning strategy with a neural network (NN) used to project state onto the action-value (Q) space. The agents NN consists of 3 fully connected layers with the first and second hidden layers having 128 and 64 nodes, respectively. The rectified linear unit (ReLU) activation function was set for all layers.

The solution was inspired by the DeepMind's proposal on the Deep Q-Learning Network though adjusted to 1D problem space and significantly decreased in size. The inspiration, however, is visible in two major improvements, i.e. reply buffer and Q-fixed dual network. The reply buffer enables reusage of already visited states; instead of discarding (state, action, reward, next state) tuple (SARS tuple) they are stored in a buffer and used in the future learning. Such change allows for mitigating state-action-state correlation issues and helps with premature convergence. In case of the Q-fixed network, the advantage is in mitigating temporal correlation by adding a delay to the learning process.

## Experiment

The learning process was divided into episodes. Each episode started with resetting the environment, i.e. restarting position of the agent and all bananas, and ended when then environment set the "done" flag to True. Each step (SARS tuple) was recorded in the reply buffer of size 100000 recent steps and every 8 steps a batch of 64 SARS tuples would be returned for the learning purposes. Additional parameters used in the process were discount (0.99), learning rate (0.0005) and tau networking learning (0.001). All these parameters were selected as "good enough" based on a brief manual parameter space exploration. The whole learning processes was set to 1000 episodes. The score for each episode is set as the total reward.

## Conclusion

The attached Jupyter notebook contains the result of the experiment. In it one can find a graph that shows a score value as a function of episode number. The graph has an increasing trend up until episode 600 and then it steadily hovers around score 15. For initial episodes (#episode < 200) the score doesn't cross value 10 and for the last 100 episodes the score rarely is mostly above 10. This indicates that the agent has learned. The variance in score can relate to many factors and most likely episode randomization is the main factor.

Although the result is satisfactory it is only a single evaluation. There were no attempts on optimizing the agent to learn faster or to be more versatile. It would be interesting to conduct further experiments with different NN configurations or different hyperparameters to verify how they affect learning trend. The solution is also problem-independent thus it would be interesting to verify how the same agent would perform in another environment with different state and action dimensionality.

More funding is required.
