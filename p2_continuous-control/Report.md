# Report

## Introduction

The goal of the experiment was to aid an arm without a brain in its eternal task to get closer to a hovering green ball. The difficulty arises from arm's inability to cooperate. Though its nature is complicated, to be precise its internal state is defined by 33 dimensional closed ball, its stubbornness to change allows only for 4 asks at the time (action space is 4D ball [-1, 1]).
As with all problems in this realm, the goal is not to solve the problem but merely extend the appearance of progress by achieving 30 points worth of closeness metric. A step during which the arm has been within the green ball is worth 0.1 closeness point.

## Methodology

On top of lack of cooperation from the inanimate objects, the complexity in the task comes from continuous domains in all dimensions. We not only want demand movement of a particular joint but we're also forced to control the force of these jerks. Following the well established idiom "continuous domain problems require continuous domain solutions" we have decided that a Deep Deterministic Policy Gradients (DDPG) method might just do the trick. 

Our DDPG agent possesses two networks for its learning disposal. They have been called *target_networks* and *networks*. One is used to estimate what is the best policy value for a given state and the other... tries the same but it's a bit slower. We're proud of them both. To addition to this off-policy approach, we have decided that our agent might learn faster if both of its networks had some critic component in addition to the proper actor. Both the Actor and the Critic networks have two fully connected layers, however, their configuration somehow changes. For one, the Actor does the difficult part, thus it's shape flow is (33 -> 400 -> 300 -> 4) with tanh as the output and relu gates internally. Criticizing others without providing much output themselves requires much less mental capacity thus the Critic is built with (33 -> 304 -> 300 -> 1) with tanh gates everywhere. The 304 contains actions made by the Actor; in the end, it needs to fill superior on its hindsight perspective. Using the trick known from the human world as "induced insecurity" we also adjusted the Actor's first hidden layer with a Dropout (prob 0.2) so that it can also feel the overwhelming peer pressure.

One thing that requires reporting is the adjustment of the reward score. Although the provided environment explanation stated that the reward is either 0 or 0.1 (success), the actual obtained values are in range from 0 to (probably) 0.1. We deduced that the reward is a proxy measure for distance to the centre provided that the arm has breached the green sphere. Given our strong sense of being right we went ahead and rescaled the reward, i.e. whenever a positive reward was obtained we counted it as 0.1.

## Experiment

We released the agent and its dual duo to figure out how to master the arm manipulation. They had a Replay buffer with 1e6 memory space, slow learning rates of 1e-4 everywhere, proper membership discount rate 0.99 and learning batch size of 64. We also allowed the *network* to copy some answers from the *target_network* with only a bit (mixing is 1e-3).

To not put too much pressure on the agent from the very beginning we introduced as *warm_up* variable (set to 10000). This made sure that the agent would simply observe the environment for 1000 steps (in practise that's about 7 episodes) when it would remember but not act.

Episodes and their length are defined by the environment. In each episode the ball changes randomly its movement. It can move clockwise or counter-clockwise with some variable speeds (even zero at which points it rests in a random position).

It took the agent about 20 episodes to star receiving noticeable scoring. It then took 30 more episodes to obtain the score of 20 (that's 200 steps when the arm is within the green sphere). The
goal of average above 30 score in the last 100 episodes was achieved after the 165th episode. This result was obtained in about 30 min on our so-so Dell laptop with CUDA support and Ubuntu 20.04. Visualisation of the reward history is provided in the attached Jupyter notebook.


## Conclusion

We conclude that the DDPG agent with its dual-duo has done a good job. The team wishes to come out clean and say how doubtful it was and the agent surpassed our expectations. In fact, it is somehow impressive as all of our team members (1 person) think that it might take them more time to learn this trick in the human realm.

We have also concluded that reward values do matter. It's obvious and it's been mentioned many times almost everywhere, but somehow our brain was closed for the notion of scaling reward value for personal gains. Our third-eye is now well aware that the reward scaler is also a hyper-parameter with the highest importance. Praise be tuning. Which also forces us to mention how disappointed we are with the project's description. Not mad; just disappointed.

For our next trick we are planning on two improvements. Firstly we would like to improve the learning speed. The main metric is the wall time to a functioning agent. We would like to try the Proximal Policy Optimization (PPO) agent as it's claimed to be better, faster, and easier to implement. It'd be interesting to see whether we could achieve a better learning slope. In addition, we would like to make it a parallel task with multiple agents learning at the same time. There's plenty of room in our laptops for more than one agent.
The second improvement would be in the distance from the ball. Current goal was to cross green sphere. We would like to play around with the reward metric and make the arm to be in the exact centre all the time, rather than simply somewhere within the ball. We would try to achieve that by updating the reward value for learning process to be a combination of 0.1 for breaching the sphere and then a scaled distance, e.g. 10 x distance from the centre.

Our final conclusion is that this is fun and exciting. Even the dumb agent seem to have more personality than many characters in TV shows so it's interesting to observe the learning process.
