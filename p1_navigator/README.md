# Project 1: Navigation

---

This notebook provides a solution to completing the Unity ML-Agents' Banana environment provided as a part of the
[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Introduction

In this environment we are an agent whose goal is to collect yellow bananas and avoid blue bananas. Each banana collection yields a reward with `+1` and `-1` being provided for a yellow and blue banana, respectively. 

At each instance, the agent can only perform one of **four actions** encoded with integers:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The **state space** is larger (`37` dimensions) and it contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

## Getting started

### Dependencies

All necessary Python dependencies are listed in the `requirements.txt` file. To install them please execute:
```bash
> pip install -r requirements.txt
```

### Downloading Unity's Banana env

Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

The notebook assumes that the Banana environment is provided for the Linux platform and its files are located in `./Banana_linux/`. Unless you're using Linux you might need to update the code in the Notebook.

## Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

To enjoy training of the banana comsumptor please open the `Navigation.ipyng` with the Jupyter Notebook.