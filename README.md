# Project: Collaboration and Competition
This project is about training a deep reinforcment learning agent how to play tennis with another trained agent.

## About the environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Setting up the environment

### Download environment

The Environment can be downloaded at

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

### Install dependencies
Dependencies needed to get the programm running are gathered in the requirements.txt to install those execute the command:

```bash
 pip install requirements.txt
```

I had problems to install torch==0.4.0 so if you execute 

```bash
pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html 
```

before the command above it should work as expected

## Run Agent

Run `tennis-train.ipynb` to train the agent

## Files
* tennis.ipynb - template given by udacity, modified for a 1000 episodes random run for benchmarking
* tennis-train.ipynb - juptyter notebook for training of the agents
* model.py - classes for the networks used by actor and critic
* ddpg.py - class of the DDPG agent
* maddpg.py - small wrapper for handling multiple agents
* er.py - experience replay buffer
* ounoise.py - Ornstein-Uhlenbeck-Process
* config.json - configuration for multiple parameters (network, hyper, etc)
* solved_1314_actor0_local.pth - weights of the 1st agent when they reached a average over 100 episodes of > 0.5
* solved_1314_actor1_local.pth - weights of the 2nd agent when they reached a average over 100 episodes of > 0.5
* end_actor0_local.pth - weights of the trained 1st agent after 2000 episodes
* end_actor0_local.pth - weights of the trained 2nd agent after 2000 episodes

