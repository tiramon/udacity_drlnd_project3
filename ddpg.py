# individual network settings for each actor + critic pair
# see networkforall for details

#from networkforall import Network
from model import Actor, Critic
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from ounoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class DDPGAgent:
    def __init__(self, config, state_size, action_size):
        super(DDPGAgent, self).__init__()
        l1 = config['network']['hidden'];
        l2 = int(config['network']['hidden']/2)
        self.actor = Actor(state_size, action_size, config['seed']['agent'],l1,l2).to(device)
        self.critic = Critic(state_size, action_size, config['seed']['agent'],l1,l2).to(device)
        self.target_actor = Actor(state_size, action_size, config['seed']['agent'],l1,l2).to(device)
        self.target_critic = Critic(state_size, action_size, config['seed']['agent'],l1,l2).to(device)

        self.noise = OUNoise(action_size, scale=config['noise']['scale'], mu=config['noise']['mu'], sigma=config['noise']['sigma'], theta=config['noise']['theta'] )

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config['LR_ACTOR'])
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config['LR_CRITIC'], weight_decay=config['WEIGHT_DECAY'])

    def resetNoise(self):
        self.noise.reset()

    def act(self, obs, noise=0.0):        
        action = self.actor(obs) + noise*self.noise.noise()
        action = np.clip(action.detach().numpy(), -1, 1)
        return action

    def target_act(self, obs, noise=0.0):
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action
