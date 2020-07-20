# individual network settings for each actor + critic pair
# see networkforall for details

#from networkforall import Network
from model import Actor, Critic
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np
import torch.nn.functional as F



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

        self.noise = OUNoise(action_size, mu=config['noise']['mu'], sigma=config['noise']['sigma'], theta=config['noise']['theta'] )

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config['LR_ACTOR'])
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config['LR_CRITIC'])#, weight_decay=config['WEIGHT_DECAY'])

    def resetNoise(self):
        self.noise.reset()

    def act(self, obs, noise=0.0):        
        action = self.actor(obs) + noise*self.noise.noise()
        action = np.clip(action.detach().numpy(), -1, 1)
        return action

    def target_act(self, obs, noise=0.0):
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action
    
    #old code from drlnd2
    def learn(self, experiences, gamma, tau):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        cl = critic_loss.cpu().detach().item()
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #from https://github.com/hortovanyi/DRLND-Continuous-Control/blob/master/ddpg_agent.py
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        al = actor_loss.cpu().detach().item()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.target_critic, tau)
        self.soft_update(self.actor, self.target_actor, tau)  
        
        #al = actor_loss.cpu().detach().item()
        #cl = critic_loss.cpu().detach().item()
        return [al, cl]

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)