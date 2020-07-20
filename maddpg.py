# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
import numpy as np
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, config, state_size, action_size, num_agents, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPGAgent(config, state_size, action_size) for i in range(num_agents)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        
    def resetNoise(self):
        [ddpg_agent.resetNoise() for ddpg_agent in self.maddpg_agent]

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors
    
    def get_critics(self):
        """get critics of all the agents in the MADDPG object"""
        critics = [ddpg_agent.critic for ddpg_agent in self.maddpg_agent]
        return critics

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""        
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, states, agent, noise=0.0):        
        """get target network actions from all the agents in the MADDPG object """        
        target_actions = [agent.target_act(state, noise) for state in states]
        #target_actions = torch.from_numpy(np.array(target_actions))
        target_actions = torch.stack(target_actions, dim=0)
        return target_actions

    def update(self, experiences, agent_number, gamma ): #, logger):
        """update the critics and actors of all the agents """
        return self.maddpg_agent[agent_number].learn(experiences, gamma, self.tau)
"""
        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
       # obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        #print('exp', experiences)
        states, actions, rewards, next_states, dones = experiences

        # ------------- critic ------------- #
        #obs_full = torch.stack(obs_full)
        #next_obs_full = torch.stack(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_states,agent)
        #target_actions = torch.cat(target_actions, dim=1)
        
        #?
        #target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)
        
        #print(states)
        #print(target_actions)
        with torch.no_grad():
            q_next = agent.target_critic(states, target_actions)
        
        y = rewards[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - dones[agent_number].view(-1, 1))
        #action = torch.cat(action, dim=1)
        #?
        #critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(states, actions)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # ------------- actor ------------- #
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input =  torch.stack([ agent.actor(state) for  state in states ], dim=0)
        
        #q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        #?
        #q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(states, q_input).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        #logger.add_scalars('agent%i/losses' % agent_number,
        #                   {'critic loss': cl,
        #                    'actor_loss': al},
        #                   self.iter)
        return [al, cl]

    def update_targets(self):
        "" "soft update targets" ""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
"""            
            




