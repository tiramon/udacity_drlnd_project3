from ddpg import DDPGAgent
import torch
import numpy as np
device = 'cpu'

class MADDPG:
    def __init__(self, config, state_size, action_size, num_agents, gamma=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(config, state_size, action_size) for i in range(num_agents)]
        
        self.gamma = gamma
        self.tau = tau
        
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
        target_actions = torch.stack(target_actions, dim=0)
        return target_actions

    def update(self, experiences, agent_number ): #, logger):
        """update the critics and actors of all the agents """
        return self.maddpg_agent[agent_number].learn(experiences, self.gamma, self.tau)
