import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

from model import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256            # Minibatch size
BUFFER_SIZE = int(1e5)      # Replay buffer size
GAMMA = 0.99                # Discount factor
LR_ACTOR = 1e-4             # Learning rate of the actor
LR_CRITIC = 1e-3            # Learning rate of the critic
TAU = 1e-2                  # Parameter for soft update of target parameters
WEIGHT_DECAY_ACTOR = 0      # Weight decay for the actor
WEIGHT_DECAY_CRITIC = 0     # Weight decay for the critic
N_LEARN_UPDATES = 1         # Number of learning updates per step
N_TIME_STEPS = 1            # Learning every N_TIME_STEPS
NOISE_SCALE = 0.2           # Amount of noise without reduction
NOISE_REDUCTION = 0.9999    # Noise reduction factor

class MADDPGAgents:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of parallel agents
            random_seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        
        # Create agents
        self.agents = [Agent(state_size, action_size, num_agents, random_seed) for i in range(num_agents)]
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, time_step, state, actions, rewards, next_state, dones, is_learning):
        """Save experience in replay memory and use random sample from buffer to learn."""
                
        # add experience to memory
        experience = (state, actions, rewards, next_state, dones)
        self.memory.add(experience)
        
        # Learn every n_time_steps
        if (time_step % N_TIME_STEPS != 0) or (not is_learning):
            return
                
        if len(self.memory) > BATCH_SIZE:    
            for i in range(N_LEARN_UPDATES):
                for agent_number in range(self.num_agents):
                    samples = self.memory.sample()
                    self.learn(agent_number, samples)
                
    def act(self, state, add_noise):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state: Full state, combined for all agents
            add_noise: Add noise (True for training, False for testing)
        """
        state = torch.from_numpy(state).float().to(device)  
        actions = [agent.act(state_agent, add_noise) for agent, state_agent in zip(self.agents, state)]
        return actions
    
    def learn(self, agent_number, samples):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            samples (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """            
        states, actions, rewards, next_states, dones = to_tensor(samples)   
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
           
        next_actions_critic = torch.cat([self.agents[agent_i].act_target(next_states[:, agent_i, :]) \
                                  for agent_i in range(self.num_agents)], dim=1)

        next_states_critic = torch.cat([next_states[:, agent_i, :] \
                                  for agent_i in range(self.num_agents)], dim=1)
                       
        Q_targets_next = self.agents[agent_number].critic_target(next_states_critic, next_actions_critic)
                
        # Compute Q targets for current states
        Q_targets = rewards[:, agent_number].view(-1,1) + \
            (GAMMA * Q_targets_next.view(-1,1) * (1 - dones[:, agent_number].view(-1,1)))
                
        # Get predicted Q values from local critic model

        states_critic = torch.cat([states[:, agent_i, :] \
                                  for agent_i in range(self.num_agents)], dim=1)
        
        actions_critic = torch.cat([actions[:, agent_i, :] \
                                  for agent_i in range(self.num_agents)], dim=1)
               
        Q_expected = self.agents[agent_number].critic(states_critic, actions_critic)
                
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
             
        # Minimize the critic loss
        self.agents[agent_number].critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent_number].critic.parameters(), 1.0)
        self.agents[agent_number].critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        
        # Get predicted actions from local actor model
        actions_pred = self.agents[agent_number].actor(states[:, agent_number, :])
       
        # Add predicted actions as input to critic
        actions_critic = torch.cat([actions_pred if agent_i == agent_number else actions[:, agent_i, :] \
                          for agent_i in range(self.num_agents)], dim=1)
       
        # Compute actor loss using local critic model
        actor_loss = -self.agents[agent_number].critic(states_critic.detach(), actions_critic).mean()
                
        # Minimize the actor loss
        self.agents[agent_number].actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent_number].actor.parameters(), 1)
        self.agents[agent_number].actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.agents[agent_number].critic, self.agents[agent_number].critic_target, TAU)
        self.soft_update(self.agents[agent_number].actor, self.agents[agent_number].actor_target, TAU)   
            
            
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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        for agent_number in range(self.num_agents):
              self.agents[agent_number].reset()
                                
class Agent:
    def __init__(self, state_size, action_size, num_agents, random_seed):
        
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(num_agents*state_size, num_agents*action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.critic_target = Critic(num_agents*state_size, num_agents*action_size).to(device)

        self.noise = OUNoise(action_size, random_seed, scale=NOISE_SCALE)
        self.noise_factor = 1.0
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_CRITIC)

    def act(self, state, add_noise=False):
                                                       
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        
        if add_noise:  
            action += (self.noise.noise()*self.noise_factor)
            self.noise_factor *= NOISE_REDUCTION

        return np.clip(action, -1, 1)       
    
    def act_target(self, state):
                                                            
        with torch.no_grad():
            action = self.actor_target(state)
        
        return action        
                
    def reset(self):
        self.noise.reset()        
        
        
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_dimension, seed, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.seed = random.seed(seed)
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.deque = deque(maxlen=self.buffer_size)

    def add(self, experience):
        """add experience to the buffer"""
        self.deque.append(experience)

    def sample(self):
        """sample from the buffer"""
        samples = random.sample(self.deque, self.batch_size)

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)    

def transpose_list(mylist):
    return list(map(list, zip(*mylist)))    
    
def to_tensor(input_list):
    make_tensor = lambda x: torch.squeeze(torch.tensor(x, dtype=torch.float)).to(device)
    return map(make_tensor, zip(input_list))        