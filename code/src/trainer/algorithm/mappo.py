"""
Vanilla Multi-Agent Proximal Policy Optimisation (MAPPO) class
Author: Bjarke Bak Madsen
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """A single actor-critic agent"""
    def __init__(self, input_dims, n_actions, seed, hidden_nodes=100):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dims, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, 1))
        
        self.actor = nn.Sequential(
            nn.Linear(input_dims, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, n_actions),
            nn.Softmax(dim=0))

    def forward(self): # This function is not used
        raise NotImplementedError

    def single_act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action

    def act(self, state):
        """Get an action for a state"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.detach(), action_logprob, state_val
    
    def evaluate(self, state, action):
        """Get evaluation for action and state"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class ReplayBuffer(object):
    """Replay buffer for (n_agent, state, state_value, action, action log probability, reward)"""
    def __init__(self, n_agents, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = np.zeros(n_agents, dtype=np.int32)
        self.n_agents = n_agents

        self.state_memory = np.zeros((n_agents, self.mem_size, input_shape))
        self.value_memory = np.zeros((n_agents, self.mem_size))
        self.action_memory = np.zeros((n_agents, self.mem_size))
        self.logprob_memory = np.zeros((n_agents, self.mem_size))
        self.reward_memory = np.zeros((n_agents, self.mem_size))

    def store_transition(self, n_agent, state, value, action, reward, logprob):
        """Store a transition in the replay buffer"""
        index = self.mem_cntr[n_agent] % self.mem_size

        self.state_memory[n_agent,index,:] = state
        self.value_memory[n_agent,index] = value
        self.action_memory[n_agent,index] = action
        self.logprob_memory[n_agent,index] = logprob
        self.reward_memory[n_agent,index] = reward
        self.mem_cntr[n_agent] += 1

    def sample_buffer(self, n_agent, batch_size):
        """Sample a minibatch from the replay buffer"""
        max_mem = min(self.mem_cntr[n_agent], self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[n_agent,batch,:]
        values = self.value_memory[n_agent,batch]
        actions = self.action_memory[n_agent,batch]
        logprobs = self.logprob_memory[n_agent,batch]
        rewards = self.reward_memory[n_agent,batch]

        return states, values, actions, logprobs, rewards
    
class MAPPO(object):
    """Multi-Agent Proximal Policy Optimisation"""
    def __init__(self, agents, alpha, gamma, eps_clip, ppo_epochs, n_actions, batch_size, input_dims,
                 use_federated=False, mem_size=200, fname='mappo_model.h5', seed=123, device=None):
        print(f'Alg: code/src/trainer/algorithm/mappo.py')
        print(f'     PPO with {agents} agent(s){" (federated!)." if use_federated else "."}')
        print(f'     {input_dims} observations and {n_actions} actions.')
        print(f'     ppo-epochs={ppo_epochs}, batch-size={batch_size}, learning-rate={alpha}.')
        self.device = torch.device("cpu") if device is None else device
        self.n_agents = agents
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.model_file = fname
        self.use_federated = use_federated

        self.eps_clip = eps_clip
        self.ppo_epochs = ppo_epochs

        #self.memory = RolloutBuffer(agents)
        self.memory = ReplayBuffer(agents, mem_size, input_dims)        
        self.policy = [ActorCritic(input_dims, n_actions, seed).to(self.device) for _ in range(agents)]
        self.policy_old = [ActorCritic(input_dims, n_actions, seed).to(self.device) for _ in range(agents)]
        self.policy_global = ActorCritic(input_dims, n_actions, seed).to(self.device) if self.use_federated else None
        self.optimizer = [optim.Adam([{'params': self.policy[n].actor.parameters(), 'lr': alpha},
                                      {'params': self.policy[n].critic.parameters(), 'lr': alpha}])
                                      for n in range(agents)] 
        self.loss_function = nn.MSELoss() 

    def remember(self, n_agent, state, state_val, action, reward, logprob):
        """Store a transition in the replay buffer"""
        self.memory.store_transition(n_agent, state, state_val, action, reward, logprob)

    def choose_action(self, n_agent, state):
        """Predict action using current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.policy_old[n_agent].single_act(state)

        return action
        
    def choose_actions(self, n_agent, state):
        """Predict action using current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old[n_agent].act(state)

        return action, action_logprob, state_val

    def learn(self):
        """Draw a random minibatch, and train agents"""
        for n_agent in range(self.n_agents):
            if self.memory.mem_cntr[n_agent] > self.batch_size:

                # Sample minibatch
                states, values, actions, logprobs, rewards = self.memory.sample_buffer(n_agent, self.batch_size)
                states = torch.FloatTensor(states).to(self.device)
                values = torch.FloatTensor(values).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                logprobs = torch.FloatTensor(logprobs).to(self.device)
                rewards = torch.FloatTensor(rewards).to(self.device)
                
                # Calculate advantages
                rewards = (rewards - rewards.mean()) / np.max([rewards.std(), 1e-7])
                advantages = rewards.detach() - values.detach()

                # Perform gradient descent in multiple epochs
                for _ in range(self.ppo_epochs):

                    # Evaluate old actions and values
                    new_logprobs, new_values, dist_entropy = self.policy[n_agent].evaluate(states, actions)
                    new_values = torch.squeeze(new_values)

                    # Probablity ratio
                    ratios = torch.exp(new_logprobs - logprobs.detach())

                    # Clipped surrogate loss
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                    loss = -torch.min(surr1, surr2)

                    # Gradient descent step
                    self.optimizer[n_agent].zero_grad()
                    loss.mean().backward()
                    self.optimizer[n_agent].step()

                # Update policy
                self.policy_old[n_agent].load_state_dict(self.policy[n_agent].state_dict())

    def aggregate_global(self):
        """Average local model weights, aggregate to global model, and update local models"""
        if self.use_federated:
            global_dict = self.policy_global.state_dict()
            for kd in global_dict.keys():
                global_dict[kd] = torch.stack([self.policy[n].state_dict()[kd] for n in range(self.n_agents)], 0).mean(0)
            self.policy_global.load_state_dict(global_dict)
            for n in range(self.n_agents):
                self.policy[n].load_state_dict(self.policy_global.state_dict())
            print('Updated with global aggregation!')

    def save_model(self):
        """Save all used agents"""
        for n_agent in range(self.n_agents):
            torch.save(self.policy_old[n_agent].state_dict(), self.model_file + f'{n_agent}_old')
            torch.save(self.policy[n_agent].state_dict(), self.model_file + f'{n_agent}')
        if self.use_federated:
            torch.save(self.policy_global.state_dict(), self.model_file + f'_GLOBAL')
        print(f'{2*self.n_agents + self.use_federated} model(s) was saved: {self.model_file}')

    def load_model(self):
        """Load all saved agents"""

        load_list = np.zeros(self.n_agents, dtype=int)
        i = 0
        for j in range(self.n_agents):
            load_list[j] = i
            if i+1 == 20:
                i = 0
            else:
                i += 1
        
        for n_agent, n_load in zip(range(self.n_agents), load_list):
            #print(f'Loaded save {n_load} into agent {n_agent}')
            checkpoint = torch.load(self.model_file + f'{n_load}_old', map_location=self.device)
            self.policy_old[n_agent].load_state_dict(checkpoint)
            self.policy_old[n_agent].eval()
            checkpoint = torch.load(self.model_file + f'{n_load}', map_location=self.device)
            self.policy[n_agent].load_state_dict(checkpoint)
            self.policy[n_agent].eval()
        if self.use_federated:
            checkpoint = torch.load(self.model_file + f'_GLOBAL')
            self.policy_global.load_state_dict(checkpoint)
            self.policy_global.eval()
        print(f'{2*self.n_agents + self.use_federated} model(s) was loaded: {self.model_file}')