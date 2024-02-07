"""
Multi-Agent Double Deep Q-Network (DDQN) class
Author: Bjarke Bak Madsen
"""
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Qnetwork(nn.Module):
    """A single main-target agent"""
    def __init__(self, input_dims, n_actions, seed, hidden_nodes=100):
        super(Qnetwork, self).__init__()
        #seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dims, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc4 = nn.Linear(hidden_nodes, n_actions)

    def forward(self, state):
        """Network call function, syntax: model(state) or model.forward(state)"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer(object):
    """Replay buffer for (n, obs, action, reward, last state)"""
    def __init__(self, n_agents, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = np.zeros(n_agents, dtype=np.int32)
        self.discrete = discrete
        self.state_memory = np.zeros((n_agents, self.mem_size, input_shape))
        self.new_state_memory = np.zeros((n_agents, self.mem_size, input_shape))
        self.action_memory = np.zeros((n_agents, self.mem_size))
        self.reward_memory = np.zeros((n_agents, self.mem_size))

    def store_transition(self, n_agent, state, action, reward, state_):
        """Store a transition in the replay buffer"""
        index = self.mem_cntr[n_agent] % self.mem_size
        self.state_memory[n_agent,index,:] = state
        self.new_state_memory[n_agent,index,:] = state_
        self.action_memory[n_agent,index] = action
        self.reward_memory[n_agent,index] = reward
        self.mem_cntr[n_agent] += 1

    def sample_buffer(self, n_agent, batch_size):
        """Sample a minibatch from the replay buffer"""
        max_mem = min(self.mem_cntr[n_agent], self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[n_agent,batch,:]
        actions = self.action_memory[n_agent,batch]
        rewards = self.reward_memory[n_agent,batch]
        states_ = self.new_state_memory[n_agent,batch,:]

        return states, actions, rewards, states_

class MAQN(object):
    """Centralised DDQN agent"""
    def __init__(self, agents, alpha, gamma, n_actions, batch_size, input_dims,
                 use_federated=False, mem_size=40000, fname='ddqn_model.h5', seed=123, device=None):
        print(f'Alg: code/src/trainer/algorithm/maddqn.py')
        print(f'     DDQN with {agents} agent(s) {"(federated!)" if use_federated else ""}')
        print(f'     {input_dims} observations and {n_actions} actions.')
        print(f'     gamma={gamma}, batch-size={batch_size}, learning-rate={alpha}.')
        self.device = torch.device("cpu") if device is None else device
        self.n_agents = agents
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_file = fname
        self.use_federated = use_federated

        self.memory = ReplayBuffer(agents, mem_size, input_dims, n_actions)
        self.q_eval = [Qnetwork(n_actions, input_dims, seed).to(self.device) for _ in range(self.n_agents)]
        self.q_target = [Qnetwork(n_actions, input_dims, seed).to(self.device) for _ in range(self.n_agents)]
        self.global_agent = Qnetwork(n_actions, input_dims, seed).to(self.device) if self.use_federated else None
        self.optimizer = [optim.Adam(self.q_eval[n].parameters(), lr=alpha) for n in range(self.n_agents)]
        
        self.update_network_parameters()        

    def remember(self, n_agent, state, action, reward, new_state):
        """Store a transition in the replay buffer"""
        self.memory.store_transition(n_agent, state, action, reward, new_state)

    def choose_action(self, n_agent, state):
        """Predict action using current model"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.q_eval[n_agent](state)
        action = np.argmax(action_values.cpu().data.numpy())
        return action

    def learn(self):
        """Draw a random minibatch, and train agents"""
        for n_agent in range(self.n_agents):
            if self.memory.mem_cntr[n_agent] > self.batch_size:
                
                state, action, reward, new_state = self.memory.sample_buffer(n_agent, self.batch_size)
                
                states = torch.FloatTensor(state).to(self.device)
                actions = torch.LongTensor(action).to(self.device)
                rewards = torch.FloatTensor(reward).to(self.device)
                new_states = torch.FloatTensor(new_state).to(self.device)

                q_values = self.q_eval[n_agent](states)
                new_q_values = self.q_eval[n_agent](new_states)
                new_q_state_values = self.q_target[n_agent](new_states)

                q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                new_q_value = new_q_state_values.gather(1, torch.max(new_q_values, 1)[1].unsqueeze(1)).squeeze(1)
                expexted_q_value = rewards + self.gamma * new_q_value

                loss = F.huber_loss(q_value, expexted_q_value)

                self.optimizer[n_agent].zero_grad()
                loss.backward()
                self.optimizer[n_agent].step()            

    def update_network_parameters(self):
        """Update target network with main network weigths"""
        for n_agent in range(self.n_agents):
            self.q_target[n_agent].load_state_dict(self.q_eval[n_agent].state_dict())

    def aggregate_global(self):
        """Average local model weights, aggregate to global model, and update local models"""
        if self.use_federated:
            global_dict = self.global_agent.state_dict()
            for kd in global_dict.keys():
                global_dict[kd] = torch.stack([self.q_eval[n].state_dict()[kd] for n in range(self.n_agents)], 0).mean(0)
            self.global_agent.load_state_dict(global_dict)
            for n in range(self.n_agents):
                self.q_eval[n].load_state_dict(self.global_agent.state_dict())
            self.update_network_parameters()
            print('Updated with global aggregation!')

    def save_model(self):
        """Save all used agents"""
        for n_agent in range(self.n_agents):
            torch.save(self.q_eval[n_agent].state_dict(), self.model_file + f'{n_agent}')
        if self.use_federated:
            torch.save(self.global_agent.state_dict(), self.model_file + f'{n_agent}_GLOBAL')
        print(f'{self.n_agents + self.use_federated} model(s) was saved: {self.model_file}')

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
            checkpoint = torch.load(self.model_file + f'{n_load}', map_location=self.device)
            self.q_eval[n_agent].load_state_dict(checkpoint)
            self.q_eval[n_agent].eval()
        if self.use_federated:
            checkpoint = torch.load(self.model_file + f'19_GLOBAL')
            self.global_agent.load_state_dict(checkpoint)
            self.global_agent.eval()
        print(f'{self.n_agents + self.use_federated} model(s) was loaded: {self.model_file}')