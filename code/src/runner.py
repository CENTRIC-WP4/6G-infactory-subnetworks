"""
Runner script for evaluation of saved models
Author: Bjarke Bak Madsen
"""
import wandb
import time
import os
import numpy as np
from tensorboardX import SummaryWriter
import networkx as nx

from src.trainer.algorithm.maddqn import MAQN
from src.trainer.algorithm.mappo import MAPPO

class Runner():
    # =================== Initialisation and setup =================== #
    def __init__(self, config):
        self.all_args = config['all_args']
        self.env = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']
        self.seed = self.all_args.seed
        if self.seed is None:
            self.seed = 1
            print('Seed value set to 1')
        np.random.seed(self.seed)

        # Directory
        self.log_interval = self.all_args.log_interval
        self.model_dir = self.all_args.model_dir
        self.use_wandb = self.all_args.use_wandb
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # Simulation parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.batch_size = self.all_args.num_mini_batch

        # Create agent(s)
        self.max_memory_length = self.all_args.max_memory_length
        self.max_switch_delay = self.all_args.max_switch_delay
        self.cnt_model_update = self.all_args.cnt_model_update
        self.gamma = self.all_args.gamma
        if 'maddqn' in self.algorithm_name:
            self.epsilon_greedy_steps = self.all_args.epsilon_greedy_steps
            self.epsilon = self.all_args.epsilon
            self.epsilon_min = self.all_args.epsilon_min
            self.epsilon_max = self.all_args.epsilon_max
            self.cnt_target_update = self.all_args.cnt_target_update
            self.epsilon_interval = (self.epsilon_max - self.epsilon_min) 
            if self.algorithm_name == 'maddqn': # Centralised learning
                print('Single DDQN for centralised learning')
                self.agent = MAQN(agents=1, alpha=self.all_args.lr, gamma=self.gamma,\
                                    n_actions=self.env.n_actions, batch_size=self.batch_size, fname=str(self.save_dir + f'/MAQN'),\
                                    input_dims=self.env.n_channels, mem_size=self.max_memory_length, seed=self.seed, device=self.device)
                self.agent_idx = np.zeros(self.num_agents, dtype=int)
            elif self.algorithm_name == 'maddqn_dec': # Decentralised learning
                print('Multiple DDQNs for distributed learning')
                self.agent = MAQN(agents=self.num_agents, alpha=self.all_args.lr, gamma=self.gamma,\
                                    n_actions=self.env.n_actions, batch_size=self.batch_size, fname=str(self.save_dir + f'/MAQN_dec'),\
                                    input_dims=self.env.n_channels, mem_size=self.max_memory_length, seed=self.seed, device=self.device)
                self.agent_idx = np.arange(self.num_agents, dtype=int)
            elif self.algorithm_name == 'maddqn_fed': # Federated learning
                self.cnt_global_update = self.all_args.cnt_global_update
                print('Multiple DDQNs for federated learning')
                self.agent = MAQN(agents=self.num_agents, alpha=self.all_args.lr, gamma=self.gamma, use_federated=True,\
                                    n_actions=self.env.n_actions, batch_size=self.batch_size, fname=str(self.save_dir + f'/MAQN_fed'),\
                                    input_dims=self.env.n_channels, mem_size=self.max_memory_length, seed=self.seed, device=self.device)
                self.agent_idx = np.arange(self.num_agents, dtype=int)
        elif 'mappo' in self.algorithm_name:
            self.ppo_epoch = self.all_args.ppo_epoch
            self.eps_clip = self.all_args.eps_clip
            if self.algorithm_name == 'mappo': # Centralised learning
                self.agent = MAPPO(agents=1, alpha=self.all_args.lr, gamma=self.gamma, eps_clip=self.eps_clip, ppo_epochs=self.ppo_epoch,\
                                    n_actions=self.env.n_actions, batch_size=self.batch_size, fname=str(self.save_dir + f'/MAPPO'),\
                                    input_dims=self.env.n_channels, mem_size=self.max_memory_length, seed=self.seed, device=self.device)
                self.agent_idx = np.zeros(self.num_agents, dtype=int)
            elif self.algorithm_name == 'mappo_dec': # Decentralised learning
                print('Multiple PPOs for distributed learning')
                self.agent = MAPPO(agents=self.num_agents, alpha=self.all_args.lr, gamma=self.gamma, eps_clip=self.eps_clip, ppo_epochs=self.ppo_epoch,\
                                    n_actions=self.env.n_actions, batch_size=self.batch_size, fname=str(self.save_dir + f'/MAPPO_dec'),\
                                    input_dims=self.env.n_channels, mem_size=self.max_memory_length, seed=self.seed, device=self.device)
                self.agent_idx = np.arange(self.num_agents, dtype=int)
            elif self.algorithm_name == 'mappo_fed': # Federated learning
                print('Multiple PPOs for federated learning')
                self.agent = MAPPO(agents=self.num_agents, alpha=self.all_args.lr, gamma=self.gamma, eps_clip=self.eps_clip, ppo_epochs=self.ppo_epoch,\
                                    n_actions=self.env.n_actions, batch_size=self.batch_size, fname=str(self.save_dir + f'/MAPPO_fed'),\
                                    input_dims=self.env.n_channels, mem_size=self.max_memory_length, seed=self.seed, device=self.device, use_federated=True)
                self.agent_idx = np.arange(self.num_agents, dtype=int)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.episodes = int(self.num_env_steps) // self.episode_length

    # =================== Training loop =================== #
    def run_model(self, episodes=2000):
        """Run a pretrained model in environment"""

        # Load saved models
        self.agent.load_model()

        # Make sure that same seed is used and reset
        np.random.seed(self.seed)

        # Transition and logging variables
        switch_indicator = 1 # Switch indicator for action transition

        # Used for logging performance
        rate_agent = np.zeros([episodes, self.episode_length, self.num_agents], dtype=np.float64)
        measured_time = np.zeros([episodes], dtype=np.float64)

        for episode in range(episodes):
            
            obs, action_alg = self.reset_model()
            switch_idx = np.random.randint(0, self.max_switch_delay, self.num_agents) + 1

            step_times = []

            for step in range(1, self.env.env.numSteps):

                # Get action with policy
                for n in range(self.num_agents):
                    if switch_idx[n] == switch_indicator:
                        start = time.time()
                        action_alg[n] = self.agent.choose_action(self.agent_idx[n], obs[0,n,:])
                        end = time.time()
                        step_times.append(end - start)
                switch_indicator += 1
                if switch_indicator > self.max_switch_delay:
                    switch_indicator = 1

                # Perform observation and save achieved rates
                obs, _, _, _ = self.env.step(actions=action_alg, time_index=step)
                rate_agent[episode,step-1,:] = self.env.env.sRate
                
            # No logging saved, only printed
            total_num_steps = (episode + 1) * self.episode_length
            #meas_fps = self.episode_length / (end - start) # Averaged step per second
            measured_time[episode] = np.mean(step_times)
            #rema_time = (((episodes * self.episode_length - total_num_steps) / meas_fps)/60) # Estimated remaining time
            if episode % (episodes // 10) == 0:
                print(f'{episode}/{episodes} episodes, {total_num_steps}/{self.num_env_steps} timesteps, t={measured_time[episode]}')
                #print(f'{episode}/{episodes} episodes, {total_num_steps}/{self.num_env_steps} timesteps, FPS {np.round(meas_fps,2)}, ETR {np.round(rema_time,2)} minutes')
                #print(f'Episode average reward:', np.mean(rate_agent[episode,:,:]))

        print('Average rate', np.mean(rate_agent))
        print('Average execution time:', np.mean(measured_time))
        return rate_agent, measured_time

    def run_benchmark(self, episodes=2000):
        """Run benchmark algorithms in environment"""

        # Make sure that same seed is used and reset
        np.random.seed(self.seed)

        # Transition and logging variables
        switch_indicator = 1 # Switch indicator for action transition

        # Used for logging performance
        rate_rdm = np.zeros([episodes, self.episode_length, self.num_agents], dtype=np.float64)
        rate_gdy = np.zeros([episodes, self.episode_length, self.num_agents], dtype=np.float64)
        rate_cgc = np.zeros([episodes, self.episode_length, self.num_agents], dtype=np.float64)
        measured_time = np.zeros([2, episodes], dtype=np.float64)
        
        for episode in range(episodes):
            
            action_rdm, action_gdy, action_cgc = self.reset_benchmark()
            action_gdy_live = action_gdy.copy()
            switch_idx = np.random.randint(0, self.max_switch_delay, self.num_agents) + 1
            step_time = [[], []]

            for step in range(1, self.env.env.numSteps):

                # Get action with policy
                for n in range(self.num_agents):
                    if switch_idx[n] == switch_indicator:
                        action_gdy_live[n] = action_gdy[n]
                switch_indicator += 1
                if switch_indicator > self.max_switch_delay:
                    switch_indicator = 1

                # Perform observation and save achieved rates
                _, _, _, _ = self.env.step(actions=action_rdm, time_index=step) # Random allocation
                rate_rdm[episode,step-1,:] = self.env.env.sRate
                _, _, _, _ = self.env.step(actions=action_gdy_live, time_index=step) # Greedy allocation
                rate_gdy[episode,step-1,:] = self.env.env.sRate
                start = time.time()
                action_gdy = self.env.available_actions[np.argmax(self.env.env.SINRAll, axis=1)]
                end = time.time()
                step_time[0].append(end-start)
                _, _, _, _ = self.env.step(actions=action_cgc, time_index=step) # CGC allocation
                rate_cgc[episode,step-1,:] = self.env.env.sRate
                start = time.time()
                action_cgc = self.centralizedColoring(self.env.env.rxPow[:self.num_agents,:self.num_agents,step])
                end = time.time()
                step_time[1].append(end-start)

            # No logging saved, only printed
            total_num_steps = (episode + 1) * self.episode_length
            
            measured_time[0,episode] = np.mean(step_time[0])
            measured_time[1,episode] = np.mean(step_time[1])
            meas_fps = self.episode_length / (end - start) # Averaged step per second
            #rema_time = (((episodes * self.episode_length - total_num_steps) / meas_fps)/60) # Estimated remaining time
            if episode % (episodes // 10) == 0:
                print(f'{episode}/{episodes} episodes, {total_num_steps}/{self.num_env_steps} timesteps, t={measured_time[:,episode]}')
                #print(f'Episode average reward (rdm/gdy/cgc):', np.mean(rate_rdm[episode,:,:]), np.mean(rate_gdy[episode,:,:]), np.mean(rate_cgc[episode,:,:]))

        print('Average rate rdm/gdy/cgc:', np.mean(rate_rdm), np.mean(rate_gdy), np.mean(rate_cgc))
        print('Average execution time gdy/cgc:', np.mean(measured_time[0,:]), np.mean(measured_time[1,:]))
        return rate_rdm, rate_gdy, rate_cgc, measured_time

    def run_interference(self, episodes=2000):
        """Run benchmark algorithms in environment"""

        # Make sure that same seed is used and reset
        np.random.seed(self.seed)

        # Transition and logging variables
        switch_indicator = 1 # Switch indicator for action transition

        # Used for logging performance
        rate_rdm = np.zeros([episodes, self.episode_length, self.num_agents, self.env.n_channels], dtype=np.float64)
        rate_gdy = np.zeros([episodes, self.episode_length, self.num_agents, self.env.n_channels], dtype=np.float64)
        rate_cgc = np.zeros([episodes, self.episode_length, self.num_agents, self.env.n_channels], dtype=np.float64)
        measured_time = np.zeros([2, episodes], dtype=np.float64)
        
        for episode in range(episodes):
            
            action_rdm, action_gdy, action_cgc = self.reset_benchmark()
            action_gdy_live = action_gdy.copy()
            switch_idx = np.random.randint(0, self.max_switch_delay, self.num_agents) + 1
            step_time = [[], []]

            for step in range(1, self.env.env.numSteps):

                # Get action with policy
                for n in range(self.num_agents):
                    if switch_idx[n] == switch_indicator:
                        action_gdy_live[n] = action_gdy[n]
                switch_indicator += 1
                if switch_indicator > self.max_switch_delay:
                    switch_indicator = 1

                # Perform observation and save achieved rates
                obs, _, _, _ = self.env.step(actions=action_rdm, time_index=step) # Random allocation
                rate_rdm[episode,step-1,:] = obs
                obs, _, _, _ = self.env.step(actions=action_gdy_live, time_index=step) # Greedy allocation
                rate_gdy[episode,step-1,:] = obs
                start = time.time()
                action_gdy = self.env.available_actions[np.argmax(self.env.env.SINRAll, axis=1)]
                end = time.time()
                step_time[0].append(end-start)
                obs, _, _, _ = self.env.step(actions=action_cgc, time_index=step) # CGC allocation
                rate_cgc[episode,step-1,:] = obs
                start = time.time()
                action_cgc = self.centralizedColoring(self.env.env.rxPow[:self.num_agents,:self.num_agents,step])
                end = time.time()
                step_time[1].append(end-start)

            # No logging saved, only printed
            total_num_steps = (episode + 1) * self.episode_length
            
            measured_time[0,episode] = np.mean(step_time[0])
            measured_time[1,episode] = np.mean(step_time[1])
            meas_fps = self.episode_length / (end - start) # Averaged step per second
            #rema_time = (((episodes * self.episode_length - total_num_steps) / meas_fps)/60) # Estimated remaining time
            if episode % (episodes // 10) == 0:
                print(f'{episode}/{episodes} episodes, {total_num_steps}/{self.num_env_steps} timesteps, t={measured_time[:,episode]}')
                #print(f'Episode average reward (rdm/gdy/cgc):', np.mean(rate_rdm[episode,:,:]), np.mean(rate_gdy[episode,:,:]), np.mean(rate_cgc[episode,:,:]))

        print('Average rate rdm/gdy/cgc:', np.mean(rate_rdm), np.mean(rate_gdy), np.mean(rate_cgc))
        print('Average execution time gdy/cgc:', np.mean(measured_time[0,:]), np.mean(measured_time[1,:]))
        return rate_rdm.flatten(), rate_gdy.flatten(), rate_cgc.flatten(), measured_time

    # =================== Training functions =================== #
    def reset_model(self):
        """Reset and initialise environment for initiliased model"""
        self.env.reset()
        obs, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions, self.num_agents), time_index=0) # Model allocation
        action_alg = np.zeros(self.num_agents, dtype=int)
        for n in range(self.num_agents):
            action_alg[n] = self.agent.choose_action(self.agent_idx[n], obs[0,n,:]) 
        
        return obs, action_alg
    
    def reset_benchmark(self):
        """Reset and initialise environment for benchmarks"""
        self.env.reset()
        _, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions, self.num_agents), time_index=0) # Model allocation
        action_rdm = np.random.choice(self.env.available_actions, self.num_agents) # Random allocation
        _, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions, self.num_agents), time_index=0) # Greedy allocation
        action_gdy = self.env.available_actions[np.argmax(self.env.env.SINRAll, axis=1)]
        _, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions, self.num_agents), time_index=0) # Greedy allocation
        action_cgc = self.centralizedColoring(self.env.env.rxPow[:self.num_agents,:self.num_agents,0])

        return action_rdm, action_gdy, action_cgc
    
    # =================== Evaluation =================== #
    def centralizedColoring(self, interMat): 
        """
        Compute action with centralised graph colouring
        """
        N = interMat.shape[0]
        G = nx.Graph()
        G.add_nodes_from(np.linspace(0,N-1,num=N))   
        for n in range(N):
            dn = interMat[n,:]
            Indx = sorted(range(N), key=lambda k: dn[k])
            for k in range(1, self.env.n_actions):
                G.add_edge(n, Indx[k]) 
        d = nx.coloring.greedy_color(G, strategy='connected_sequential_bfs', interchange=True)
        act = np.asarray(list(d.values()),dtype=int)
        idx = np.argwhere(act >= self.env.n_actions).flatten()
        act[idx] = np.random.choice(np.arange(self.env.n_actions), len(idx))
        return act