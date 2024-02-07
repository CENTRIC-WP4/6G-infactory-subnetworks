"""
Training loop for Multi-Agent Double Deep Q-Learning
Author: Bjarke Bak Madsen
"""
import wandb
import time
import os
import numpy as np
from tensorboardX import SummaryWriter
import networkx as nx
import psutil

from src.trainer.algorithm.maddqn import MAQN

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
            self.seed = 123
        np.random.seed(self.seed)

        # Simulation parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.use_wandb = self.all_args.use_wandb
        self.batch_size = self.all_args.num_mini_batch
        self.first_episode = self.all_args.set_first_episode
        self.episodes = self.all_args.set_episodes

        # DDQN parameters
        self.epsilon_greedy_steps = self.all_args.epsilon_greedy_steps
        self.gamma = self.all_args.gamma
        self.epsilon = self.all_args.epsilon
        self.epsilon_min = self.all_args.epsilon_min
        self.epsilon_max = self.all_args.epsilon_max
        self.max_memory_length = self.all_args.max_memory_length
        self.max_switch_delay = self.all_args.max_switch_delay
        self.cnt_target_update = self.all_args.cnt_target_update
        self.cnt_model_update = self.all_args.cnt_model_update
        self.cnt_global_update = self.all_args.cnt_global_update
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)  

        # Directory
        self.log_interval = self.all_args.log_interval
        self.model_dir = self.all_args.model_dir
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
        
        # Create agent(s)
        if self.algorithm_name == 'maddqn': # Centralised learning
            self.ddqn_agent = MAQN(agents=1, alpha=self.all_args.lr, gamma=self.gamma,\
                                n_actions=self.env.n_actions, batch_size=self.batch_size, fname=str(self.save_dir + f'/MAQN'),\
                                input_dims=self.env.n_channels, mem_size=self.max_memory_length, seed=self.seed, device=self.device)
            self.agent_idx = np.zeros(self.num_agents, dtype=int)
        elif self.algorithm_name == 'maddqn_dec': # Decentralised learning
            self.ddqn_agent = MAQN(agents=self.num_agents, alpha=self.all_args.lr, gamma=self.gamma,\
                                n_actions=self.env.n_actions, batch_size=self.batch_size, fname=str(self.save_dir + f'/MAQN_dec'),\
                                input_dims=self.env.n_channels, mem_size=self.max_memory_length, seed=self.seed, device=self.device)
            self.agent_idx = np.arange(self.num_agents, dtype=int)
        elif self.algorithm_name == 'maddqn_fed': # Federated learning
            self.ddqn_agent = MAQN(agents=self.num_agents, alpha=self.all_args.lr, gamma=self.gamma, use_federated=True,\
                                n_actions=self.env.n_actions, batch_size=self.batch_size, fname=str(self.save_dir + f'/MAQN_fed'),\
                                input_dims=self.env.n_channels, mem_size=self.max_memory_length, seed=self.seed, device=self.device)
            self.agent_idx = np.arange(self.num_agents, dtype=int)
        else:
            raise NotImplementedError

        # Continue training from a specific episode
        if self.first_episode is not None and self.episodes is not None:
            self.ddqn_agent.load_model()
            self.total_step = self.first_episode * self.episode_length
        else:
            self.total_step = 0
            self.first_episode = 0
            self.episodes = int(self.num_env_steps) // self.episode_length # Total number of episodes

    # =================== Training loop =================== #
    def run(self):

        print(f'Run: code/src/runner_maddqn.py')
        print(f'     episodes={self.episodes}, steps={self.env.env.numSteps-1}.')
        print(f'     seed={self.seed}.')

        np.random.seed(self.seed)

        # Transition and logging variables
        train_rewards, rdm_rewards, gdy_rewards, cgc_rewards = 0, 0, 0, 0
        done_episodes_rewards, rdm_episodes_rewards, gdy_episodes_rewards, cgc_episodes_rewards = [],[],[],[]
        switch_indicator, target_cnt, act_cnt = 1, 1, 0 # Switch indicator for action transition
        process = psutil.Process() # Used to estimate memory usage
        warmup_steps = 1000 # Number of steps before training starts
        prev_reward = -np.inf # Conditional variable for saving

        for episode in range(self.first_episode, self.episodes):
            
            start = time.time()
            obs, action_alg, action_rdm, action_gdy, action_cgc = self.reset()
            action_gdy_live = action_gdy.copy()
            switch_idx = np.random.randint(0, self.max_switch_delay, self.num_agents) + 1
            rdm_pct, action_history_live = [],[] # Used for logging

            for step in range(1, self.env.env.numSteps):
                self.total_step += 1

                # Epsilon-greedy with Q-models
                for n in range(self.num_agents):
                    if switch_idx[n] == switch_indicator:
                        action_gdy_live[n] = action_gdy[n]
                        if np.random.rand() < self.epsilon:
                            action_alg[n] = np.random.choice(self.env.available_actions, 1)
                            rdm_pct.append(1)
                        else:
                            action_alg[n] = self.ddqn_agent.choose_action(self.agent_idx[n], obs[0,n,:]) 
                            action_history_live.append(action_alg[n])
                            rdm_pct.append(0)
                if self.total_step > warmup_steps:
                    self.epsilon -= self.epsilon_interval/self.epsilon_greedy_steps
                    self.epsilon = max(self.epsilon, self.epsilon_min)

                # Perform observation and save it to replay buffer
                _obs, rewards, dones, _ = self.env.step(actions=action_alg, time_index=step)
                for n in range(self.num_agents):
                    if switch_idx[n] == switch_indicator:
                        self.ddqn_agent.remember(self.agent_idx[n], obs[0,n,:], action_alg[n], rewards[0,n,0], _obs[0,n,:])
                obs = _obs
                switch_indicator += 1
                if switch_indicator > self.max_switch_delay:
                    switch_indicator = 1

                # Run random, greedy, and CGC
                _, rew_rdm, _, _ = self.env.step(actions=action_rdm, time_index=step) # Random allocation
                _, rew_gdy, _, _ = self.env.step(actions=action_gdy_live, time_index=step) # Greedy allocation
                action_gdy = self.env.available_actions[np.argmax(self.env.env.SINRAll, axis=1)]
                _, rew_cgc, _, _ = self.env.step(actions=action_cgc, time_index=step) # CGC allocation
                action_cgc = self.centralizedColoring(self.env.env.rxPow[:self.num_agents,:self.num_agents,step])

                # Process all rewards for logging (Average of all steps in a episode)
                train_rewards += np.mean(rewards, axis=1).flatten()
                rdm_rewards += np.mean(rew_rdm, axis=1).flatten()
                gdy_rewards += np.mean(rew_gdy, axis=1).flatten()
                cgc_rewards += np.mean(rew_cgc, axis=1).flatten()
                if np.all(dones):
                    done_episodes_rewards.append(train_rewards/self.env.num_env_steps)
                    rdm_episodes_rewards.append(rdm_rewards/self.env.num_env_steps)
                    gdy_episodes_rewards.append(gdy_rewards/self.env.num_env_steps)
                    cgc_episodes_rewards.append(cgc_rewards/self.env.num_env_steps)
                    train_rewards, rdm_rewards, gdy_rewards, cgc_rewards = 0, 0, 0, 0

                # Start training:
                if target_cnt % self.cnt_model_update == 0 and self.total_step > warmup_steps:
                    self.ddqn_agent.learn()
                if target_cnt % self.cnt_target_update == 0 and self.total_step > warmup_steps:
                    self.ddqn_agent.update_network_parameters()
                if target_cnt % self.cnt_global_update == 0 and self.total_step > warmup_steps and self.ddqn_agent.use_federated:
                    self.ddqn_agent.aggregate_global()
                target_cnt += 1

            # Logging information
            total_num_steps = (episode + 1) * self.episode_length
            if episode % self.log_interval == 0:

                # Calculate additional logging parameters
                end = time.time()
                meas_fps = self.episode_length / (end - start) # Averaged step per second
                rema_time = (((self.episodes * self.episode_length - total_num_steps) / meas_fps)/60)/60 # Estimated remaining time
                memory = process.memory_info().rss / 1024 / 1024 # Estimated memory use

                # Print logging information
                print(f'\n{episode}/{self.episodes} episodes, {total_num_steps}/{self.num_env_steps} timesteps, FPS {np.round(meas_fps,2)}, ETR {np.round(rema_time,2)} hours')
                print(f'Memory use was approximated to {memory} MB')
                print(f'Epsilon greedy: Current measurement {np.round(100 * sum(rdm_pct)/len(rdm_pct),3)}%, next desired {np.round(100*self.epsilon,3)}%')
                print('Current trn/rdm/gdy/cgc:', np.round(done_episodes_rewards[-1][0],1), np.round(rdm_episodes_rewards[-1][0],1), np.round(gdy_episodes_rewards[-1][0],1), np.round(cgc_episodes_rewards[-1][0],1))
                print('Average trn/rdm/gdy/cgc:', np.round(np.mean(done_episodes_rewards),1), np.round(np.mean(rdm_episodes_rewards),1), np.round(np.mean(gdy_episodes_rewards),1), np.round(np.mean(cgc_episodes_rewards),1))

                # Save logging parameters
                train_info = {}
                train_info['FPS'] = meas_fps
                train_info['memory'] = memory
                train_info['epsilon'] = self.epsilon
                train_info['train_reward'] = done_episodes_rewards[-1][0]
                train_info['random_reward'] = rdm_episodes_rewards[-1][0]
                train_info['greedy_reward'] = gdy_episodes_rewards[-1][0]
                train_info['cgc_reward'] = cgc_episodes_rewards[-1][0]
                if len(action_history_live) >= 1:
                    counts = np.array([np.sum(np.array(action_history_live).flatten() == k) for k in range(self.env.n_actions)])
                    act_cnt += counts
                    print(f'Actions hist: All {act_cnt}, {np.round(act_cnt/np.sum(act_cnt),3)}%, current {len(action_history_live)}/{len(rdm_pct)} {np.round(counts/np.sum(counts),3)}%')
                    for k in range(self.env.n_actions):
                        train_info[f"actions_ch{k}"] = counts[k]
                self.log_train(train_info, total_num_steps)

            # Checkpoint (Save if epsilon has relaxed and episode reward has increased)
            if self.epsilon == self.epsilon_min and done_episodes_rewards[-1][0] > prev_reward:
                prev_reward = done_episodes_rewards[-1][0]
                self.ddqn_agent.save_model()
    
    # =================== Training functions =================== #
    def reset(self):
        """Reset and initialise environment"""
        self.env.reset()
        obs, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions, self.num_agents), time_index=0) # Model allocation
        action_alg = np.zeros(self.num_agents, dtype=int)
        for n in range(self.num_agents):
            if np.random.rand() < self.epsilon:
                action_alg[n] = np.random.choice(self.env.available_actions, 1)
            else:
                action_alg[n] = self.ddqn_agent.choose_action(self.agent_idx[n], obs[0,n,:]) 
        action_rdm = np.random.choice(self.env.available_actions, self.num_agents) # Random allocation
        _, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions, self.num_agents), time_index=0) # Greedy allocation
        action_gdy = self.env.available_actions[np.argmax(self.env.env.SINRAll, axis=1)]
        _, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions, self.num_agents), time_index=0) # Greedy allocation
        action_cgc = self.centralizedColoring(self.env.env.rxPow[:self.num_agents,:self.num_agents,0])

        return obs, action_alg, action_rdm, action_gdy, action_cgc
    
    def log_train(self, train_infos, total_num_steps):        
        #print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    # =================== Evaluation =================== #
    def centralizedColoring(self, interMat): 
        """
        Compute action with centralised graph colouring

        Param :interMat: Receive power matrix (np.array)
        Return :act: CGC actions (np.array)
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