"""
Runner script for learning
Reference: https://github.com/PKU-MARL/Multi-Agent-Transformer
"""
import wandb
import time
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
import networkx as nx
import psutil

from src.trainer.utils.shared_buffer import SharedReplayBuffer
from src.trainer.mat_trainer import MATTrainer as TrainAlgo
from src.trainer.algorithm.transformer_policy import TransformerPolicy as Policy

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner():
    """Base class for training recurrent policies"""
    # =================== Initialisation and setup =================== #
    def __init__(self, config):
        self.all_args = config['all_args']
        self.env = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']     

        # Parameters
        self.max_switch_delay = self.all_args.max_switch_delay
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # Interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # Directory
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

        share_observation_space = self.env.share_observation_space[0] if self.use_centralized_V else self.env.observation_space[0]

        print("obs_space: ", self.env.observation_space)
        print("share_obs_space: ", self.env.share_observation_space)
        print("act_space: ", self.env.action_space)

        # Policy network
        self.policy = Policy(self.all_args,
                             self.env.observation_space[0],
                             share_observation_space,
                             self.env.action_space[0],
                             self.num_agents,
                             device=self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # Training algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device=self.device)
        
        # Replay buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.env.observation_space[0],
                                         share_observation_space,
                                         self.env.action_space[0],
                                         self.all_args.env_name)

    # =================== Training loop =================== #
    # todo: Implementation can only evaluate through a single rollout thread
    def run(self):
        
        # Transition and logging variables
        episodes = int(self.num_env_steps) // self.episode_length # Total number of episodes
        train_rewards, rdm_rewards, gdy_rewards, cgc_rewards = 0, 0, 0, 0
        done_episodes_rewards, rdm_episodes_rewards, gdy_episodes_rewards, cgc_episodes_rewards = [],[],[],[]
        switch_indicator, target_cnt, act_cnt = 1, 1, 0 # Switch indicator for action transition
        process = psutil.Process() # Used to estimate memory usage
        total_step = 0 # Counter for logging
        prev_reward = -np.inf # Conditional variable for saving
        self.warmup()

        for episode in range(episodes):

            start = time.time()
            obs, action_rdm, action_gdy, action_cgc = self.reset()
            action_gdy_live = action_gdy.copy()
            switch_idx = np.random.randint(0, self.max_switch_delay, self.num_agents) + 1
            action_history_live = [] # Used for logging

            for step in range(1, self.env.env.numSteps):
                total_step += 1

                # Sample actions
                values, action_alg, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                action_history_live.append(action_alg.flatten())

                # Collect observations and rewards
                obs, rewards, dones, infos = self.env.step(actions=action_alg, time_index=step)
                share_obs = obs.copy()
                data = obs, share_obs, rewards, dones, infos, self.env.available_actions, \
                       values, action_alg, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

                # Run random, greedy, and CGC
                for n in range(self.num_agents):
                    if switch_idx[n] == switch_indicator:
                        action_gdy_live[n] = action_gdy[n]
                switch_indicator += 1
                if switch_indicator > self.max_switch_delay:
                    switch_indicator = 1
                _, rew_rdm, _, _ = self.env.step(actions=action_rdm, time_index=step) # Random allocation
                _, rew_gdy, _, _ = self.env.step(actions=action_gdy_live, time_index=step) # Greedy allocation
                action_gdy = self.env.env.channels[np.argmax(self.env.env.SINRAll, axis=1)]
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

            # Compute return and update network
            self.compute()
            _ = self.train()
            
            # Logging information
            total_num_steps = (episode + 1) * self.episode_length
            if episode % self.log_interval == 0:

                # Calculate additional logging parameters
                end = time.time()
                meas_fps = self.episode_length / (end - start) # Averaged step per second
                rema_time = (((episodes * self.episode_length - total_num_steps) / meas_fps)/60)/60 # Estimated remaining time
                memory = process.memory_info().rss / 1024 / 1024 # Estimated memory use

                # Print logging information
                print(f'\n{episode}/{episodes} episodes, {total_num_steps}/{self.num_env_steps} timesteps, FPS {np.round(meas_fps,2)}, ETR {np.round(rema_time,2)} hours')
                print(f'Memory use was approximated to {memory} MB')
                print('Current trn/rdm/gdy/cgc:', np.round(done_episodes_rewards[-1][0],1), np.round(rdm_episodes_rewards[-1][0],1), np.round(gdy_episodes_rewards[-1][0],1), np.round(cgc_episodes_rewards[-1][0],1))
                print('Average trn/rdm/gdy/cgc:', np.round(np.mean(done_episodes_rewards),1), np.round(np.mean(rdm_episodes_rewards),1), np.round(np.mean(gdy_episodes_rewards),1), np.round(np.mean(cgc_episodes_rewards),1))

                # Save logging parameters
                train_info = {}
                train_info['FPS'] = meas_fps
                train_info['memory'] = memory
                train_info['train_reward'] = done_episodes_rewards[-1][0]
                train_info['random_reward'] = rdm_episodes_rewards[-1][0]
                train_info['greedy_reward'] = gdy_episodes_rewards[-1][0]
                train_info['cgc_reward'] = cgc_episodes_rewards[-1][0]
                if len(action_history_live) >= 1:
                    #counts, _ = np.histogram(np.array(action_history_live).flatten(), bins=np.arange(self.env.n_channels+1))
                    counts = np.array([np.sum(np.array(action_history_live).flatten() == k) for k in range(self.env.n_channels)])
                    act_cnt += counts
                    print(f'Actions hist: All {act_cnt}, {np.round(act_cnt/np.sum(act_cnt),3)}%, current {np.round(counts/np.sum(counts),3)}%')
                    for k in range(self.env.n_channels):
                        train_info[f"actions_ch{k}"] = counts[k]
                self.log_train(train_info, total_num_steps)

            # Checkpoint (Save if accumulative reward has increased)
            #if (episode % self.save_interval == 0 or episode == episodes - 1):
            #    self.save(episode, done_episodes_rewards[-1])

    # =================== Training functions =================== #
    def warmup(self):
        """Initialise environment and insert first observation into buffer"""
        obs, _, _, _ = self.reset()
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = self.buffer.obs[0].copy()
        self.buffer.available_actions[0] = self.env.available_actions

    def reset(self):
        """Reset and initialise environment"""
        self.env.reset()
        obs, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions-1, self.num_agents), time_index=0) # Model allocation
        action_rdm = np.random.choice(self.env.available_actions-1, self.num_agents) # Random allocation
        _, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions-1, self.num_agents), time_index=0) # Greedy allocation
        action_gdy = self.env.env.channels[np.argmax(self.env.env.SINRAll, axis=1)]
        _, _, _, _ = self.env.step(actions=np.random.choice(self.env.available_actions-1, self.num_agents), time_index=0) # Greedy allocation
        action_cgc = self.centralizedColoring(self.env.env.rxPow[:self.num_agents,:self.num_agents,0])

        return obs, action_rdm, action_gdy, action_cgc
    
    @torch.no_grad()
    def collect(self, step):
        """Sample action from trainer"""
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]))
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones = dones.reshape(self.n_rollout_threads, self.num_agents) # Only works for 1 rollout thread!!!
        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, None, active_masks,
                           available_actions)

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data"""
        self.trainer.prep_rollout()
        if self.buffer.available_actions is None:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]),
                                                         np.concatenate(self.buffer.available_actions[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer"""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos
    
    def save(self, episode, value, value_type='reward'):
        """Save policy's best actor and critic networks (Only output 1 file)"""
        if value_type == 'reward':
            if value >= self.prev_reward:
                print(f'\nEpisode {episode} reward improvement {np.round(value,2)} > {np.round(self.prev_reward,2)}')
                self.prev_reward = value
                self.policy.save(self.save_dir, 'reward')
        elif value_type == 'value':
            if value <= self.prev_value:
                print(f'\nEpisode {episode} value loss improvement {np.round(value,2)} < {np.round(self.prev_value,2)}')
                self.prev_value = value
                self.policy.save(self.save_dir, 'value')
        elif value_type == 'policy':
            if value <= self.prev_policy:
                print(f'\nEpisode {episode} policy loss improvement {np.round(value,2)} < {np.round(self.prev_policy,2)}')
                self.prev_policy = value
                self.policy.save(self.save_dir, 'policy')

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
            for k in range(1, self.env.n_channels):
                G.add_edge(n, Indx[k]) 
        d = nx.coloring.greedy_color(G, strategy='connected_sequential_bfs', interchange=True)
        act = np.asarray(list(d.values()),dtype=int)
        idx = np.argwhere(act >= self.env.n_channels).flatten()
        act[idx] = np.random.choice(np.arange(self.env.n_channels), len(idx))
        return act

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.policy.restore(model_dir)

    @torch.no_grad()
    def eval(self, eval_episodes=3, total_num_steps=None):
        """Evaluate rewards of current policy"""
        eval_episode_rewards = []
        one_episode_rewards = 0
        action_hist = np.zeros(len(self.env.available_actions), dtype=int)

        self.env.reset()
        temp_action = np.random.choice(self.env.available_actions, self.num_agents) - 1
        eval_obs, _, _, _ = self.env.step(temp_action)
        eval_share_obs = eval_obs.copy()
        self.buffer.available_actions[0] = self.env.available_actions
        ava = self.buffer.available_actions[0]

        eval_rnn_states = np.zeros((self.all_args.eval_episodes, self.num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)

        print(f'\nEvaluating with {eval_episodes} episodes')
        for episode in range(eval_episodes):
            self.env.reset()
            for step in range(self.episode_length):
                # Get actions
                self.trainer.prep_rollout()
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(ava),
                                            deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.all_args.eval_episodes))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.all_args.eval_episodes))
                action_hist += np.bincount(eval_actions[0,:,0], minlength=len(action_hist))

                # Observation and reward
                eval_obs, eval_rewards, dones, _ = self.env.step(eval_actions)
                eval_share_obs = eval_obs.copy()
                eval_dones = dones.reshape(self.n_rollout_threads, self.num_agents) # Only works for 1 rollout thread!!!
                eval_dones_env = np.all(eval_dones, axis=1)
                eval_rewards = np.mean(eval_rewards, axis=1).flatten()
                one_episode_rewards += eval_rewards

                # Self-attention
                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                            dtype=np.float32)

                if eval_dones_env:
                    eval_episode_rewards.append(one_episode_rewards)
                    one_episode_rewards = 0
                    print(f'Number of episodes evaluated: {episode+1}')
                    print(f'Action histogram: {action_hist}')

        if total_num_steps is not None:
            key_min = '/eval_min_episode_rewards'
            key_average = '/eval_average_episode_rewards'
            key_max = '/eval_max_episode_rewards'
            eval_infos = {key_min : [np.min(eval_episode_rewards)],
                          key_average : eval_episode_rewards,
                          key_max : [np.max(eval_episode_rewards)]}
            self.log_eval(eval_infos, total_num_steps)
            print('Logged evaluation')
        print(f'Evaluation of reward: min {np.min(eval_episode_rewards)}  / mean {np.mean(eval_episode_rewards)} / max {np.max(eval_episode_rewards)}')

    def log_eval(self, env_infos, total_num_steps):
        """Log evaluation info"""
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)