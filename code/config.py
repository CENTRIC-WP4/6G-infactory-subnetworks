class Args():
    """This class simulates a lazy argparser"""
    def __init__(self, alg="mat", prob="channel", clutter="sparse", observation='I_minmax', reward='rate', t_main=1, t_agg=1024, n_agent=20, numDev=1, n_channel=4):
 
        # Initialisation
        self.seed = 123
        self.algorithm_name = alg
        self.env_name = "subnetworks"
        self.scenario = "factory"
        self.experiment_name = clutter
        self.problem = prob
        self.observation = observation
        self.reward = reward

        # Environment parameters
        self.n_agent = n_agent # Number of agents
        self.numDev = numDev # Numbe of devices per agent
        self.n_channel = n_channel # Number of frequency channels
        self.u_level = n_channel # Discrete power levels
        self.episode_length = 200 # Steps per episode
        self.num_env_steps = 2000*self.episode_length  # Total number of steps
        self.dt = 0.005 # Environment sample frequency

        # Shared algorithm parameters
        self.num_mini_batch = 256 # Mini batch size
        self.lr = 0.001 # Learning rate
        self.max_memory_length = 64000 # Memory length
        self.max_switch_delay = 10 # Maximum channel switch delay
        self.cnt_model_update = 8 if 'mappo' in alg else 1 # How many steps between main update
        self.cnt_global_update = t_agg # How many steps between global aggregation

        # DDQN parameters
        self.epsilon_greedy_steps = 1500 * self.episode_length # Number of greedy steps
        self.gamma = 0.99 # Discount factor for past rewards
        self.epsilon = 1.0 # Epsilon greedy parameter
        self.epsilon_min = 0.001 # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0 # Maximum epsilon greedy
        self.cnt_target_update = 10 # How many steps between target update

        # MAPPO parameters
        self.ppo_epoch = 2 # Number of PPO learning epochs
        self.eps_clip = 0.2 # Clipping parameters

        # Logging parameters
        self.save_interval = 1
        self.log_interval = 1
        self.model_dir = None
        self.use_eval = True
        self.eval_interval = 20
        self.eval_episodes = 1
        self.set_first_episode = None
        self.set_episodes = None

        # Multi-agent transformer
        self.n_training_threads = 16
        self.critic_lr = 5e-4
        self.opti_eps = 1e-5
        self.weight_decay = 0
        self.gain = 0.01
        self.entropy_coef = 0.01
        self.dec_actor = False
        self.use_centralized_V = True
        self.encode_state = False
        self.n_block = 1
        self.n_embd = 24
        self.n_head = 1
        self.share_policy = False
        self.share_actor = False
        self.stacked_frames = 1
        self.use_stacked_frames = False
        self.hidden_size = 64
        self.layer_N = 2
        self.use_ReLU = True
        self.use_popart = False
        self.use_valuenorm = True
        self.use_feature_normalization = True
        self.use_orthogonal = True
        self.add_move_state = False
        self.add_local_obs = False
        self.add_distance_state = False
        self.add_enemy_action_state = False
        self.add_agent_id = False
        self.add_visible_state = False
        self.add_xy_state = False
        self.use_state_agent = False
        self.use_mustalive = True
        self.add_center_xy = False
        self.cuda = False
        self.cuda_deterministic = False
        self.n_rollout_threads = 1
        self.n_eval_rollout_threads = None
        self.n_render_rollout_threads = None
        self.user_name = 'xxx'
        self.use_wandb = False
        self.use_obs_instead_of_state = False
        self.use_naive_recurrent_policy = False
        self.use_recurrent_policy = False
        self.recurrent_N = 1
        self.data_chunk_length = 10
        self.use_clipped_value_loss = True
        self.clip_param = 0.2
        self.value_loss_coef = 1
        self.use_max_grad_norm = True
        self.max_grad_norm = 10.0
        self.use_gae = True
        self.gae_lambda = 0.95
        self.use_proper_time_limits = False
        self.use_huber_loss = True
        self.use_value_active_masks = True
        self.use_policy_active_masks = True
        self.huber_delta = 10
        self.use_linear_lr_decay = False
        self.save_gifs = False
        self.use_render = False
        self.render_episodes = 5
        self.ifi = 0.1