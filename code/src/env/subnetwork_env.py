import numpy as np
import gym
from .multiagentenv import MultiAgentEnv
from src.trainer.utils.util import get_shape_from_obs_space

class MultiAgentSubnetworks(MultiAgentEnv):
    """This class creates the environment and wraps functionality for training algorithms"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scenario = kwargs["env_args"]["scenario"]        # Mobility model: Factory (string)
        self.experiment = kwargs["env_args"]["experiment"]    # Clutter type: Sparse/dense (string)
        self.problem = kwargs["env_args"]["problem"]          # Problem type: Channel/discrete power/continious power (string)
        self.observation_type = kwargs["env_args"]["observation"] # Observation type: Channel/sinr (string)
        self.reward_type = kwargs["env_args"]["reward"]       # Reward type: Shannon/pen (string)
        self.n_agents = kwargs["env_args"]["n_agent"]         # Number of subnetworks (integer)
        self.numDev = kwargs["env_args"]["numDev"]            # Number of devices per subnetwork (integer)
        self.n_channels = kwargs["env_args"]["n_channel"]     # Number of channels (integer)
        self.num_env_steps = kwargs["env_args"]["num_steps"]  # Number of timesteps (integer)
        self.seed = kwargs["env_args"]["seed"]
        self.dt = kwargs["env_args"]["dt"]
        self.u_levels = kwargs["env_args"]["u_level"] if self.problem == 'joint' else 1

        # Create environment
        if self.scenario == 'factory':
            from .infactory_env import env_subnetwork
        else:
            NotImplementedError

        self.env = env_subnetwork(numCell=self.n_agents, numDev=self.numDev, clutter=self.experiment, steps=self.num_env_steps + 1, dt=self.dt, problem=self.problem,
                                  group=self.n_channels, level=self.u_levels, reward_type=self.reward_type, observation_type=self.observation_type, seed=self.seed)

        # Create action space
        if self.problem == 'channel':
            self.n_actions = self.n_channels
        elif self.problem == 'power_d':
            raise NotImplementedError
        elif self.problem == 'power_c':
            raise NotImplementedError
        elif self.problem == 'joint':
            power_levels = np.linspace(start=self.env.Pmin, stop=self.env.Pmax, num=self.u_levels)
            self.comb_act = np.array(np.meshgrid(np.arange(self.n_channels), power_levels)).T.reshape(-1,2)
            self.n_actions = self.comb_act.shape[0]
            print(f'Problem type is joint ({self.n_channels}) channel and ({self.u_levels}) power allocation with {self.n_actions} actions')
        elif self.problem == 'repitition':
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.available_actions = np.arange(self.n_actions, dtype=int)

        # Create observation space
        if self.observation_type == 'channel':
            self.observation_space = [gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_channels,)) for n in range(self.n_agents)]
        elif self.observation_type == 'I' or self.observation_type == 'I_minmax' or self.observation_type == 'sinr' or self.observation_type == 'sinr_minmax':
            self.observation_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_channels,)) for n in range(self.n_agents)]
        else:
            raise NotImplementedError
        self.share_observation_space = self.observation_space
        self.obs_dim = get_shape_from_obs_space(self.observation_space[0])[0]

    def reset(self):
        self.env.generate_mobility()
        self.env.generate_channel()

    def _action_decoder(self, actions):
        if actions is None:
            return None, None
        else:
            if self.problem == 'channel':
                chl_action = actions.reshape(-1)
                pow_action = None
            elif self.problem == 'power_d':
                raise NotImplementedError
            elif self.problem == 'power_c':
                raise NotImplementedError
            elif self.problem == 'joint':
                chl_action = self.comb_act[actions][:,0].astype(int)
                pow_action = self.comb_act[actions][:,1]
            elif self.problem == 'repetition':
                raise NotImplementedError
            else:
                raise NotImplementedError
            return chl_action, pow_action

    def step(self, actions, time_index=0):
        #chl_action, pow_action = self._action_decoder(actions)
        #o, r, d, i = self.env.step(time_index=time_index, chl_action=chl_action, pow_action=pow_action)
        o, r, d, i = self.env.step(actions=actions, time_index=time_index)

        obs = o.reshape(1, self.n_agents, self.obs_dim)
        rewards = r.reshape(1, self.n_agents, 1)
        dones = np.array(self.n_agents, dtype=bool) * d

        return obs, rewards, dones, i