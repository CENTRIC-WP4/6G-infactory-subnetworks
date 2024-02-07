import sys
import os
import numpy as np
import wandb
import socket
import setproctitle
import pickle
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from tbparse import SummaryReader

sys.path.append("../../")
from config import Args
from src.env.subnetwork_env import MultiAgentSubnetworks

def make_env(all_args):
    """Function to create environment"""
    numDev = 1 if not hasattr(all_args, 'numDev') else all_args.numDev # Compatibility with old logs
    env_args = {"scenario": all_args.scenario,
                "experiment": all_args.experiment_name,
                "problem": all_args.problem,
                "observation": all_args.observation,
                "reward": all_args.reward,
                "n_agent": all_args.n_agent,
                "n_channel": all_args.n_channel,
                "numDev": numDev,
                "num_steps" : all_args.episode_length,
                "seed" : all_args.seed,
                "dt" : all_args.dt}
    if all_args.problem == 'joint':
        env_args["u_level"] = all_args.u_level
    env = MultiAgentSubnetworks(env_args=env_args)
    return env

def make_runner(config, train=True):
    """This function maps a config file to the runner class"""
    alg = config["all_args"].algorithm_name
    if train:
        if alg in ['mat', 'mat_dec', 'mat_fed']:
            from src.runner_mat import Runner # Multi-agent transformer
            raise NotImplementedError
        elif alg in ['maddqn', 'maddqn_dec', 'maddqn_fed']:
            from src.runner_maddqn import Runner # Multi-agent double deep Q network
        elif alg in ['mappo', 'mappo_dec', 'mappo_fed']:
            from src.runner_mappo import Runner # Multi-agent proximal policy optimization
        else:
            raise NotImplementedError
    else:
        from src.runner import Runner

    return Runner(config)

def initialise_trainer(curr_run=None, alg='maddqn', prob='channel', observation='I_minmax', 
                       reward='rate', clutter='sparse', t_agg=1024, n_agent=20, n_channel=4, numDev=1):
    """This function maps training loops with runner and environment"""

    # Collect configuration
    all_args = Args(alg=alg, prob=prob, clutter='sparse', observation=observation, reward=reward, t_agg=t_agg, n_agent=n_agent, n_channel=n_channel, numDev=numDev)

    if n_agent < 10: 
        print('Warning!: Bad performance for a lower number of subnetworks.')

    # Setup processing schemes
    if all_args.cuda and torch.cuda.is_available():
        #print("Warming up GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        #print("Warming up CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Setup directory
    dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                            0] + "/results") / all_args.env_name / all_args.scenario / all_args.algorithm_name / all_args.experiment_name#"/../results") / all_args.env_name / all_args.scenario / all_args.algorithm_name / all_args.experiment_name
    if not dir.exists():
        os.makedirs(str(dir))

    # Setup logging
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                            project=all_args.env_name,
                            entity=all_args.user_name,
                            notes=socket.gethostname(),
                            name=str(all_args.algorithm_name) + "_" +
                                str(all_args.experiment_name) +
                                "_seed" + str(all_args.seed),
                            group=all_args.map_name,
                            dir=str(dir),
                            job_type="training",
                            reinit=True)
    else:
        if curr_run is None:
            if not dir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in dir.iterdir() if
                                    str(folder.name).startswith('run')]
                if len(exst_run_nums) == 0:
                    curr_run = 'run1'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = dir / curr_run
        if not run_dir.exists():
            print(f'\nCreating folder for current run: {run_dir}')
            os.makedirs(str(run_dir))
            with open(run_dir / 'config.pkl', 'wb') as file:
                pickle.dump(all_args, file)
        else:
            print(f'\nFound existing folder for current run: {run_dir}')
            with open(run_dir / 'config.pkl', 'rb') as file:
                all_args = pickle.load(file)
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))
    
    all_args.n_agent = n_agent
    all_args.experiment_name = clutter
    all_args.observation = observation
    
    # Start environments and runner
    env = make_env(all_args)
    eval_env = None
    config = {"all_args": all_args,
              "envs": env,
              "eval_envs": eval_env,
              "num_agents": all_args.n_agent,
              "device": device,
              "run_dir": run_dir}
    
    return config, curr_run, dir

def test_delays(delays=[10], alg='maddqn', run='c-rate', clutter='sparse', episodes=100, save_results=False):
    """Test different action transition delays for distributed greedy baseline"""

    fpath = Path('../results/evaluation/')
    if not fpath.exists():
        os.makedirs(str(fpath))
    fname = f'delay_{episodes}ep.mat'

    config, _, _ = initialise_trainer(run, alg=alg, clutter=clutter)
    mdict = {'delays': delays}
    for delay in delays:
        config["all_args"].max_switch_delay = delay
        evaluator = make_runner(config, train=False)
        print(f'\nRunning benchmarks for {episodes} episodes with max delay {config["all_args"].max_switch_delay}')
        _, rate_gdy, _, _ = evaluator.run_benchmark(episodes)
        mdict[f'{delay}'] = np.sort(rate_gdy.flatten())

    if save_results:
        savemat(fpath / fname, mdict)
        print(f'Saved results: {fpath / fname}')

def test_interference_levels(N_list=[10], alg='maddqn', run='c-rate', clutter='sparse', episodes=100, save_results=False):
    """Statistical analysis for interference levels in environment"""

    fpath = Path('../results/evaluation/')
    if not fpath.exists():
        os.makedirs(str(fpath))
    fname = f'interference_{clutter}_{episodes}ep.mat'
    mdict = {}
    mdict['N_list'] = N_list
    mdict['labels'] = ['Random', 'Greedy', 'CGC']
    for N in N_list:
        config, _, _ = initialise_trainer(run, alg=alg, clutter=clutter, observation='I', n_agent=N)
        evaluator = make_runner(config, train=False)
        print(f'\nRunning benchmarks for {episodes} episodes with {N} subnetworks')
        rate_rdm, rate_gdy, rate_cgc, _ = evaluator.run_interference(episodes)
        mdict[f'{N}_Random'] = np.sort(rate_rdm.flatten())
        mdict[f'{N}_Greedy'] = np.sort(rate_gdy.flatten())
        mdict[f'{N}_CGC'] = np.sort(rate_cgc.flatten())

    if save_results:
        savemat(fpath / fname, mdict)
        print(f'Saved results: {fpath / fname}')


def test_pretrained_model(alg, run, benchmark=False, clutter='sparse', episodes=2000, save_results=False):
    """Run pretrained models in a environment"""
    fpath = Path('../results/evaluation/')
    if not fpath.exists():
        os.makedirs(str(fpath))

    config, curr_run, dir = initialise_trainer(run, alg=alg, clutter=clutter)
    evaluator = make_runner(config, train=False)

    print(f'\nRunning {alg} ({run}) in {clutter} clutter for {episodes} episodes')
    rate_agent, measured_fps = evaluator.run_model(episodes)

    if save_results:
        fname = f'{alg}_{run}_{clutter}_{episodes}ep.mat'
        mdict = {'model': np.sort(rate_agent.flatten()), 'fps': measured_fps}
        savemat(fpath / fname, mdict)
        print(f'Saved results: {fpath / fname}')

    if benchmark:
        print(f'\nRunning benchmarks for {episodes} episodes')
        rate_rdm, rate_gdy, rate_cgc, _ = evaluator.run_benchmark(episodes)
        if save_results:
            fname = f'benchmarks_{clutter}_{episodes}ep.mat'
            mdict = {'rdm': np.sort(rate_rdm.flatten()), 'gdy': np.sort(rate_gdy.flatten()), 'cgc': np.sort(rate_cgc.flatten())}
            savemat(fpath / fname, mdict)
            print(f'Saved results: {fpath / fname}')

def test_n_sensitivity(n_agent_list=[10], reward='rate', clutter='sparse', episodes=2000, save_results=False, n_agent=20):
    """Test sensitivity of pretrained models versus number of subnetworks"""
    fpath = Path('../results/evaluation/')
    if not fpath.exists():
        os.makedirs(str(fpath))

    runs = [['maddqn', f'c-{reward}'], ['maddqn_dec', f'd-{reward}'], ['maddqn_fed', f'f-{reward}-512'],
            ['mappo', f'c-{reward}'], ['mappo_dec', f'd-{reward}'], ['mappo_fed', f'f-{reward}-512']]
    labels = ['C-MADDQN', 'D-MADDQN', "F-MADDQN: $T^{" + str('Agg') + "}=512$", 'C-MAPPO', 'D-MAPPO', "F-MAPPO: $T^{" + str('Agg') + "}=512$"]
    mdict = {}
    mdict["n_agent_list"] = n_agent_list
    mdict["labels"] = labels
    mdict["runs"] = runs

    for n_agent in n_agent_list:

        print(f'\nTesting {n_agent} agents:')

        for alg, run in runs:
            print(f'--Running {alg} ({run}) in {clutter} clutter for {episodes} episodes')
            config, _, _ = initialise_trainer(run, alg=alg, clutter=clutter, n_agent=n_agent)
            evaluator = make_runner(config, train=False)
            rate_agent, _ = evaluator.run_model(episodes)

            mdict[f'{n_agent}_{alg}_{run}_min'] = np.min(rate_agent)
            mdict[f'{n_agent}_{alg}_{run}_avg'] = np.mean(rate_agent)
            mdict[f'{n_agent}_{alg}_{run}_max'] = np.max(rate_agent)

        print(f'--Running bencmarks for {n_agent} agents')
        rate_rdm, rate_gdy, rate_cgc, _ = evaluator.run_benchmark(episodes)
        mdict[f'{n_agent}_rdm_min'] = np.min(rate_rdm)
        mdict[f'{n_agent}_rdm_avg'] = np.mean(rate_rdm)
        mdict[f'{n_agent}_rdm_max'] = np.max(rate_rdm)
        mdict[f'{n_agent}_gdy_min'] = np.min(rate_gdy)
        mdict[f'{n_agent}_gdy_avg'] = np.mean(rate_gdy)
        mdict[f'{n_agent}_gdy_max'] = np.max(rate_gdy)
        mdict[f'{n_agent}_cgc_min'] = np.min(rate_cgc)
        mdict[f'{n_agent}_cgc_avg'] = np.mean(rate_cgc)
        mdict[f'{n_agent}_cgc_max'] = np.max(rate_cgc)

    fname = f'{reward}_n_sensitivity.mat'
    savemat(fpath / fname, mdict)
    print(f'Saved results: {fpath / fname}')

def test_clutter_sensitivity(clutter_list=['sparse'], reward='rate', episodes=2000, save_results=False, n_agent=20):
    """Test sensitivity of pretrained models versus different types of clutter"""

    fpath = Path('../results/evaluation/')
    if not fpath.exists():
        os.makedirs(str(fpath))
    runs = [['maddqn', f'c-{reward}'], ['maddqn_dec', f'd-{reward}'], ['maddqn_fed', f'f-{reward}-512'],
            ['mappo', f'c-{reward}'], ['mappo_dec', f'd-{reward}'], ['mappo_fed', f'f-{reward}-512']]
    labels = ['C-MADDQN', 'D-MADDQN', "F-MADDQN: $T^{" + str('Agg') + "}=512$", 'C-MAPPO', 'D-MAPPO', "F-MAPPO: $T^{" + str('Agg') + "}=512$"]
    mdict = {}
    mdict["clutter_list"] = clutter_list
    mdict["labels"] = labels
    mdict["runs"] = runs

    for clutter in clutter_list:

        print(f'\nTesting {n_agent} agents in {clutter} clutter:')

        for alg, run in runs:
            print(f'--Running {alg} ({run}) in {clutter} clutter for {episodes} episodes')

            config, _, _ = initialise_trainer(run, alg=alg, clutter=clutter, n_agent=n_agent)
            evaluator = make_runner(config, train=False)
            rate_agent, _ = evaluator.run_model(episodes)

            mdict[f'{clutter}_{alg}_{run}_min'] = np.min(rate_agent)
            mdict[f'{clutter}_{alg}_{run}_avg'] = np.mean(rate_agent)
            mdict[f'{clutter}_{alg}_{run}_max'] = np.max(rate_agent)

        print(f'--Running bencmarks for {n_agent} agents in {clutter}')
        rate_rdm, rate_gdy, rate_cgc, _ = evaluator.run_benchmark(episodes)
        mdict[f'{clutter}_rdm_min'] = np.min(rate_rdm)
        mdict[f'{clutter}_rdm_avg'] = np.mean(rate_rdm)
        mdict[f'{clutter}_rdm_max'] = np.max(rate_rdm)
        mdict[f'{clutter}_gdy_min'] = np.min(rate_gdy)
        mdict[f'{clutter}_gdy_avg'] = np.mean(rate_gdy)
        mdict[f'{clutter}_gdy_max'] = np.max(rate_gdy)
        mdict[f'{clutter}_cgc_min'] = np.min(rate_cgc)
        mdict[f'{clutter}_cgc_avg'] = np.mean(rate_cgc)
        mdict[f'{clutter}_cgc_max'] = np.max(rate_cgc)

    fname = f'{reward}_clutter_sensitivity.mat'
    savemat(fpath / fname, mdict)
    print(f'Saved results: {fpath / fname}')

def load_log(run, dir, save_file=False, window_size=10):
    """This function is used to plot training process (real-time)"""

    run_dir = dir / run
    if os.path.exists(run_dir):
        log_dir = dir / run / 'logs'
        with open(dir / run / 'config.pkl', 'rb') as file:
            all_args = pickle.load(file)
    else:
        raise AssertionError(f'Cannot find log directory "{run}"')

    print(f'\nLoaded log: {all_args.n_agent} agents in {all_args.scenario} with {all_args.experiment_name} clutter.')
    print(f'            Algorithm {all_args.algorithm_name} for {all_args.problem}.')
    print(f'            Observation is {all_args.observation} and reward is {all_args.reward}.')

    file_paths = []
    categories = []
    for root, _, files in os.walk(log_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            categories.append(os.path.split(root)[-1])
            file_paths.append(filepath)

    print('Log categories:', categories)

    data = {}
    rew_fcat = []
    act_fcat = []
    for fpath, fcat in zip(file_paths, categories):
        reader = SummaryReader(fpath)
        df = reader.scalars
        if 'value' in df.keys():
            data[fcat] = df['value']
        if 'reward' in fcat and 'rewards' not in fcat:
            rew_fcat.append(fcat)
        if 'actions_ch' in fcat:
            act_fcat.append(fcat)

    # Performance
    fig, ax1 = plt.subplots(figsize=[6.4, 2/3*4.8])
    ax2 = ax1.twinx()
    fps = np.array(data['FPS'])
    mem = np.array(data['memory'])
    ax1.plot(np.arange(len(fps)), fps, c='C0')
    ax1.set_ylabel('Steps per second', c='C0')
    ax1.set_xlabel('Episode number')
    ax2.plot(np.arange(len(mem)), mem, c='C1')
    ax2.set_ylabel('Memory use [MB]', c='C1')
    ax1.set_title(f'{all_args.algorithm_name.upper()} training performance')

    # Actions
    fig, (ax1, ax2) = plt.subplots(2)
    hists = []
    label = []
    for i, fcat in enumerate(act_fcat):
        label.append(fcat.split('_ch')[-1])
        hists.append(np.array(data[fcat]))
    all_actions = np.sum(np.array(hists), axis=1)
    sample_actions = np.sum(np.array(hists)[:,-10:], axis=1)
    label = np.array(label,dtype=float)
    idx = np.argsort(label)
    for i in idx:
        ar = np.array(data[act_fcat[i]])
        m_pad = np.pad(ar, (window_size//2, window_size-1-window_size//2), mode='edge')
        m_smooth = np.convolve(m_pad, np.ones((window_size,))/window_size, mode='valid') 
        ax2.plot(m_smooth.T, label=str(int(label[i])))
    ax2.plot(np.sum(np.array(hists), axis=0).T, c='grey', alpha=1/2, label='All')
    ax2.set_ylabel('Number of indices')
    ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax2.get_legend().set_title('Action:')
    ax2.grid()
    ax2.set_xlabel('Episode number')


    ax1.grid(zorder=0)
    ax1.bar(label[idx]-0.2, all_actions[idx]/np.sum(all_actions), label='Total', width=0.4, zorder=3)
    ax1.bar(label[idx]+0.2, sample_actions[idx]/np.sum(sample_actions), label='Last 10', width=0.4, zorder=3)
    ax1.set_xlabel('Action')
    ax1.set_ylabel('Percentage of time')
    ax1.set_xticks(np.arange(len(label),dtype=int))
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax1.set_title(f'{all_args.algorithm_name.upper()} training action histogram')
    fig.tight_layout()

    # Training rewards
    fig, ax = plt.subplots()
    if 'epsilon' in data.keys():
        epsilon = np.array(data['epsilon'])
        ax2 = ax.twinx()
        ax2.set_ylabel('Epsilon', c='grey')
        ax2.plot(np.arange(len(epsilon)), epsilon, c='grey', alpha=1/2)
    
    window = np.ones(int(window_size))/float(window_size)
    episode_save = None
    for i, fcat in enumerate(rew_fcat):
        
        label = fcat.split('_')[0].capitalize()
        label = all_args.algorithm_name.upper() if label == 'Train' else label
        label = label.upper() if label == 'Cgc' else label

        # Plot averaged data
        ar = np.array(data[fcat])
        m_pad = np.pad(ar, (window_size//2, window_size-1-window_size//2), mode='edge')
        m_smooth = np.convolve(m_pad, np.ones((window_size,))/window_size, mode='valid') 
        xma = np.arange(len(m_smooth))
        ax.plot(xma, m_smooth, label=label, rasterized=True, c=f'C{i}')

        if 'train_reward' in fcat:
            episode_save = np.argmax(ar)
            reward_save = ar[episode_save]
    
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.855))
    ax.get_legend().set_title(f'{all_args.n_agent} subnetworks:')
    ax.grid()
    ax.set_xlabel('Episode number')
    ax.set_ylabel(f'Episode reward moving average')
    ax.set_title(f'Training: {((all_args.reward).replace("_", " ")).capitalize()} reward in {all_args.experiment_name} clutter with {all_args.n_channel} channels')

    if save_file:
        fig.savefig(dir / run / f'training_evaluation.pdf', bbox_inches='tight', dpi=100)

    print('\nNumber of episodes reached:', len(ar))
    print('Step time:', all_args.dt)
    print('Steps per episode:', all_args.episode_length)
    print('Number of episodes:', all_args.num_env_steps / all_args.episode_length)
    print('Mini batch:', all_args.num_mini_batch)

    if len(ar) >= all_args.epsilon_greedy_steps / all_args.episode_length:
        print('Espilon is relaxed and models was saved!')
        print(f'Model(s) was save on episode {episode_save}, with reward {reward_save}')
        print(f'Last epsilon value: {epsilon[-1]}')
    else:
        print('No models saved')

def export_mat(config, savepath='log'):
    """Function to export saved log file as .mat file"""

    run_dir = config["run_dir"]
    print(f'\nExporting log to .mat: {run_dir}')
    run = Path(run_dir).parts[-1]
    dir = run_dir.parent

    if os.path.exists(run_dir):
        log_dir = dir / run / 'logs'
        with open(dir / run / 'config.pkl', 'rb') as file:
            all_args = pickle.load(file)
    else:
        raise AssertionError(f'Cannot find log directory "{run}"')

    # Locate files and categories
    file_paths = []
    categories = []
    for root, _, files in os.walk(log_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            categories.append(os.path.split(root)[-1])
            file_paths.append(filepath)
    # Get data (rewards only)
    mdic = {}
    rew_fcat = []
    for fpath, fcat in zip(file_paths, categories):
        if 'reward' in fcat and 'rewards' not in fcat:
            reader = SummaryReader(fpath)
            df = reader.scalars
            rew_fcat.append(fcat)
            mdic[fcat] = np.array(df['value'])
    
    # Find the evaluation folder, and save file
    print('Categories logged:', rew_fcat)
    if savepath == 'log':
        mat_dir = log_dir / 'log.mat'
        savemat(mat_dir, mdic)
        print(f'The file was saved: {mat_dir}')
    elif savepath == 'evaluation':
        fpath = Path('../results/evaluation/')
        if not fpath.exists():
            os.makedirs(str(fpath))
        mat_dir = fpath / ('trn_' + config["all_args"].algorithm_name + '_' + run + '.mat')
        savemat(mat_dir, mdic)
        print(f'The file {mat_dir.parts[-1]} was saved in evaluation folder!')
    else:
        print('Did not understand the savepath! Please use log or evaluation.')
        return mdic

if __name__ == '__main__':

    # Run this to export .mat from a specific run!
    # <alg> and <run> must be defined accordingly to desired log
    # Pathsystem is given by: /results/factory/<alg>/sparse/<run>/
    alg = 'mappo' # <alg>
    run = 'run52' # <run>
    # The file can be output to the origin pathsysten, or to the /results/evaluation:
    out = 'log' # <log> or <evaluation>
    config, curr_run, dir = initialise_trainer(run, alg=alg)
    export_mat(config, out)