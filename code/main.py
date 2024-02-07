"""
Main script to setup environment and training scheme
Author: Bjarke Bak Madsen
"""
import sys
sys.path.append("../../")
from utils import initialise_trainer, make_runner, export_mat, test_pretrained_model, test_delays, test_interference_levels, test_n_sensitivity, test_clutter_sensitivity, load_log

if __name__ == '__main__':

    # Remark that evaluation/models are saved in ../results relative to this file
    # Main configuration of simulation (Further configuration can be found in config.py)
    alg = 'mappo' # Algorithm: maddqn, maddqn_dec, maddqn_fed, mappo, mappo_dec, mappo_fed
    reward = 'rate' # Reward type: rate, sinr, binary, composite_reward
    prob = 'channel' # Problem type: channel, joint
    observation = 'I' # Obsevation type: I, I_minmax, sinr, sinr_minmax (NOTICE: minmax is for N=20,M=1 for now!)
    channels = 4 # Number of frequency channels
    n_subnetworks = 20 # Number of subnetworks
    m_devices = 2 # Number of devices per subnetwork (NOT FULLY FUNCTIONAL!)

    config, curr_run, dir = initialise_trainer(None, alg=alg, prob=prob, reward=reward, t_agg=1024, n_channel=channels, n_agent=n_subnetworks, numDev=m_devices, observation=observation)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    export_mat(config)              # Export rewards (trn/rdm/gdy/cgc) to .mat file
    #test_pretrained_model('maddqn', 'c-rate', episodes=10, save_results=True, n_agent=30, benchmark=True)

    #episodes = 1000
    #test_delays([1,2,10,20], episodes=episodes, save_results=True)
    #test_interference_levels([10, 20, 30, 40, 50], clutter='sparse', episodes=episodes, save_results=True)
    #test_interference_levels([10, 20, 30, 40, 50], clutter='dense', episodes=episodes, save_results=True)
    #N_list = [10, 20, 30, 40, 50]
    #clutter_list = ['sparse0', 'sparse', 'sparse2', 'dense0', 'dense', 'dense2']
    #test_n_sensitivity(N_list, reward='rate', episodes=episodes)
    #test_n_sensitivity(N_list, reward='bina', episodes=episodes)
    #test_clutter_sensitivity(clutter_list, reward='rate', episodes=episodes)
    #test_clutter_sensitivity(clutter_list, reward='bina', episodes=episodes)

    """
    # Runner1
    test_pretrained_model('maddqn', 'c-rate', episodes=episodes, save_results=True, benchmark=True)
    test_pretrained_model('maddqn_dec', 'd-rate', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn', 'c-bina', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn_fed', 'f-bina-512', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_fed', 'f-rate-512', episodes=episodes, save_results=True)
    test_pretrained_model('mappo', 'c-bina', episodes=episodes, save_results=True)

    # Runner2
    test_pretrained_model('maddqn_fed', 'f-rate-512', episodes=episodes, save_results=True)
    test_pretrained_model('mappo', 'c-rate', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn_dec', 'd-bina', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_dec', 'd-bina', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_dec', 'd-rate', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_fed', 'f-bina-512', episodes=episodes, save_results=True)

    
    # Evaluate trained models
    episodes = 100
    test_pretrained_model('maddqn', 'c-rate', episodes=episodes, save_results=True, benchmark=True)
    test_pretrained_model('maddqn', 'c-sinr', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn', 'c-bina', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn_dec', 'd-rate', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn_dec', 'd-sinr', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn_dec', 'd-bina', episodes=episodes, save_results=True)
    #test_pretrained_model('maddqn_fed', 'f-rate-128', episodes=episodes, save_results=True)
    #test_pretrained_model('maddqn_fed', 'f-rate-256', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn_fed', 'f-rate-512', episodes=episodes, save_results=True)
    #test_pretrained_model('maddqn_fed', 'f-rate-1024', episodes=episodes, save_results=True)
    #test_pretrained_model('maddqn_fed', 'f-sinr-128', episodes=episodes, save_results=True)
    #test_pretrained_model('maddqn_fed', 'f-sinr-256', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn_fed', 'f-sinr-512', episodes=episodes, save_results=True)
    #test_pretrained_model('maddqn_fed', 'f-sinr-1024', episodes=episodes, save_results=True)
    #test_pretrained_model('maddqn_fed', 'f-bina-128', episodes=episodes, save_results=True)
    #test_pretrained_model('maddqn_fed', 'f-bina-256', episodes=episodes, save_results=True)
    test_pretrained_model('maddqn_fed', 'f-bina-512', episodes=episodes, save_results=True)
    #test_pretrained_model('maddqn_fed', 'f-bina-1024', episodes=episodes, save_results=True)
    test_pretrained_model('mappo', 'c-rate', episodes=episodes, save_results=True)
    test_pretrained_model('mappo', 'c-sinr', episodes=episodes, save_results=True)
    test_pretrained_model('mappo', 'c-bina', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_dec', 'd-rate', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_dec', 'd-sinr', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_dec', 'd-bina', episodes=episodes, save_results=True)
    #test_pretrained_model('mappo_fed', 'f-rate-128', episodes=episodes, save_results=True)
    #test_pretrained_model('mappo_fed', 'f-rate-256', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_fed', 'f-rate-512', episodes=episodes, save_results=True)
    #test_pretrained_model('mappo_fed', 'f-rate-1024', episodes=episodes, save_results=True)
    #test_pretrained_model('mappo_fed', 'f-sinr-128', episodes=episodes, save_results=True)
    #test_pretrained_model('mappo_fed', 'f-sinr-256', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_fed', 'f-sinr-512', episodes=episodes, save_results=True)
    #test_pretrained_model('mappo_fed', 'f-sinr-1024', episodes=episodes, save_results=True)
    #test_pretrained_model('mappo_fed', 'f-bina-128', episodes=episodes, save_results=True)
    #test_pretrained_model('mappo_fed', 'f-bina-256', episodes=episodes, save_results=True)
    test_pretrained_model('mappo_fed', 'f-bina-512', episodes=episodes, save_results=True)
    #test_pretrained_model('mappo_fed', 'f-bina-1024', episodes=episodes, save_results=True)

    ### Runner 1: Queue
    reward = 'rate'
    t_main = 8
    config, curr_run, dir = initialise_trainer('c-rate', alg='mappo', reward=reward)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-rate-128', alg='mappo_fed', reward=reward, t_agg=128)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-rate-256', alg='mappo_fed', reward=reward, t_agg=256)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-rate-512', alg='mappo_fed', reward=reward, t_agg=512)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop

    ### Runner 2: Queue
    reward = 'sinr'
    t_main = 8
    config, curr_run, dir = initialise_trainer('c-sinr', alg='mappo', reward=reward)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('d-sinr', alg='mappo_dec', reward=reward)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-sinr-128', alg='mappo_fed', reward=reward, t_agg=128)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-sinr-256', alg='mappo_fed', reward=reward, t_agg=256)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-sinr-512', alg='mappo_fed', reward=reward, t_agg=512)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
  

    ### Runner 3: Queue
    reward = 'binary'
    t_main = 8
    config, curr_run, dir = initialise_trainer('d-bina', alg='mappo_dec', reward=reward)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-bina-128', alg='mappo_fed', reward=reward, t_agg=128)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-bina-256', alg='mappo_fed', reward=reward, t_agg=256)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-bina-512', alg='mappo_fed', reward=reward, t_agg=512)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop


    ### Runner 4: Queue (mixed)
    config, curr_run, dir = initialise_trainer('c-bina', alg='mappo', reward='binary')
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('d-rate', alg='mappo_dec', reward='rate')
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-rate-1024', alg='mappo_fed', reward='rate', t_agg=1024)
    runner = make_runner(config)    # Create training loop as object
    runner.run()                    # Run the training loop
    config, curr_run, dir = initialise_trainer('f-sinr-1024', alg='mappo_fed', reward='sinr', t_agg=1024)
    runner = make_runner(config)    # Create training loop as object
    runner.run()    
    config, curr_run, dir = initialise_trainer('f-bina-1024', alg='mappo_fed', reward='binary', t_agg=1024)
    runner = make_runner(config)    # Create training loop as object
    runner.run()    
    """