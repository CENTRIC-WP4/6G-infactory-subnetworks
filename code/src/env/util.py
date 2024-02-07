
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from infactory_env import env_subnetwork, centralizedColoring

def check_devLoc(N=2, M=2, steps=200, dt=0.005, channels=2, seed=123, timestep=0):
    env = env_subnetwork(numCell=N, numDev=M, group=channels, level=1, fname='Factory_values.mat', dt=dt, steps=steps+1, reward_type='rate', clutter='sparse', observation_type='I', seed=seed)
    env.generate_mobility()
    env.generate_channel()

    devLoc = env.deploy_devices()
    dLoc = np.repeat(devLoc[:,:,np.newaxis], env.numSteps, axis=2) \
        + np.repeat(env.loc, env.numDev, axis=1)
    allLoc = np.concatenate((dLoc, env.loc), axis=1) # [m1, m2, .., mM, n1, n2, ..., nN]

    fig, ax = plt.subplots()
    ax.scatter(0, 0, marker='x')
    i = 0
    for n in range(N):
        for m in range(M):
            ax.scatter(devLoc[0,i], devLoc[1,i], c=f'C{n}', label=f'N={n}')
            temp = np.array((devLoc[0,i], devLoc[1,i])) - np. array((0, 0))
            euclid_dist = np.round(np.sqrt(np. dot(temp. T, temp)),1)
            ax.text(devLoc[0,i], devLoc[1,i], f'n={n}\nm={m}\nr={euclid_dist}')
            i += 1
    ax.set_aspect(1)
    ax.grid()
    ax.set_xlim([-11/20 * env.cellDia, 11/20 * env.cellDia])
    ax.set_ylim([-11/20 * env.cellDia, 11/20 * env.cellDia])

    print('Distance check (allLoc):')
    i = 0
    for n in range(N):
        acc = np.array((allLoc[0,M*N+n,timestep], allLoc[1,M*N+n,timestep]))
        for m in range(M):
            dev = np.array((allLoc[0,i,timestep], allLoc[1,i,timestep]))
            temp = acc - dev
            euc_dist = np.round(np.sqrt(np. dot(temp. T, temp)),1)
            print(f'n={n}, m={m}, distance={euc_dist}')
            i += 1

def plot_test_delay(max_delays, N=50, num_steps_tot=1000, save_fig=False, steps=100, dt=0.01):
    """
    Compute action with centralised graph colouring

    Param :clutter_list: List of strings with desired clutter type (list)
    Param :N_list: List of integers, representing number of subnetworks (list)
    Param :N_plot_CDF: Boolean list representing which CDFs to plot (list)
    Param :num_steps_tot: Total number of environment steps per result (integer)
    Param :save_figs: Save figures in results if true (bool)
    """
    lim_sinr = np.inf
    lim_rate = np.inf

    data_sinr = []
    data_rate = []

    for i, max_delay in enumerate(max_delays):

        print(f'\nMax delay: {max_delay} (Run {i+1}/{len(max_delays)})')
        
        #env = env_subnetwork(numCell=N, group=4, fname='Factory_values.mat', steps=50+1, reward_type='rate', clutter='sparse')
        env = env_subnetwork(numCell=N, group=4, fname='Factory_values.mat', dt=dt, steps=steps+1, reward_type='rate', clutter='sparse', observation_type='I', seed=N)
        num_steps = 0
        switch_indicator = 1

        while num_steps < num_steps_tot:
        #for episode in range(episodes):
        
            env.generate_mobility()
            env.generate_channel()

            switch_idx = np.random.randint(0, max_delay, env.numCell)+1

            # Greedy initialisation
            o, _, _, _, = env.step(0, np.random.choice(env.channels, env.numCell))
            sinr_gdy, rate_gdy, obs_gdy = [], [], []
            action_gdy = env.channels[np.argmax(env.SINRAll, axis=1)]
            action_gdy_live = action_gdy.copy()

            for step in range(1, env.numSteps):

                # Greedy
                for n in range(env.numCell):
                    if switch_idx[n] == switch_indicator:
                        action_gdy_live[n] = action_gdy[n]
                switch_indicator += 1
                if switch_indicator > max_delay:
                    switch_indicator = 1

                o, _, _, _, = env.step(step, action_gdy_live)
                sinr_gdy.append(env.SINR)
                rate_gdy.append(env.sRate)  
                obs_gdy.append(o.flatten())

                action_gdy = env.channels[np.argmax(env.SINRAll, axis=1)]
                
                num_steps += env.numCell * env.numGroups

        data_sinr.append(np.sort(np.array(sinr_gdy).flatten()))
        data_rate.append(np.sort(np.array(rate_gdy).flatten()))

    if len(max_delays) > 1:
        fig1, ax1 = plt.subplots(2) # SINR
        fig, ax = plt.subplots() # SINR
        
        score = np.mean(np.array(data_sinr), axis=1)
        ax1[0].plot(max_delays, score)
        ax1[0].set_xlabel('Max delay steps')
        ax1[0].set_ylabel('Average SINR [dB]')
        ax1[0].grid()

        score_median = np.argwhere(score == score.flat[np.abs(score - np.mean(score)).argmin()])[0][0]
        idx = [np.argmin(score), score_median, np.argmax(score)]
        for i in idx:
            plot_sinr_gdy = np.sort(data_sinr[i])
            norm_sinr_gdy = 1. * np.arange(len(data_sinr[i])) / (len(data_sinr[i]) - 1)
            lim_sinr = np.min([lim_sinr, 
                               plot_sinr_gdy[np.argwhere(norm_sinr_gdy <= 0.01).flatten()[-1]]])
            ax1[1].plot(plot_sinr_gdy, norm_sinr_gdy, label=f'$\\tau_{max} = {max_delays[i]}$')
            ax.plot(plot_sinr_gdy, norm_sinr_gdy, label="$\\tau_{" + str('max') + "}$" + f" = {max_delays[i]}")
        ax1[1].set_xlabel('SINR [dB]')
        ax1[1].set_ylabel('CDF')
        ax1[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1[1].set_xlim([lim_sinr, ax1[1].get_xlim()[-1]])
        ax1[1].grid()
        ax.set_xlabel('SINR [dB]')
        ax.set_ylabel('CDF')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlim([lim_sinr, ax.get_xlim()[-1]])
        ax.grid()
        ax.set_title(f'CDF of SINR with greedy selection for {env.numGroups} channels in {env.clutter} clutter')

        fig1.tight_layout()

        fig2, ax2 = plt.subplots(2) # Rate
        
        score = np.mean(np.array(data_rate), axis=1)
        ax2[0].plot(max_delays, score)
        ax2[0].set_xlabel('Max delay steps')
        ax2[0].set_ylabel('Average rate [bps/Hz]')
        ax2[0].grid()

        score_median = np.argwhere(score == score.flat[np.abs(score - np.mean(score)).argmin()])[0][0]
        idx = [np.argmin(score), score_median, np.argmax(score)]
        for i in idx:
            plot_rate_gdy = np.sort(data_rate[i])
            norm_rate_gdy = 1. * np.arange(len(data_rate[i])) / (len(data_rate[i]) - 1)
            lim_rate = np.min([lim_rate, 
                               plot_rate_gdy[np.argwhere(norm_rate_gdy <= 0.01).flatten()[-1]]])
            ax2[1].plot(plot_rate_gdy, norm_rate_gdy, label=f'$\\tau_0 = {max_delays[i]}$')
        ax2[1].set_xlabel('Rate [bps/Hz]')
        ax2[1].set_ylabel('CDF')
        ax2[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2[1].set_xlim([lim_rate, ax2[1].get_xlim()[-1]])
        ax2[1].grid()
        
        fig2.tight_layout()

        if save_fig:
            #fig1.savefig(f'../../../results/greedy_max_delay_{env.numGroups}ch_sinr.pdf', bbox_inches='tight', dpi=100)
            #fig2.savefig(f'../../../results/greedy_max_delay_{env.numGroups}ch_rate.pdf', bbox_inches='tight', dpi=100)
            fig.savefig(f'../../../results/greedy_max_delay_{env.numGroups}ch_rate.pdf', bbox_inches='tight', dpi=100)
    else:
        raise NotImplementedError
    
    return max_delays[idx[-1]] # Return the best setting

def plot_environment_metrics(clutter_list, N_list, N_plot_CDF=None, num_steps_tot=1000, max_delay=10, save_fig=True, fpath='../../../results/', steps=100, dt=0.01):
    """
    Compute action with centralised graph colouring

    Param :clutter_list: List of strings with desired clutter type (list)
    Param :N_list: List of integers, representing number of subnetworks (list)
    Param :N_plot_CDF: Boolean list representing which CDFs to plot (list)
    Param :num_steps_tot: Total number of environment steps per result (integer)
    Param :save_figs: Save figures in results if true (bool)
    """
    print(f'Max delay: {max_delay}')
    if save_fig:
        with open(fpath + 'outputs.txt', 'a') as fname:
            fname.write(f'\nStarting new run! (plot_environment_metrics)\n   Clutter: {clutter_list}\n   Steps: {num_steps_tot}\n   Max delay: {max_delay}\n')

    for clutter in clutter_list:
        fig1, ax1 = plt.subplots(sum(N_plot_CDF), sharex=True) # SINR
        fig2, ax2 = plt.subplots(sum(N_plot_CDF), sharex=True) # Rate
        fig3, ax3 = plt.subplots()
        lim_sinr = np.inf
        lim_rate = np.inf

        _o_rdm = [[] for _ in range(4)]
        _o_gdy = [[] for _ in range(4)]
        _o_cgc = [[] for _ in range(4)]

        i = 0
        for j, (N, plot_CDF) in enumerate(zip(N_list, N_plot_CDF)):
            
            env = env_subnetwork(numCell=N, group=4, fname='Factory_values.mat', dt=dt, steps=steps+1, reward_type='rate', clutter=clutter, observation_type='I_minmax', seed=N)
            num_steps = 0
            switch_indicator = 1

            while num_steps < num_steps_tot:
            
                env.generate_mobility()
                env.generate_channel()

                switch_idx = np.random.randint(0, max_delay, env.numCell)+1
                
                # Random initialisation
                action_rdm = np.random.choice(env.channels, env.numCell)
                o, _, _, _, = env.step(0, action_rdm)
                sinr_rdm, rate_rdm, obs_rdm = [], [], [] 
                # Greedy initialisation
                o, _, _, _, = env.step(0, np.random.choice(env.channels, env.numCell))
                sinr_gdy, rate_gdy, obs_gdy = [], [], []
                action_gdy = env.channels[np.argmax(env.SINRAll, axis=1)]
                action_gdy_live = action_gdy.copy()
                # CGC initialisation
                o, _, _, _, = env.step(0, np.random.choice(env.channels, env.numCell))
                sinr_cgc, rate_cgc, obs_cgc = [], [], []
                action_cgc = centralizedColoring(env.rxPow[:env.numCell, :env.numCell, 0], env.numGroups)

                for step in range(1, env.numSteps):

                    # Random
                    o, _, _, _, = env.step(step, action_rdm)
                    sinr_rdm.append(env.SINR)
                    rate_rdm.append(env.sRate)
                    obs_rdm.append(o)

                    # Greedy (With precautions for pin-pon effect)
                    for n in range(env.numCell):
                        if switch_idx[n] == switch_indicator:
                            action_gdy_live[n] = action_gdy[n]
                    switch_indicator += 1
                    if switch_indicator > max_delay:
                        switch_indicator = 1
                    o, _, _, _, = env.step(step, action_gdy_live)
                    sinr_gdy.append(env.SINR)
                    rate_gdy.append(env.sRate)  
                    obs_gdy.append(o)
                    action_gdy = env.channels[np.argmax(env.SINRAll, axis=1)]
                    
                    # CGC
                    o, _, _, _, = env.step(step, action_cgc)
                    #print(o.shape)
                    sinr_cgc.append(env.SINR)
                    rate_cgc.append(env.sRate)     
                    obs_cgc.append(o)
                    action_cgc = centralizedColoring(env.rxPow[:env.numCell, :env.numCell, step], env.numGroups)   

                    num_steps += env.numCell * env.numGroups

            # Plot interference analysis
            print(np.array(obs_rdm).shape)
            for k in range(env.numGroups):
                _o_rdm[k].append(np.array(obs_rdm)[:,k])
                _o_gdy[k].append(np.array(obs_gdy)[:,k])
                _o_cgc[k].append(np.array(obs_cgc)[:,k])

            obs_rdm = list(np.array(obs_rdm).flatten())
            obs_gdy = list(np.array(obs_gdy).flatten())
            obs_cgc = list(np.array(obs_cgc).flatten())
            print(f'\n{N} subnetworks in {clutter} clutter')
            print('Random (min/avg/max)', np.min(obs_rdm), np.mean(obs_rdm), np.max(obs_rdm))
            print('Greedy (min/avg/max)', np.min(obs_gdy), np.mean(obs_gdy), np.max(obs_gdy))
            print('CentGC (min/avg/max)', np.min(obs_cgc), np.mean(obs_cgc), np.max(obs_cgc))
            if save_fig:
                with open(fpath + 'outputs.txt', 'a') as fname:
                    fname.write(f'\nInterference {N} subnetworks in {clutter} clutter\n   Random (min/avg/max): {np.min(obs_rdm)}, {np.mean(obs_rdm)}, {np.max(obs_rdm)}\n   Greedy (min/avg/max): {np.min(obs_gdy)}, {np.mean(obs_gdy)}, {np.max(obs_gdy)}\n   CentGC (min/avg/max): {np.min(obs_cgc)}, {np.mean(obs_cgc)}, {np.max(obs_cgc)}\n')
                
            bp = ax3.boxplot([obs_rdm, obs_gdy, obs_cgc], positions=np.array([j-0.25,j,j+0.25]), zorder=3, widths=0.2, patch_artist=True)
            col = ['C0', 'C1', 'C2']
            for n, c in enumerate(col):
                bp['boxes'][n].set_color(c)
                bp['boxes'][n].set_facecolor('white')
                bp['medians'][n].set_color(c)
                bp['fliers'][n].set_markeredgecolor(c)

            if plot_CDF:
                # Plot data SINR
                data = np.sort(np.array(sinr_rdm).flatten())
                plot_sinr_rdm = np.sort(data)
                norm_sinr_rdm = 1. * np.arange(len(data)) / (len(data) - 1)
                data = np.sort(np.array(sinr_gdy).flatten())
                plot_sinr_gdy = np.sort(data)
                norm_sinr_gdy = 1. * np.arange(len(data)) / (len(data) - 1)
                data = np.sort(np.array(sinr_cgc).flatten())
                plot_sinr_cgc = np.sort(data)
                norm_sinr_cgc = 1. * np.arange(len(data)) / (len(data) - 1)

                print(f'SINR analysis rdm: min {np.min(plot_sinr_rdm)}/ max {np.max(plot_sinr_rdm)}')
                print(f'SINR analysis gdy: min {np.min(plot_sinr_gdy)}/ max {np.max(plot_sinr_gdy)}')
                print(f'SINR analysis cgc: min {np.min(plot_sinr_cgc)}/ max {np.max(plot_sinr_cgc)}')
                if save_fig:
                    with open(fpath + 'outputs.txt', 'a') as fname:
                        fname.write(f'SINR {N} subnetworks in {clutter} clutter\n   Random (min/avg/max): {np.min(plot_sinr_rdm)}, {np.mean(plot_sinr_rdm)}, {np.max(plot_sinr_rdm)}\n   Greedy (min/avg/max): {np.min(plot_sinr_gdy)}, {np.mean(plot_sinr_gdy)}, {np.max(plot_sinr_gdy)}\n   CentGC (min/avg/max): {np.min(plot_sinr_cgc)}, {np.mean(plot_sinr_cgc)}, {np.max(plot_sinr_cgc)}\n')

                # Plot data rate
                data = np.sort(np.array(rate_rdm).flatten())
                plot_rate_rdm = np.sort(data)
                norm_rate_rdm = 1. * np.arange(len(data)) / (len(data) - 1)
                data = np.sort(np.array(rate_gdy).flatten())
                plot_rate_gdy = np.sort(data)
                norm_rate_gdy = 1. * np.arange(len(data)) / (len(data) - 1)
                data = np.sort(np.array(rate_cgc).flatten())
                plot_rate_cgc = np.sort(data)
                norm_rate_cgc = 1. * np.arange(len(data)) / (len(data) - 1)

                # Limits
                lim_sinr = np.min([lim_sinr, 
                                plot_sinr_rdm[np.argwhere(norm_sinr_rdm <= 0.01).flatten()[-1]],
                                plot_sinr_cgc[np.argwhere(norm_sinr_cgc <= 0.01).flatten()[-1]],
                                plot_sinr_gdy[np.argwhere(norm_sinr_gdy <= 0.01).flatten()[-1]]])
                lim_rate = np.min([lim_rate, 
                                plot_rate_rdm[np.argwhere(norm_rate_rdm <= 0.01).flatten()[-1]],
                                plot_rate_cgc[np.argwhere(norm_rate_cgc <= 0.01).flatten()[-1]],
                                plot_rate_gdy[np.argwhere(norm_rate_gdy <= 0.01).flatten()[-1]]])

                # Plotting
                if len(N_list) > 1:
                    ax1[i].plot(plot_sinr_rdm, norm_sinr_rdm, label='Random')
                    ax1[i].plot(plot_sinr_gdy, norm_sinr_gdy, label='Greedy')
                    ax1[i].plot(plot_sinr_cgc, norm_sinr_cgc, label='CGC')
                    ax1[i].grid()
                    ax1[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax1[i].get_legend().set_title(f'{N} subnetworks')
                    ax1[i].set_xlim([lim_sinr, ax1[i].get_xlim()[-1]])
                    ax2[i].plot(plot_rate_rdm, norm_rate_rdm, label='Random')
                    ax2[i].plot(plot_rate_gdy, norm_rate_gdy, label='Greedy')
                    ax2[i].plot(plot_rate_cgc, norm_rate_cgc, label='CGC')
                    ax2[i].grid()
                    ax2[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax2[i].get_legend().set_title(f'{N} subnetworks:')
                    ax2[i].set_xlim([lim_rate, ax2[i].get_xlim()[-1]])
                    if i == 0:
                        ax1[0].set_title(f'CDF of SINR for {env.numGroups} channels in {env.clutter} clutter')
                        ax1[-1].set_xlabel('SINR [dB]')
                        ax2[0].set_title(f'CDF of rate for {env.numGroups} channels in {env.clutter} clutter')
                        ax2[-1].set_xlabel('Rate [bps/Hz]')
                else:
                    ax1.plot(plot_sinr_rdm, norm_sinr_rdm, label='Random')
                    ax1.plot(plot_sinr_gdy, norm_sinr_gdy, label='Greedy')
                    ax1.plot(plot_sinr_cgc, norm_sinr_cgc, label='CGC')
                    ax1.grid()
                    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax1.get_legend().set_title(f'{N} subnetworks')
                    ax1.set_xlim([lim_sinr, ax1.get_xlim()[-1]])
                    ax2.plot(plot_rate_rdm, norm_rate_rdm, label='Random')
                    ax2.plot(plot_rate_gdy, norm_rate_gdy, label='Greedy')
                    ax2.plot(plot_rate_cgc, norm_rate_cgc, label='CGC')
                    ax2.grid()
                    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax2.get_legend().set_title(f'{N} subnetworks:')
                    ax2.set_xlim([lim_rate, ax2.get_xlim()[-1]])
                    ax1.set_title(f'CDF of SINR for {env.numGroups} channels in {env.clutter} clutter')
                    ax1.set_xlabel('SINR [dB]')
                    ax2.set_title(f'CDF of rate for {env.numGroups} channels in {env.clutter} clutter')
                    ax2.set_xlabel('Rate [bps/Hz]')
                
                if save_fig:
                    fig1.savefig(fpath + f'CDF_SINR_{len(N_list)}_{env.numGroups}ch_{clutter}_{max_delay}delay.pdf', bbox_inches='tight', dpi=100)
                    fig2.savefig(fpath + f'CDF_Rate_{len(N_list)}_{env.numGroups}ch_{clutter}_{max_delay}delay.pdf', bbox_inches='tight', dpi=100)
                i += 1

        ax3.grid()
        ax3.set_rasterized(True)
        ax3.set_xticks(np.arange(len(N_list)))
        ax3.set_xticklabels(N_list)
        ax3.set_title(f'Interference levels for {env.numGroups} channels in {clutter} clutter')
        legend_elements = [Line2D([0],[0],color='C0',marker='o',label='Random'),
                           Line2D([0],[0],color='C1',marker='o',label='Greedy'),
                           Line2D([0],[0],color='C2',marker='o',label='CGC')]
        ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.set_xlabel('Number of subnetworks')
        ax3.set_ylabel('Interference power [dB]')
        if save_fig:
            fig3.savefig(fpath + f'boxplot_{len(N_list)}_{env.numGroups}ch_{clutter}_{max_delay}delay.pdf', bbox_inches='tight', dpi=100)

        fig, ax = plt.subplots()
        for k in range(env.numGroups):
            bp = ax.boxplot([list(np.array(_o_rdm[k]).flatten()), list(np.array(_o_gdy[k]).flatten()), list(np.array(_o_cgc[k]).flatten())], positions=np.array([k-0.25,k,k+0.25]), zorder=3, widths=0.2, patch_artist=True)
            col = ['C0', 'C1', 'C2']
            for n, c in enumerate(col):
                bp['boxes'][n].set_color(c)
                bp['boxes'][n].set_facecolor('white')
                bp['medians'][n].set_color(c)
                bp['fliers'][n].set_markeredgecolor(c)

        ax.set_xticks(np.arange(env.numGroups))
        ax.set_xticklabels(np.arange(env.numGroups,dtype=int))