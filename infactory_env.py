# -*- coding: utf-8 -*-
"""
Deploy 6G in-robot subnetworks in a factory environment inspired by 3GPP and 5G ACIA.
Mobility Model: Predefined factory map, where subnetworks move in the alleys.
Channel Model: Path loss, correlated shadowing, small-scale fading with Jake's Doppler model
"""
import numpy as np
import os.path
import sys
import gym
from gym import spaces
from scipy.io import loadmat, savemat
from scipy.spatial.distance import cdist
from scipy.special import k0 as bessel
from scipy.stats import norm
import networkx as nx

class env_subnetwork(gym.Env):
    # =================== Initialisation of class =================== #
    def __init__(self, numCell=4, numDev=1, dt=0.005, clutter='sparse', problem='channel', 
                 group=4, level=4, rate=10, pOut=0.01, steps=100,
                 reward_type='constrained', observation_type='I',
                 fname='code/src/env/Factory_values.mat', seed=1):

        self.problem = problem
        self.fname = fname
        if seed is not None:
            np.random.seed(seed) # Everytime environment is called to reproduce from seed

        # Subnetworks and simulation
        self.numCell = numCell          # Number of subnetworks
        self.numDev = numDev                 # Number of devices in each subnetwork (NOT IMPLEMENTED!)
        self.cellDia = 1                # Radius of subnetworks [m]
        self.sampTime = dt              # Samplerate [s]
        self.transTime = dt             # Transition time [s]
        self.numSteps = steps           # Total number of environment steps
        self.simTime = int(self.numSteps * self.transTime) # Total simulation time [s]
        self.reward_type = reward_type  # Reward type [string]
        self.observation_type = observation_type # Observation type [string]
        self.updateT = self.numSteps * self.sampTime # Update time [s]

        # Requirements
        self.r_min = 10
        self.lambda_1 = 1
        self.lambda_2 = 1e-3

        # Mobility model and environment
        self.minDist = 1                # Minimum seperation distance [m]
        self.speed = 3                  # Robot movement speed [m/s]
        self.factoryArea = [180, 80]    # Size of factory [m x m]

        # Channel model
        self.corrDist = 10              # Correlated shadowing distance [m]
        self.mapRes = 0.1               # Correlated shadowing map grid reolution [m]
        self.clutter = clutter
        self.init_clutter(self.clutter)      # Parameters based on factory environment
        self.mapX = np.arange(0,self.factoryArea[0]+self.mapRes,step=self.mapRes) # Map x-coordinates
        self.mapY =  np.arange(0,self.factoryArea[1]+self.mapRes,step=self.mapRes) # Map y-coordinates
        self.codeBlockLength = 1        # Length of coded blocks
        self.codeError = 1e-3           # Codeword decoding error probability
        self.qLogE = norm.ppf(norm.cdf(self.codeError)) * np.log10(np.e) # Constant for channel rate penalty

        # Traffic model and radio communication
        self.numChannels = 1            # Number of channels
        self.numGroups = group          # Number of channel groups
        self.numLevels = level
        self.fc = 6e9                   # Carrier frequency [GHz]
        self.wave = 3e8 / self.fc       # Carrier wavelength [m]
        self.fd = self.speed / self.wave# Max Doppler frequency shift [Hz]
        self.totBW = self.numGroups*10*10**6# Total bandwidth [Hz]
        self.noiseFactor = 10           # Noise power factor
        self.txPow = -10                # TX power
        self.rho = np.float32(bessel(2*np.pi*self.fd*self.sampTime)) # Fading parameter

        self.channelBW = self.totBW/(self.numGroups*self.numChannels) # Bandwidth of channels
        self.noisePower = 10**((-174+self.noiseFactor+10*np.log10(self.channelBW))/10) # Power of the noise
        #self.intPowMax = -14.096877011071049 if self.clutter == 'dense' else -11.634749182802036
        
        self.intPowMax = -52.66387821883611 if 'dense' in self.clutter else -48.10078099565358
        self.intPowNorm = self.intPowMax - 10*np.log10(self.noisePower) # Min-max normalisation constant

        self.sinrMax = 43.95373571433217 if 'dense' in self.clutter else 43.73062810851178
        self.sinrMin = 11.380395703242927 if 'dense' in self.clutter else 4.784518853326086
        self.sinrNorm = self.sinrMax - self.sinrMin

        # Action space
        self.powerLevels = np.array([self.txPow]) # Initial power levels for subnetworks [dBm]
        self.channels = np.array([i for i in range(self.numGroups)]) # Set of channels
        self.combAction = np.array(np.meshgrid(self.channels, self.powerLevels)).T.reshape(-1,2) # Combined action array

        # Optimisation constraints and reward function
        #self.reqRate = np.full(self.numCell, rate) # Minimum throughputs (Requirement) [Mbps]
        #self.pOutage = np.full(self.numCell, pOut) # Outage probabilities (Requirement) 
        #self.Pmax = 0 # Maximum sum of transmission power [dBm]
        #self.w_next = np.zeros([self.numCell, self.numDev]) # Initial weight factors for reliability reward signals
        #self.p_next = np.zeros(self.numCell) # Initial weight factors for power reward signals

        self.rate_req = np.full([self.numCell, self.numDev], 11)
        self.SINR_req = 10 * np.log10(2**self.rate_req - 1)
        self.Pmax = 0 # [dBm]
        self.Pmin = -10 # [dBm]

        print(f'Env: code/src/env/infactory_env.py')
        print(f'     Factory with n={self.numCell} subnetworks and m={self.numDev} devices in {self.clutter} clutter.')
        print(f'     {self.problem.capitalize()} allocation for k={self.numGroups} channels and u={self.numLevels} power levels.')
        print(f'     Action space for channels={self.channels} and power={self.powerLevels}.')

        if self.problem == 'joint':
            self.Plevels = np.linspace(start=self.Pmin, stop=self.Pmax, num=self.numLevels)
            self.comb_act = np.array(np.meshgrid(np.arange(self.numGroups), self.Plevels)).T.reshape(-1,2)
            print(f'     Joint={self.comb_act}.')

        self.generate_factory_map()

    def init_clutter(self, clutter):
        """
        Get clutter parameters based on sparse or dense scenario.
        """
        self.clutType = clutter         # Type of clutter (sparse or dense)
        self.shadStd = [4.0]            # Shadowing std (LoS)
        if clutter == 'sparse0':
            self.clutSize = 10.0        # Clutter element size [m]  
            self.clutDens = 0.1         # Clutter density [%]
            self.shadStd.append(5.7)    # Shadowing std (NLoS)
        elif clutter == 'sparse':
            self.clutSize = 10.0        # Clutter element size [m]  
            self.clutDens = 0.2         # Clutter density [%]
            self.shadStd.append(5.7)    # Shadowing std (NLoS)
        elif clutter == 'sparse2':
            self.clutSize = 10.0        # Clutter element size [m]  
            self.clutDens = 0.35         # Clutter density [%]
            self.shadStd.append(5.7)    # Shadowing std (NLoS)
        elif clutter == 'dense0':
            self.clutSize = 2.0         # Clutter element size [m]  
            self.clutDens = 0.45         # Clutter density [%]
            self.shadStd.append(7.2)    # Shadowing std (NLoS)
        elif clutter == 'dense':
            self.clutSize = 2.0         # Clutter element size [m]  
            self.clutDens = 0.6         # Clutter density [%]
            self.shadStd.append(7.2)    # Shadowing std (NLoS)
        elif clutter == 'dense2':
            self.clutSize = 2.0         # Clutter element size [m]  
            self.clutDens = 0.8        # Clutter density [%]
            self.shadStd.append(7.2)    # Shadowing std (NLoS)
        else:
            raise NotImplementedError

    # =================== Mobility Model Functions =================== #
    def generate_factory_map(self, fname='code/src/env/Factory_values.mat'):
        """
        Generate the predefined factory map. All intersections (path points) is defined
        as points in a plane, and the matrix links define the legal moves between points.
        A global parameter on the shadowing map will be fetched from a local file.
        
        Param :fname: File name for shadowing values (string)
        Return :path_points: Waypoints (list)
        Return :link: Legal moves between waypoints (np.array)
        """
        path_points = [[1, 79],           [48.5, 79], [51.5, 79], [91, 79], [94, 79],                                                        [179, 79],
                                [4, 76],  [48.5, 76], [51.5, 76], [91, 76], [94, 76],                                             [176, 76], 
                                          [48.5, 59], [51.5, 59], [91, 59], [94, 59],                       [161, 59], [164, 59], [176, 59],
                                          [48.5, 56], [51.5, 56], [91, 56], [94, 56],                       [161, 56], [164, 56],            [179, 56],
                                [4, 39],  [48.5, 39], [51.5, 39], [91, 39], [94, 39], [131, 39], [134, 39], [161, 39], [164, 39], 
                       [1, 36],           [48.5, 36], [51.5, 36], [91, 36], [94, 36], [131, 36], [134, 36], [161, 36], [164, 36],
                                                                                                 [134, 24], [161, 24], 
                                                                                      [131, 21],                       [164, 21]]
        points = len(path_points)
        link = np.zeros((points, points))
        link[0,35] = 1
        link[1,[0,7]] = 1
        link[2,1] = 1
        link[3,[2,9]] = 1
        link[4,3] = 1
        link[5,4] = 1
        link[6,7] = 1
        link[7,[8,12]] = 1
        link[8,[2,9]] = 1
        link[9,[10,14]] = 1
        link[10,[4,11]] = 1
        link[11,18] = 1
        link[12,19] = 1
        link[13,[8,12]] = 1
        link[14,[13,21]] = 1
        link[15,[10,14]] = 1
        link[16,[15,23]] = 1
        link[17,16] = 1
        link[18,17] = 1
        link[19,[20,27]] = 1
        link[20,[13,21]] = 1
        link[21,[22,29]] = 1
        link[22,[15,23]] = 1
        link[23,[24,33]] = 1
        link[24,[17,25]] = 1
        link[25,5] = 1
        link[26,6] = 1
        link[27,[26,36]] = 1
        link[28,[20,27]] = 1
        link[29,[28,38]] = 1
        link[30,[22,29]] = 1
        link[31,[30,40]] = 1
        link[32,31] = 1
        link[33,[32,42]] = 1
        link[34,[24,33]] = 1
        link[35,36] = 1
        link[36,37] = 1
        link[37,[28,38]] = 1
        link[38,39] = 1
        link[39,[30,40]] = 1
        link[40,[41,46]] = 1
        link[41,[32,42]] = 1
        link[42,[43,45]] = 1
        link[43,34] = 1
        link[44,41] = 1
        link[45,44] = 1
        link[46,47] = 1
        link[47,43] = 1

        self.path_points = path_points
        self.link = link

        # Load shadowing values
        if 'cMapFFT' not in globals() and os.path.exists(f'./{self.fname}'):
            dic = loadmat(self.fname)
            global cMapFFT
            cMapFFT = dic['data']

    def generate_mobility(self):
        """
        Random generation of states for every timestep. Initialisation is random on legal
        map paths, where robots are seperated with minimum 3 meters. The task of a robot
        is random, based on legal moves. Collisions between robots with a common destination
        are avoided by stopping every robot within minimum seperation distance of the robot
        with the shortest distance to the destination.
                
        Param :path_points: Waypoints (list)
        Param :link: Legal moves between waypoints (np.array)
        Return :loc: State locations (np.array)
        """
        # Initialise random start positions
        N = round(self.updateT/self.sampTime)
        self.loc = np.zeros([2,self.numCell,N],dtype=np.float64)
        orientation = np.zeros([2,self.numCell,N],dtype=np.float64)
        task_list = np.zeros(self.numCell, dtype=np.int16)
        maxItr = 10000
        nValid = 0
        itr = 0
        while nValid < self.numCell:
            randPoint = np.random.randint(0, len(self.path_points)-1)
            legalDirs = np.argwhere(self.link[randPoint] > 0)
            randDir = np.random.randint(len(legalDirs))
            diff = np.array(self.path_points[legalDirs[randDir][0]]) - np.array(self.path_points[randPoint])
            newX, newY = self.path_points[randPoint] + np.random.rand(1) * np.array(diff)
            if itr == maxItr-1 or all(np.greater(((self.loc[0,:,0] - newX)**2 + (self.loc[1,:,0] - newY)**2), (self.minDist + self.cellDia)**2)):
                self.loc[0,nValid,0] = newX
                self.loc[1,nValid,0] = newY
                orientation[:,nValid,0] = np.sign(diff)
                task_list[nValid] = legalDirs[randDir][0]
                itr = 0
                nValid += 1
            itr += 1
            
        # Generate the rest of the states
        prevDist = np.ones(self.numCell) * np.Inf
        for n in range(1,N):
            # Update movement for robots
            temp = self.loc[:,:,n-1] + orientation[:,:,n-1] * self.speed * self.sampTime
            orientation[:,:,n] = orientation[:,:,n-1]

            # If two subnetworks are close, make sure they dont collide
            dist_pw = cdist(temp.T, temp.T)
            np.fill_diagonal(dist_pw, np.Inf)
            _,idx = np.where(dist_pw <= self.minDist)
            nIdx,_ = np.unique(idx,return_index=True)
            if nIdx.any():
                u, c = np.unique(task_list[nIdx], return_counts=True)
                dup = u[c > 1]
                if dup.any():
                    goal = np.array([self.path_points[i] for i in dup])
                    for g, nIds in zip(goal, dup):
                        dists, idSet = [],[]
                        for Id in np.intersect1d(idx, np.argwhere(task_list == nIds)):
                            dx = g[0] - temp[0,Id]
                            dy = g[1] - temp[1,Id]
                            dists.append(np.sqrt(dx**2 + dy**2))
                            idSet.append(Id)
                        dists, idSet = zip(*sorted(zip(dists, idSet))) # Sort both lists after increasing distance
                        for i in range(1,len(dists)):
                            if dists[i] - dists[i-1] <= self.minDist + self.cellDia:
                                temp[:,idSet[i]] = self.loc[:,idSet[i],n-1]

            # If any robot are close to a path point, update task list
            dx = temp[0] - np.array(self.path_points)[task_list][:,0]
            dy = temp[1] - np.array(self.path_points)[task_list][:,1]
            dist = np.sqrt(dx**2 + dy**2)
            if any(dist < self.speed * self.sampTime) or any(prevDist < dist):
                idx1 = np.where(dist < self.speed * self.sampTime)[0]
                idx2 = np.where(prevDist < dist)[0]
                reachDest = np.unique(np.concatenate((idx1, idx2)))
                prevDist = dist
                for _, d in enumerate(reachDest):
                    legalDirs = np.where(self.link[task_list[d]] > 0)
                    randDir = np.random.choice(legalDirs[0])
                    diff = np.array(self.path_points[randDir]) - np.array(self.path_points[task_list[d]])
                    orientation[:,d,n] = np.sign(diff)
                    task_list[d] = randDir
                    prevDist[d] = np.Inf
            else:
                prevDist = dist
    
            self.loc[:,:,n] = temp

    def deploy_devices(self):
        """
        Deploy an equal number of devices for each subnetwork. Locations are static
        relative to the APs.

        Return :devLoc: Device locations (np.array)
        """
        devLoc = np.zeros((2, self.numCell * self.numDev), dtype=np.float64)
        for i in range(self.numDev):
            loc_angle = np.random.uniform(low=0.0, high=1.0, size=self.numCell) * 2 * np.pi
            devLoc[:,i*self.numCell:(i+1)*self.numCell] = np.array([self.cellDia/2 * np.cos(loc_angle),
                                                                    self.cellDia/2 * np.sin(loc_angle)])
        return devLoc
    
    # =================== Channel Model Functions =================== #
    def channel_pathLoss(self, dist):
        """
        Calculate path loss of a link in factories based on 3GPP.

        Return :Gamma: Path loss (float)
        """
        PrLoS = np.exp(dist * np.log(1 - self.clutDens) / self.clutSize)
        NLoS = PrLoS <= (1 - PrLoS)
        idx = np.logical_not(np.eye(dist.shape[0]), dtype=bool)
        Gamma = np.zeros(dist.shape)
        Gamma[idx] = 31.84 + 21.5 * np.log10(dist[idx]) + 19 * np.log10(self.fc/1e9)
        if self.clutType == 'sparse':
            Gamma[NLoS] = np.max([Gamma[NLoS], 
                                33 + 25.5 * np.log10(dist[NLoS]) + 20 * np.log10(self.fc/1e9)], 
                                axis=0)
        elif self.clutType == 'dense':
            Gamma[NLoS] = np.max([Gamma[NLoS], 
                                33 + 25.5 * np.log10(dist[NLoS]) + 20 * np.log10(self.fc/1e9),
                                18.6 + 35.7 * np.log10(dist[NLoS]) + 20 * np.log10(self.fc/1e9)], 
                                axis=0)
        return Gamma#10**(Gamma/10)

    def init_shadow(self):
        """
        Generate covariance map of correlated shadowing values.

        Return :map: Correlated shadowing map (np.array)
        """
        nx, ny = [len(self.mapX), len(self.mapY)]
        # Generate covariance map
        if 'cMapFFT' not in globals(): # Calculate this part only once
            print('Calculating new shadowing map.')
            cMap = np.zeros([nx, ny], dtype=np.float64)
            for x in range(nx):
                for y in range(ny):
                    cMap[x,y]= np.exp((-1) \
                                    * np.sqrt(np.min([np.absolute(self.mapX[0]-self.mapX[x]),
                                                    np.max(self.mapX)-np.absolute(self.mapX[0]-self.mapX[x])])**2 \
                                            + np.min([np.absolute(self.mapY[0]-self.mapY[y]),
                                                    np.max(self.mapY)-np.absolute(self.mapY[0]-self.mapY[y])])**2) \
                                    / self.corrDist)

            global cMapFFT
            cMapFFT = np.fft.fft2(cMap * self.shadStd[1])
        Z = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
        map = np.real(np.fft.fft2(np.multiply(np.sqrt(cMapFFT), Z) / np.sqrt(nx * ny))) * np.sqrt(2)
        return map

    def channel_shadow(self, allLoc, dist, map):
        """
        Compute channel correlated shadowing values

        Param :allLoc: Locations of APs and devices (np.array)
        Param :dist: Distance matrix for subnetworks (np.array)
        Param :map: Correlated shadowing map (np.array)
        Return :Psi: Shadowing value matrix (np.array)
        """
        # Convert locations to the shadowing map
        idx = np.array(np.round(allLoc[0,:], 1) / self.mapRes, dtype=int)
        idy = np.array(np.round(allLoc[1,:], 1) / self.mapRes, dtype=int)
        ids = np.ravel_multi_index([idx,idy], map.shape)

        # Calculate shadowing values
        f = (map.flatten()[ids]).reshape(1,-1)
        f_AB = np.add(f.T, f)
        temp = np.exp((-1) * dist / self.corrDist)
        Psi = np.multiply(np.divide(1 - temp, np.sqrt(2) * np.sqrt(1 + temp)), f_AB)  
        return Psi#10**(Psi/10)

    def channel_fading(self, time_index=0):
        """
        Compute channel fading values with Jake's doppler model.
        Rho is the precalculated zeroth order bessel function J0(2*pi*fd*td).

        Param :time_index: The instantanious time-slot (integer)
        Return :h: Small-scale fading values (np.array)
        """
        nTot = self.numCell * (1 + self.numDev)
        if time_index == 0:
            self.h = np.sqrt(0.5 * (np.random.randn(nTot, nTot)**2 \
                                  + 1j * np.random.randn(nTot, nTot)**2))
        else:
            self.h = self.h * self.rho + np.sqrt(1. - self.rho**2) * 0.5 * (np.random.randn(nTot, nTot)**2\
                                                                           + 1j * np.random.randn(nTot, nTot)**2)
        return np.abs(self.h)

    def generate_channel(self):
        """
        Compute Rx powers for all timesteps based on channel model.
        """
        # Compute locations of all APs and devices
        devLoc = self.deploy_devices()
        dLoc = np.repeat(devLoc[:,:,np.newaxis], self.numSteps, axis=2) \
             + np.repeat(self.loc, self.numDev, axis=1)
        allLoc = np.concatenate((dLoc, self.loc), axis=1) # [m1, m2, .., mM, n1, n2, ..., nN]

        map = self.init_shadow() # Initialise correlated shadowing map
        nTot = self.numCell * (1 + self.numDev)
        self.rxPow = np.zeros([nTot, nTot, self.numSteps], dtype=np.float64)
        self.dist = np.zeros([nTot, nTot, self.numSteps], dtype=np.float64)
        for time_index in range(self.numSteps):
            # Calculate distances
            nLoc = allLoc[:,:,time_index]
            self.dist[:,:,time_index] = cdist(nLoc.T, nLoc.T)
            # Calculate general losses
            Gamma = self.channel_pathLoss(self.dist[:,:,time_index])
            Psi = self.channel_shadow(nLoc, self.dist[:,:,time_index], map)
            h = self.channel_fading(time_index)
            # Calculate Rx power
            self.rxPow[:,:,time_index] = Psi + Gamma + 20 *np.log10(h)

    # =================== Problem based step functions =================== #
    def channel_step(self, chl_action, time_index):

        #pow_action = self.powerLevels * np.ones(self.numCell)
        pow_action = self.Pmin * np.ones(self.numCell)
        rxPow = self.rxPow[:,:,time_index]

        # Interference power        
        intPow = np.zeros([self.numCell,self.numGroups],dtype=np.float64)
        for k in range(self.numGroups):
            indm = np.argwhere(chl_action == k)
            for n in range(self.numCell):
                intPow[n,k] = np.sum(10**((pow_action[n]-rxPow[n,self.numCell+indm[indm !=n]])/10)) + self.noisePower

        # SINR and channel rates
        self.SINRAll = np.zeros([self.numCell, self.numGroups], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        for n in range(self.numCell):
            for k in range(self.numGroups):
                self.SINRAll[n,k] = (10**((pow_action[n] - rxPow[n,self.numCell+n])/10)) / intPow[n,k]
            self.SINR[n] = self.SINRAll[n,chl_action[n]]
            self.sRate[n] = np.log2(1 + self.SINR[n])
            #V_nm = 1 - (1 / (1 + self.SINR[n,chl_action[n]])**2)
            #self.pRate[n] = self.sRate[n] - np.sqrt(V_nm / self.codeBlockLength) * self.qLogE
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)

        return intPow

    def channel_step_devices(self, chl_action, time_index):

        #pow_action = self.powerLevels * np.ones(self.numCell)
        pow_action = self.Pmin * np.ones(self.numCell)
        rxPow = self.rxPow[:,:,time_index]

        # Interference power        
        intPow = np.zeros([self.numCell,self.numDev,self.numGroups],dtype=np.float64)
        for k in range(self.numGroups):
            indm = np.argwhere(chl_action == k)
            _m = 0
            for n in range(self.numCell):
                for m in range(self.numDev):
                    intPow[n,m,k] = np.sum(10**((pow_action[n]-rxPow[_m,self.numCell*self.numDev+indm[indm !=n]])/10)) + self.noisePower
                    _m += 1

        # SINR and channel rates
        self.SINRAll = np.zeros([self.numCell, self.numDev, self.numGroups], dtype=np.float64)
        self.SINR = np.zeros([self.numCell, self.numDev],dtype=np.float64)
        self.sRate = np.zeros([self.numCell, self.numDev],dtype=np.float64)
        _m = 0
        for n in range(self.numCell):
            for m in range(self.numDev):
                for k in range(self.numGroups):
                    self.SINRAll[n,m,k] = (10**((pow_action[n] - rxPow[_m,self.numCell*self.numDev+n])/10)) / intPow[n,m,k]
                self.SINR[n,m] = self.SINRAll[n,m,chl_action[n]]
                self.sRate[n,m] = np.log2(1 + self.SINR[n,m])
                _m += 1
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)
        #print(self.sRate.shape)

        self.SINRAll = self.SINRAll[:,0,:]
        self.SINR = self.SINR[:,0]
        self.sRate = self.sRate[:,0]
        intPow = intPow[:,0,:]
        """# SINR and channel rates
        intPow = intPow[:,0,:]
        self.SINRAll = np.zeros([self.numCell, self.numGroups], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        for n in range(self.numCell):
            for k in range(self.numGroups):
                self.SINRAll[n,k] = (10**((pow_action[n] - rxPow[n,self.numCell+n])/10)) / intPow[n,k]
            self.SINR[n] = self.SINRAll[n,chl_action[n]]
            self.sRate[n] = np.log2(1 + self.SINR[n])
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)"""
        
        return intPow

    def joint_step(self, actions, time_index):

        chl_action = self.comb_act[actions][:,0].astype(int)
        pow_action = self.comb_act[actions][:,1]

        rxPow = self.rxPow[:,:,time_index]

        # Interference power        
        intPow = np.zeros([self.numCell,self.numGroups],dtype=np.float64)
        for k in range(self.numGroups):
            indm = np.argwhere(chl_action == k)
            for n in range(self.numCell):
                intPow[n,k] = np.sum(10**((pow_action[n]-rxPow[n,self.numCell+indm[indm !=n]])/10)) + self.noisePower

        # SINR and channel rates
        self.SINRAll = np.zeros([self.numCell, self.numGroups*self.numLevels], dtype=np.float64)
        self.SINR = np.zeros([self.numCell],dtype=np.float64)
        self.sRate = np.zeros([self.numCell],dtype=np.float64)
        
        for n in range(self.numCell):
            i = 0
            for k in range(self.numGroups):
                for u in range(self.numLevels):
                    self.SINRAll[n,i] = (10**((self.Plevels[u] - rxPow[n,self.numCell+n])/10)) / intPow[n,k]
                    i += 1
            self.SINR[n] = self.SINRAll[n,actions[n]]
            self.sRate[n] = np.log2(1 + self.SINR[n])
        self.SINRAll = 10 * np.log10(self.SINRAll)
        self.SINR = 10 * np.log10(self.SINR)

        return intPow

    # =================== Environment action =================== #
    #def step(self, time_index=0, chl_action=None, pow_action=None):
    def step(self, actions, time_index=0):
        """
        Perform a step in the environment, for a given time step and action

        Param :time_index: Integer time index (np.array)
        Param :chl_action: Integer channel actions (np.array)
        Param :pow_action: Float power levels (np.array)
        Return :obs: Observations, type defined with observation_type (np.array)
        Return :reward: Rewards, type defined with reward_type (np.array)
        Return :done: True if episode is done (list)
        Return :info: Additional information (dictionary)
        """
        # Decode input action
        if self.problem == 'channel':
            if self.numDev <= 1: # Remove this IF statement after development!
                intPow = self.channel_step(actions, time_index)
            else:
                intPow = self.channel_step_devices(actions, time_index)
        elif self.problem == 'joint':
            intPow = self.joint_step(actions, time_index)
        else:
            raise NotImplementedError

        # Observation
        if self.observation_type == 'I':
            obs = 10.0*np.log10(intPow)
        elif self.observation_type == 'I_minmax':
            obs = (10.0*np.log10(intPow) - 10.0*np.log10(self.noisePower)) / self.intPowNorm
        elif self.observation_type == 'sinr':
            obs = self.SINRAll
        elif self.observation_type == 'sinr_minmax':
            obs = (self.SINRAll - self.sinrMin) / self.sinrNorm
        else:
            raise NotImplementedError
        
        #print(self.SINR)

        # Reward signal
        if self.reward_type == 'rate':
            reward = self.sRate
        elif self.reward_type == 'sinr':
            reward = self.SINR
        elif self.reward_type == 'binary':
            alpha = 10
            idx = self.SINR <= self.SINR_req[:,0]
            reward = np.full(self.numCell, alpha, dtype=float)
            reward[idx] = (-1) * alpha
        elif self.reward_type == 'composite_reward':
            rate_sum = np.sum(self.sRate, axis=1)
            idx = rate_sum <= self.r_min
            reward = self.lambda_1 * rate_sum
            reward[idx] -= self.lambda_2 * (rate_sum[idx] - self.r_min)
        #elif self.reward_type == 'constrained':
        #    self.w_next = np.max([self.w_next + self.reqSINR - self.SINR, np.zeros(self.w_next.shape)], axis=0)
        #    self.p_next = np.max([self.p_next + self.numDev * 10**((pow_action)/10) - self.Pmax, np.zeros(self.p_next.shape)], axis=0)
        #    reward = np.sum(self.pRate, axis=1) - np.sum(self.w_next, axis=1) - self.p_next
        else:
            raise NotImplementedError

        # Supplementary outputs
        done = [time_index >= (self.numSteps-1) for _ in range(self.numCell)] # (Return True if episode is done)
        info = {} # No info

        return obs, reward, done, info

# =================== Used for analysis of the environment =================== #
def centralizedColoring(interMat, numGroups): 
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
        for k in range(1, numGroups):
            G.add_edge(n, Indx[k]) 
    d = nx.coloring.greedy_color(G, strategy='connected_sequential_bfs', interchange=True)
    act = np.asarray(list(d.values()),dtype=int)
    idx = np.argwhere(act >= numGroups).flatten()
    act[idx] = np.random.choice(np.arange(numGroups), len(idx))
    return act

if __name__ == '__main__':

    from util import *

    N=20
    M=2
    channels=4
    seed=123
    reward = 'composite_reward'

    env = env_subnetwork(numCell=N, numDev=M, group=channels, level=1, fname='Factory_values.mat', dt=0.005, steps=201, reward_type=reward, clutter='sparse', observation_type='I', seed=seed)
    env.generate_mobility()
    env.generate_channel()
    act = np.random.choice(channels, N)
    print(act)
    obs, rew, _, _ = env.step(actions=act, time_index=0)

    # CHECK DEVICE AND ACCESS POINT DEPLOYMENT
    #N, M, seed = 10, 10, 123
    #check_devLoc(N=N, M=M, seed=seed, timestep=0)
    #check_devLoc(N=N, M=M, seed=seed, timestep=np.random.randint(200))
    #check_devLoc(N=N, M=M, seed=seed, timestep=-1)

    """    
    #num_steps_tot = 10000
    #plot_environment_metrics(['sparse'], [20], N_plot_CDF=[True], num_steps_tot=num_steps_tot, max_delay=5, save_fig=False, fpath='../../../../results/', dt=0.01,steps=10)
    #plot_environment_metrics(['sparse'], [20], N_plot_CDF=[True], num_steps_tot=num_steps_tot, max_delay=10, save_fig=False, fpath='../../../../results/', dt=0.005,steps=200)
    #plot_environment_metrics(['sparse'], [20], N_plot_CDF=[True], num_steps_tot=num_steps_tot, max_delay=16, save_fig=False, fpath='../../../../results/', dt=0.008,steps=124)
    if False:
        max_delays = [1, 2, 10]#[n+1 for n in range(50)][::2] # 1, 3, 5, ..., 49 
        clutter_list = ['sparse', 'dense']
        N_list = [10, 20, 25, 50]
        N_plot_CDF = [True, True, False, True]#[True for _ in N_list]
        num_steps_tot = 1000000
        save_figs = True

        max_delay = plot_test_delay(max_delays, N=20, num_steps_tot=num_steps_tot, save_fig=save_figs)
        print(f'\nMax delay = {max_delay}')
        plot_environment_metrics(clutter_list, N_list, N_plot_CDF=N_plot_CDF, num_steps_tot=num_steps_tot, max_delay=10, save_fig=save_figs, steps=200, dt=0.005)
    """