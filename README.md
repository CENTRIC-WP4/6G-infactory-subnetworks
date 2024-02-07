# 6G in-Factory Subnetworks

## Overview
This repository contains code for interference management in 6G in-X subnetworks, with a focus on machine learning techniques. The deployment scenario considered is uncoordinated autonomous robots inside a factory. The in-factory scenario comprises multiple mobile robots on a factory floor inspired by those defined by 5G-ACIA, as illustrated in the figure:

![image](https://github.com/CENTRIC-WP4/6G-infactory-subnetworks/blob/main/in-factory.gif)

## Expected outputs

For now the results contains the performance (average rate and average execution time) for different episodes.


## Getting started

The code has been tested to work on Python 3.7 under Windows 10.

1. Clone the repository:
    ```bash
    git clone https://github.com/CENTRIC-WP4/6G-infactory-subnetworks
    ```

2. Install the required packages using `pip3`:
   ```bash
   cd 6G-infactory-subnetworks
   pip3 install -r requirements.txt
   ```

3. Run the code:
    ```bash
    cd code
    python main.py
   ```
## Repository

- `code/main.py` Main file to run. Here important parameters for documentation can be defined for multiple runs, when running on cloud servers.
- `code/config.py` Simulation configuration. Any values saved here will be fetched upon execution. The configuration will also save as a pickle file in the run folder.
- `results/` Contains saved log files and any other files generated for evaluation (This folder is added to gitignore, the current results are from a project).
 
When running a simulation through the main file, consider `alg` (algorithm: maddqn, maddqn_dec, maddqn_fed), `reward` (reward: rate, sinr, binary, composite_reward), `prob` (problem: channel, joint), `observation` (observation: I, I_minmax, sinr, sinr_minmax).

Create a new run with the function: `config, _, _ = initialise_trainer(NAME, <input simulation parameters here>)`. An example is present in the main file. The same function can also load an old run, if a run was found under `NAME`: `config, _, _ = initialise_trainer(NAME, alg)`. If `NAME=None`, a name will be generated. `config` contains all parameters about the run, and will be saved in the result folders. An algorithm runner object can be created with `make_runner`. Models and logs are automatically saved in `results/subnetworks/factory/<alg>/sparse/<NAME>`.


## Baselines

The implemented interference management techniques are either heuristics or reinforcement learning based. 

The considered benchmarks include: 
  1. Fixed channel assignment: channel are randomly assigned to each subnetwork at initialization without the possiblity for dynamic updates. 
  2. Greedy: each subnetwork selects the least interfered channel using its own sensing information
  3. Centralized Graph Coloring: select the best channel based on a global view on the system

The considered algorithms include:
  1. Multi-Agent Double Deep Q-Networks (MAPPO)
  2. Multi-Agent Proximal Policy Optimization (MADDQN)
  3. Multi-Agent Transformer (MAT - NOT FULLY IMPLEMENTED!)

Training frameworks include:
  1. Centralized training with distributed execution
  2. Distributed training and execution
  3. Federated training and distributed execution

Models and logs are automatically saved in results/subnetworks/...

## Usage

This code can be used to implement a 6G in-factory subnetwork scenario for protocol learning and other multi-agent reinforcement learning tasks.


## How to contribute

There are two main ways to contribute to this repository:

1. **Implementing New Problems**: Different algorithms for protocol learning can be implemented on top of the current subnetwork scenario.
2. **Evaluation Benchmarks with Different Deployment Configurations**

## References

1. Adeogun, Ramoni, Gilberto Berardinelli, and Preben E. Mogensen. "Enhanced interference management for 6G in-X subnetworks." IEEE Access 10 (2022): 45784-45798.
2. Adeogun, Ramoni, Gilberto Berardinelli, and Preben Mogensen. "Learning to dynamically allocate radio resources in mobile 6G in-x subnetworks." 2021 IEEE 32nd Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC). IEEE, 2021.



## License

This project is licensed under the MIR license - see the [LICENSE](https://github.com/CENTRIC-WP4/Multiple-access-with-MuJoCo-robots/blob/main/LICENSE).
