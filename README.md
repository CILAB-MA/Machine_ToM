[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Machine_ToM
The Implementation of "Machine Theory of Mind", ICML 2018, You can read the paper http://proceedings.mlr.press/v80/rabinowitz18a/rabinowitz18a.pdf
## Contents
- [1. Install](#install) 
- [2. Code Structure](#structure) 
- [3. Run the Code](#run-the-code) 
- [4. Experiment Description](#experiment-description) 

## Install
- will be update soon 
## Structure
```bash
└─Machine_ToM
    ├─agent : Agent directory used in Experiments
    ├─environment : Environment
    ├─experiment :  Experiment files
    ├─model : Machine ToM models
    └─utils : dataloader, storage and visualization
```
## Run the code

```python
python main.py --num_exp 2 --sub_exp 1 --num_epoch 1000
```
### Check the environment, agent, etc
- Environment
```python
python environment/env.py
```
- Agent
```python
python agent/reward_seeking_agent.py
```

## Experiment Description
- Experiment 1: In this experiment, we predict the future action of current state with random agents whose policies are depending on Dirichlet dist. You can adjust the number of past trajectory by `--sub_exp`.
- Experiment 2: 
In this experiment, we predict the future action, consumption, successor representation of value iteration agents. The number of walls is sampled between 0 and 4.

There are three sub experiments. 
  - first sub experiment : MToM with the full trajectory of an agent on single past MDP. Agent gets a panalty(-0.01) for every move.
  - second sub experiment : MToM with partial trajectory(one step) of an agent on single past MDP. Agent gets panalty(-0.01) for every move.
  - third sub experiment :  same as first sub experiment. But agent gets high panalty(-0.05) for every move.
