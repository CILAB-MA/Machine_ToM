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
### Inference

```python
python -m experiment.experiment2 --num_exp 2
```
### Train
```python
python -m experiment.experiment2 --num_exp 2
```
### Check the environment, agent, etc
- Environment
```python
python -m experiment.experiment2 --num_exp 2
```
- Agent
```python
python -m experiment.experiment2 --num_exp 2
```
- Storage
```python
python -m experiment.experiment2 --num_exp 2
```
## Experiment Description
- Experiment 1 : 