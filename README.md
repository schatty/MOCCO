# Guided Exploration in Reinforcement Learning via Monte Carlo Critic Optimization

The source code for the paper Guided Exploration in Reinforcement Learning via Monte Carla Critic Optimization [Arxiv](https://arxiv.org/abs/2206.12674), presented at
* [AAMAS 2024](https://www.aamas2024-conference.auckland.ac.nz/accepted/extended-abstracts/)
* [ICML 2022, DARL Workshop](https://darl-workshop.github.io)

## Requirements

The experiments were run with `python3.10` and `mujoco 2.3.7`, install full env via 

```
pip install -r requirements.txt
```

## Usage

__Run single training__

```
python train.py --env point_mass-easy --algo MOCCO --device cuda:0 --seed 0
```

__Run on many seeds__

For running an algorithm on many many seeds to reproduce paper results, specify needed algorithm as a script file (in `/scripts` folder) and set needed env as a first argument:

```
bash scripts/mocco.sh point_mass-easy cuda:0
```

## Results

![mocco_eval-2](https://github.com/schatty/MOCCO/assets/23639048/bbcf07c2-7b8e-4b34-8940-c81100d41d28)
