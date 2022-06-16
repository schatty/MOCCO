# Guided Exploration in Reinforcement Learning via Monte Carlo Critic Optimization

<img align="right" width="280" src="https://user-images.githubusercontent.com/23639048/174118217-84722ed8-3fa1-4159-b9b4-4b0913841cbc.png">

The source code for the paper "Guided Exploration in Reinforcement Learning via Monte Carlo Critic Optimization" (ArXiv link TBA) accepted at [ICML 2022, DARL Workshop](https://darl-workshop.github.io).

__Idea__
* Replace the random normal noise from the deterministic algorithm with guided exploration
* Train an ensemble of action-conditioned Monte-Carlo Critics as a differentiable controller giving the direction towards the most unexplored environmental regions
* Use the provided action direction from a controller as an auxiliary action component to guide a policy during exploration.
* __MOCCO__ algorithm is the combination of the aforementioned guided exploration + using a mean of MC Critics ensemble as a second critic estimate for alleviating Q-value overestimation.

## Usage

The current version of the code works with [wandb](https://wandb.ai) as a logger tool.

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

![mocco_eval](https://user-images.githubusercontent.com/23639048/174117437-adbfa41a-606c-4e85-95c8-d39c41766920.png)

The paper curves are available at `data/curves` directory.
