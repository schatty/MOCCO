import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from .nn import DoubleCritic, MLP
from .utils import initialize_weight


class Dynamics(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(400, 400), hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.mlp = MLP(
            input_dim=state_shape[0] + action_shape[0] + 1,
            output_dim=state_shape[0] + 2, # + reward + return
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        ).apply(initialize_weight)

    def forward(self, states, actions, t):
        x = torch.cat([states, actions, t], dim=-1)
        return self.mlp(x)


class ModelDynamics:

    def __init__(self, state_shape, action_shape, device, log_every=5000, seed=0, wandb=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.update_step = 0
        self.log_every = log_every

        self.dynamics = Dynamics(state_shape, action_shape).to(device)
        self.optim_dynamics = Adam(self.dynamics.parameters(), lr=3e-4)

        assert wandb is not None, "wandb as a named argument is required"
        self.wandb = wandb

    def update(self, states, actions, rewards, next_states, returns, t):
        model_pred = self.dynamics(states, actions, t)
        next_states_pred, reward_pred, return_pred = model_pred[:, :-2], model_pred[:, -2].unsqueeze(1), model_pred[:, -1].unsqueeze(1)

        alpha = 0.01
        beta = 1.0
        gamma = 1e-3
        loss_states = (next_states - next_states_pred).pow(2).mean()
        loss_reward = (rewards - reward_pred).pow(2).mean()
        loss_return = (returns - return_pred).pow(2).mean()
        loss = alpha * loss_states + beta * loss_reward + gamma * loss_return

        self.optim_dynamics.zero_grad()
        loss.backward()
        self.optim_dynamics.step()

        if self.update_step % self.log_every == 0:
            self.wandb.log({"dynamics/loss_state": alpha * loss_states.item(), "update_step": self.update_step})
            self.wandb.log({"dynamics/loss_reward": beta * loss_reward.item(), "update_step": self.update_step})
            self.wandb.log({"dynamics/loss_return": gamma * loss_return.item(), "update_step": self.update_step})
            self.wandb.log({"dynamics/loss": loss.item(), "update_step": self.update_step})
            self.wandb.log({"dynamics/eval": loss_states.item(), "update_step": self.update_step})
            self.wandb.log({"dynamics/states_mean": states.mean().item(), "update_step": self.update_step})
            self.wandb.log({"dynamics/reward_mean": rewards.mean().item(), "update_step": self.update_step})

        self.update_step += 1