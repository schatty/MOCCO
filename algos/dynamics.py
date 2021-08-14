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
            input_dim=state_shape[0] + action_shape[0],
            output_dim=state_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        ).apply(initialize_weight)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
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

    def update(self, states, actions, next_states):
        next_states_pred = self.dynamics(states, actions)

        loss = (next_states - next_states_pred).pow(2).mean()

        self.optim_dynamics.zero_grad()
        loss.backward()
        self.optim_dynamics.step()

        if self.update_step % self.log_every == 0:
            self.wandb.log({"dynamics/loss": loss.item(), "update_step": self.update_step})
            self.wandb.log({"dynamics/eval": loss.item(), "update_step": self.update_step})

        self.update_step += 1

    def get_action_grad(self, states, actions, next_states):
        next_states_pred = self.dynamics(states, actions)

        loss = (next_states - next_states_pred).pow(2).mean()

        self.optim_dynamics.zero_grad()
        loss.backward(retain_graph=True)

        dx = torch.autograd.grad(loss, actions)[0]

        return dx
