from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


from .nn import Critic, MLP
from .utils import Clamp, initialize_weight, soft_update, disable_gradient


class DeterministicPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.mlp = MLP(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        ).apply(initialize_weight)

    def forward(self, states):
        return torch.tanh(self.mlp(states))


class DDPG:

    def __init__(self, state_shape, action_shape, device, seed, batch_size=256,
                 expl_noise=0.1, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4,
                 max_action=1.0, target_update_coef=5e-3, log_every=5000, wandb=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.update_step = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dtype = torch.uint8 if len(state_shape) == 3 else torch.float
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.expl_noise = expl_noise
        self.max_action = max_action
        self.discount = gamma
        self.log_every = log_every

        assert wandb is not None, "wandb as a named argument is required"
        self.wandb = wandb

        self.actor = DeterministicPolicy(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)

        self.actor_target = deepcopy(self.actor).to(self.device).eval()

        self.critic = Critic(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)

        self.critic_target = deepcopy(self.critic).to(self.device).eval()
        disable_gradient(self.critic_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.target_update_coef = target_update_coef

    def explore(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            noise = (torch.randn(self.action_shape) * self.max_action * self.expl_noise).to(self.device)
            action = self.actor(state) + noise
        return action.cpu().numpy()[0]

    def update(self, states, actions, rewards, dones, next_states):
        self.update_step += 1
        self.update_critic(states, actions, rewards, dones, next_states)

        self.update_actor(states)
        soft_update(self.critic_target, self.critic, self.target_update_coef)
        soft_update(self.actor_target, self.actor, self.target_update_coef)

    def update_critic(self, states, actions, rewards, dones, next_states):
        q1 = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_actions = next_actions.clamp(-self.max_action, self.max_action)
            q_next = self.critic_target(next_states, next_actions)

        q_target = rewards + (1.0 - dones) * self.discount * q_next
        loss_critic = (q1 - q_target).pow(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

        if self.update_step % self.log_every == 0:
            self.wandb.log({"algo/q1": q1.detach().mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/q_target": q_target.mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/abs_q_err": (q1 - q_target).detach().mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/critic_loss": loss_critic.item(), "update_step": self.update_step})
            self.wandb.log({"algo/q1_grad_norm": self.critic.q1.get_layer_norm(), "update_step": self.update_step})
            self.wandb.log({"algo/actor_grad_norm": self.actor.mlp.get_layer_norm(), "update_step": self.update_step})
            
    def update_actor(self, states):
        actions = self.actor(states)
        qs1 = self.critic.Q1(states, actions)
        loss_actor = -qs1.mean()

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        if self.update_step % self.log_every == 0:
            self.wandb.log({"algo/loss_actor": loss_actor.item(), "update_step": self.update_step})

    def exploit(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]
