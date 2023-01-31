from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


from .nn import Critic, MLP, MCCritic
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
    def __init__(self, state_shape, action_shape, max_action=1, discount=0.99, tau=5e-3,
                 batch_size=256, device="cuda:0", seed=0, wandb=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.actor = DeterministicPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_shape, action_shape).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2, lr=3e-4)

        assert wandb is not None, "wandb as a named argument is required"
        self.wandb = wandb

        self.expl_noise = 0.1
        self.action_shape = action_shape
        self.dtype = torch.float
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.device = device

    def exploit(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def explore(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)

        with torch.no_grad():
            noise = (torch.randn(self.action_shape) * self.max_action * self.expl_noise).to(self.device)
            action = self.actor(state) + noise

        a = action.cpu().numpy()[0]
        return np.clip(a, -self.max_action, self.max_action)

    def update(self, batch):
        # Sample replay buffer 
        #state, action, next_state, reward, done = batch
        state, action, reward, done, next_state = batch

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1.0 - done) * self.discount * target_Q.detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def accumulate_action_gradient(self, *args):
        pass
