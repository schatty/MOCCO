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


class MCCritic(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.q1 = MLP(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

        self.q2 = MLP(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

        self.q3 = MLP(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x), self.q2(x), self.q3(x)

    def Q1(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x)

    def get_action_grad(self, optim, states, actions): #, actions):
        q1, q2, q3 = self.forward(states, actions)
        q_cat = torch.cat((q1, q2, q3), dim=1).flatten()
        var = torch.var(q_cat)

        optim.zero_grad()
        var.backward(retain_graph=True)
        da = torch.autograd.grad(var, actions)

        return da[0]


class GEMBO:

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

        self.critic_mc = MCCritic(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)

        self.critic_target = deepcopy(self.critic).to(self.device).eval()
        disable_gradient(self.critic_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_critic_mc = Adam(self.critic_mc.parameters(), lr=lr_critic)

        self.target_update_coef = target_update_coef

        noise_std = 0.58
        self.da_std_buf = np.zeros((10000, *action_shape))
        self.norm_noise = np.sqrt(action_shape[0]) * noise_std
        self.da_std_cnt = 0
        self.da_std_max = np.zeros(*action_shape)
        self.da_std_size = 0

    def get_guided_noise(self, state, a_pi=None, with_info=False):
        #print("state: ", state.shape, state)
        if a_pi is None:
            a_pi = self.actor(state)  # [1, ACTION_DIM]
        #print("a_pi: ", a_pi.shape)
        #print("a_pi: ", a_pi.shape, a_pi)
        
        d_a = self.critic_mc.get_action_grad(self.optim_critic_mc, state, a_pi)  # [1, ACTION_DIM]
        #print("da: ", d_a, d_a.shape)
        da_std_len = min(self.da_std_size, self.da_std_buf.shape[0])
        da_std = self.da_std_buf[:da_std_len, :].std(axis=0)
        #print("da_std: ", da_std, da_std.shape)
        scale = torch.tensor(da_std / self.da_std_max).float().to(self.device)
        #print("scale: ", scale)
        #print("scale: ", type(scale), scale.shape)

        #print("da: ", d_a.shape)
        d_a_norm = torch.linalg.norm(d_a, dim=1, keepdim=True)
        #print("d_a_norm: ", d_a_norm.shape)
        d_a_normalized = d_a / d_a_norm * self.norm_noise
        #print("d_a_normalized: ", d_a_normalized.shape)
        noise = d_a_normalized * scale  # [1, 6]

        if with_info:
            return noise, da_std, scale
        return noise

    def explore(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)

        a_pi = self.actor(state)
        noise, da_std, scale = self.get_guided_noise(state, a_pi=a_pi, with_info=True)
        noise = noise.cpu()
 
        self.accumulate_action_gradient(state)

        # Logging
        if self.update_step % self.log_every == 0:
            for i_a in range(noise.shape[1]):
                self.wandb.log({f"guided_noise/noise_a{i_a}": noise[0, i_a].item(), "update_step": self.update_step})
                self.wandb.log({f"guided_noise_da/da_run_std_{i_a}": da_std[i_a].item(), "update_step": self.update_step})
                self.wandb.log({f"guided_noise_scale/scale_a{i_a}": scale[i_a].item(), "update_step": self.update_step})

        return (a_pi.detach().cpu() + noise).numpy()[0]

    def update(self, batch, batch_mc):
        self.update_step += 1

        self.update_critic_mc(*batch_mc)
        self.update_critic(*batch)
        self.update_actor(batch[0])

        soft_update(self.critic_target, self.critic, self.target_update_coef)
        soft_update(self.actor_target, self.actor, self.target_update_coef)

    def update_critic(self, states, actions, rewards, dones, next_states):
        q1 = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
        noise = self.get_guided_noise(next_states).detach()
        #print("noise: ", noise.shape)
        next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
        #print("Success!")
        #_ = input('stop update_critic')
        q_next = self.critic_target(next_states, next_actions)

        q_target = rewards + (1.0 - dones) * self.discount * q_next
        q_mc_1, q_mc_2, q_mc_3 = self.critic_mc(states, actions)
        q_mc_cat = torch.cat((q_mc_1, q_mc_2, q_mc_3), dim=1)
        q_mc = torch.mean(q_mc_cat, dim=1, keepdim=True).detach()

        td_loss = (q1 - q_target).pow(2).mean()
        mc_loss = (q1 - q_mc).pow(2).mean()
        loss_critic = td_loss + mc_loss

        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

        if self.update_step % self.log_every == 0:
            self.wandb.log({"algo/q1": q1.detach().mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/q_target": q_target.mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/q_mc": q_mc.mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/abs_q_err": (q1 - q_target).detach().mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/critic_loss_td": td_loss.item(), "update_step": self.update_step})
            self.wandb.log({"algo/critic_loss_mc": mc_loss.item(), "update_step": self.update_step})
            self.wandb.log({"algo/critic_loss_total": loss_critic.item(), "update_step": self.update_step})
            self.wandb.log({"algo/q1_grad_norm": self.critic.q1.get_layer_norm(), "update_step": self.update_step})
            self.wandb.log({"algo/actor_grad_norm": self.actor.mlp.get_layer_norm(), "update_step": self.update_step})

            for i_a in range(actions.shape[1]):
                self.wandb.log({f"guided_noise_critic/noise_a{i_a}": noise[0, i_a].item(), "update_step": self.update_step})

    def update_critic_mc(self, states, actions, qs_mc):
        q1, q2, q3 = self.critic_mc(states, actions)
        loss_mc = (q1 - qs_mc).pow(2).mean() + (q2 - qs_mc).pow(2).mean() + (q3 - qs_mc).pow(2).mean()
        self.optim_critic_mc.zero_grad()
        loss_mc.backward()
        self.optim_critic_mc.step()

        if self.update_step % self.log_every == 0:
            self.wandb.log({"algo/mc_loss": loss_mc.item(), "update_step": self.update_step})
 
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

    def accumulate_action_gradient(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        a_pi = self.actor(state)
        d_a = self.critic_mc.get_action_grad(self.optim_critic_mc, state, a_pi).detach()
        self.da_std_buf[self.da_std_cnt, :] = d_a.cpu().numpy().flatten()
        self.da_std_cnt = (self.da_std_cnt + 1) % self.da_std_buf.shape[0]
        self.da_std_size += 1

        da_std_len = min(self.da_std_size, self.da_std_buf.shape[0])
        da_std = self.da_std_buf[:da_std_len, :].std(axis=0)
        #print("Max prev: ", self.da_std_max, self.da_std_max.shape)
        self.da_std_max = np.maximum(self.da_std_max, da_std)

