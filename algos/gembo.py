import torch
import numpy as np
from copy import copy, deepcopy

import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from .nn import DynamicsCritic, MLP
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


class GEMBO:

    def __init__(self, state_shape, action_shape, device, seed, batch_size=256, policy_noise=0.2,
                 expl_noise=0.1, noise_clip=0.5, policy_freq=2, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4,
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
        self.policy_noise = policy_noise
        self.expl_noise = expl_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
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

        self.critic = DynamicsCritic(
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

        self.norm_noise = np.sqrt(action_shape[0]) * expl_noise
        self.da_std_buf = []

    def explore(self, state, noise=None):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            if noise is None:
                noise = (torch.randn(self.action_shape) * self.max_action * self.expl_noise).to(self.device)
                noise.unsqueeze_(0)
            action = self.actor(state) + noise

        # Log the noise
        if self.update_step % self.log_every == 0:
            for i_noise in range(noise.shape[1]):
                self.wandb.log({f"noise/a_{i_noise}": noise[0, i_noise], "update_step": self.update_step})
            
        return action.cpu().numpy()[0]

    def update(self, states, actions, rewards, dones, next_states):
        self.update_step += 1
        self.update_critic(states, actions, rewards, dones, next_states)

        #if self.update_step % self.policy_freq == 0:
        if self.update_step % 1 == 0:
            self.update_actor(states)
            soft_update(self.critic_target, self.critic, self.target_update_coef)
            soft_update(self.actor_target, self.actor, self.target_update_coef)

    def update_critic(self, states, actions, rewards, dones, next_states):
        q, s1_delta, s2_delta, s3_delta = self.critic(states, actions)
        #print("q s1_delta s2_delta s3_delta: ", q.shape, s1_delta.shape, s2_delta.shape, s3_delta.shape)
        #print("q: ", q)
        #print("s1_delta: ", s1_delta)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-self.max_action, self.max_action)

            _, s1_delta_tgt, s2_delta_tgt, s3_delta_tgt = self.critic(states, actions)

            # next_actions_1 = self.actor(states + s1_delta_tgt) + noise
            #next_actions_1 = next_actions_1.clamp(-self.max_action, self.max_action)
            q1_next, _, _, _ = self.critic_target(states + s1_delta_tgt, next_actions)

            
            #next_actions_2 = self.actor(states + s2_delta_tgt) + noise
            #next_actions_2 = next_actions_2.clamp(-self.max_action, self.max_action)
            q2_next, _, _, _ = self.critic_target(states + s2_delta_tgt, next_actions)


            #next_actions_3 = self.actor(states + s3_delta_tgt) + noise
            #next_actions_3 = next_actions_3.clamp(-self.max_action, self.max_action)
            q3_next, _, _, _ = self.critic_target(states + s3_delta_tgt, next_actions)

            q_next = torch.min(torch.min(q1_next, q2_next), q3_next)
            #print("q_next: ", q_next.shape)

        q_target = rewards + (1.0 - dones) * self.discount * q_next
        td_error = (q - q_target).pow(2).mean()


        s_delta_ = next_states - states
        s0_delta_loss = (s_delta_ - s1_delta).pow(2).mean()
        s1_delta_loss = (s_delta_ - s2_delta).pow(2).mean()
        s2_delta_loss = (s_delta_ - s3_delta).pow(2).mean()

        loss_critic = td_error + s0_delta_loss + s1_delta_loss + s2_delta_loss

        self.optim_critic.zero_grad()
        loss_critic.backward()
        #(s0_delta_loss + s1_delta_loss + s2_delta_loss).backward()
        self.optim_critic.step()
 
        #self.optim_critic.zero_grad()
        #loss_critic.backward()
        #(td_error).backward()
        #self.optim_critic.step()

        if self.update_step % self.log_every == 0:
            self.wandb.log({"algo/q": q.detach().mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/q_target": q_target.mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/abs_q_err": (q - q_target).detach().mean().cpu(), "update_step": self.update_step})
            self.wandb.log({"algo/critic_loss": loss_critic.item(), "update_step": self.update_step})
            self.wandb.log({"algo/td_error: ": td_error.item(), "update_step": self.update_step})
            self.wandb.log({"algo/s0_delta_loss": s0_delta_loss.item(), "update_step": self.update_step})
            self.wandb.log({"algo/s_delta_sum": s0_delta_loss + s1_delta_loss + s2_delta_loss, "update_step": self.update_step})
            self.wandb.log({"algo/q1_grad_norm": self.critic.q.get_layer_norm(), "update_step": self.update_step})
            self.wandb.log({"algo/actor_grad_norm": self.actor.mlp.get_layer_norm(), "update_step": self.update_step})

            # Off-policy noise
            #for i_noise in range(min(2, noise.shape[1])):
            #    self.wandb.log({f"algo/noise_critic_{i_noise}": noise[0, i_noise].item(),
            #                    "update_step": self.update_step})
 
    def update_actor(self, states):
        actions = self.actor(states)
        qs, _, _, _ = self.critic(states, actions)
        loss_actor = -qs.mean()

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

    def get_guided_noise(self, state, next_state, reward, model_dynamics):
        a_pi = self.actor(state)
        d_a = model_dynamics.get_action_grad(state, a_pi, next_state, reward).detach()
        d_a_norm = torch.linalg.norm(d_a)
        noise = d_a / d_a_norm * self.norm_noise

        self.da_std_buf.append(d_a.cpu().numpy().flatten())
        if len(self.da_std_buf) > 1000:
            self.da_std_buf.pop(0)

        # Logging
        if self.update_step % 100 == 0:
            #print("shape of numpy std buf: ", np.array(self.da_std_buf).shape)
            for i in range(d_a.shape[1]):
                self.wandb.log({f"noise/d_a_{i}_magnitude": d_a[:, i].abs().mean(), "update_step": self.update_step})
                self.wandb.log({f"noise/d_a_{i}_std": np.array(self.da_std_buf)[:, i].std(),
                               "update_step": self.update_step})
                self.wandb.log({f"noise/d_a{i}_mag_smoothed": np.abs(np.array(self.da_std_buf)[:, i]).mean()})

        return noise
