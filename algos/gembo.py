import torch
import numpy as np

from .td3 import TD3


class GEMBO(TD3):
    def __init__(self, state_shape, action_shape, device, seed, batch_size=256, policy_noise=0.2,
                 expl_noise=0.1, noise_clip=0.5, policy_freq=2, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4,
                 max_action=1.0, target_update_coef=5e-3, log_every=5000, wandb=None):
        super().__init__(state_shape=state_shape, action_shape=action_shape, device=device, seed=seed,
                         batch_size=batch_size, policy_noise=policy_noise, expl_noise=expl_noise,
                         noise_clip=noise_clip, policy_freq=policy_freq, gamma=gamma, lr_actor=lr_actor,
                         lr_critic=lr_critic, max_action=max_action, target_update_coef=target_update_coef,
                         log_every=log_every, wandb=wandb)

        self.norm_noise = np.sqrt(action_shape[0]) * expl_noise

    def get_guided_noise(self, state, next_state, reward, model_dynamics):
        a_pi = self.actor(state)
        d_a = model_dynamics.get_action_grad(state, a_pi, next_state, reward).detach()
        d_a_norm = torch.linalg.norm(d_a)
        noise = d_a / d_a_norm * self.norm_noise
        return noise