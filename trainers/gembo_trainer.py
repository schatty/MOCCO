import os

import numpy as np
import wandb

from .buffers.episodic_replay_buffer import EpisodicReplayBuffer
from .buffers.mc_episodic_replay_buffer import MCEpisodicReplayBuffer
from .mf_trainer import ModelFreeTrainer


class GemboTrainer(ModelFreeTrainer):

    def __init__(self, state_shape=None, action_shape=None, env=None, env_test=None, algo=None, buffer_size=int(3e6),
                 gamma=0.99, device=None, num_steps=int(1e6), start_steps=int(1e3), batch_size=128,
                 eval_interval=int(2e3), num_eval_episodes=10, save_buffer_every=0, visualize_every=0,
                 estimate_q_every=0, stdout_log_every=int(1e5), seed=0, log_dir=None, wandb=None):
        """
        Args:
            state_shape: Shape of the state.
            action_shape: Shape of the action.
            env: Enviornment object.
            env_test: Environment object for evaluation.
            algo: Codename for the algo (SAC).
            buffer_size: Buffer size in transitions.
            gamma: Discount factor.
            device: Name of the device.
            num_step: Number of env steps to train.
            start_steps: Number of environment steps not to perform training at the beginning.
            batch_size: Batch-size.
            eval_interval: Number of env step after which perform evaluation.
            save_buffer_every: Number of env steps after which save replay buffer.
            visualize_every: Number of env steps after which perform vizualization.
            stdout_log_every: Number of evn steps after which log info to stdout.
            seed: Random seed.
            log_dir: Path to the log directory.
            wandb: W&B logger instance.
        """
        super().__init__(state_shape=state_shape, action_shape=action_shape, env=env, env_test=env_test,
                         algo=algo, buffer_size=buffer_size, gamma=gamma, device=device, num_steps=num_steps,
                         start_steps=start_steps, batch_size=batch_size, eval_interval=eval_interval,
                         num_eval_episodes=num_eval_episodes, save_buffer_every=save_buffer_every,
                         visualize_every=visualize_every, estimate_q_every=estimate_q_every, seed=seed,
                         log_dir=log_dir, wandb=wandb)

        self.buffer_mc = MCEpisodicReplayBuffer(
            buffer_size=10000,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            gamma=gamma
        )

    def train(self):
        ep_step = 0
        mean_reward = 0
        state = self.env.reset()

        for env_step in range(self.num_steps + 1):
            ep_step += 1
            if env_step <= 1000:# self.start_steps:
                action = self.env.action_space.sample()
                self.algo.accumulate_action_gradient(state)
            else:
                action = self.algo.explore(state)

            # Point at which initial action std statistics are calculated
            #print("env_step: ", env_step)
            #if env_step == 1000: #self.start_steps:
            #    self.algo.init_da_std()

            next_state, reward, done, _ = self.env.step(action)

            done_masked = done
            if ep_step == self.env._max_episode_steps:
                done_masked = False

            self.buffer.append(state, action, reward, done_masked, episode_done=done)
            self.buffer_mc.append(state, action, reward, done_masked, episode_done=done, env=self.env, policy=self.algo)
            if done:
                next_state = self.env.reset()
                ep_step = 0
            state = next_state

            if len(self.buffer) < self.batch_size:
                continue
            batch = self.buffer.sample(self.batch_size)
            batch_mc = self.buffer_mc.sample(self.batch_size)
            self.algo.update(batch, batch_mc)

            if env_step % self.eval_interval == 0:
                mean_reward = self.evaluate()
                wandb.log({"trainer/ep_reward": mean_reward, "env_step": env_step})
                wandb.log({"trainer/avg_reward": batch[2].mean(), "env_step": env_step})
                wandb.log({"trainer/buffer_transitions": len(self.buffer), "env_step": env_step})
                wandb.log({"trainer/buffer_episodes": self.buffer.num_episodes, "env_step": env_step})
                wandb.log({"trainer/buffer_last_ep_len": self.buffer.get_last_ep_len(), "env_step": env_step})

            if self.visualize_every > 0 and env_step % self.visualize_every == 0:
                imgs = self.visualize_policy()
                if imgs is not None:
                    wandb.log({"video": wandb.Video(imgs, fps=25, format="gif"), "env_step": env_step})

            if self.save_buffer_every > 0 and env_step % self.save_buffer_every == 0:
                self.buffer.save(f"{self.log_dir}/buffers/buffer_step_{env_step}.pickle")

            if self.estimate_q_every > 0 and env_step % self.estimate_q_every == 0:
                q_est = self.estimate_true_q()
                q_critic = self.estimate_critic_q()
                if q_est is not None:
                    wandb.log({"trainer/Q-estimate": q_est, "env_step": env_step})
                    wandb.log({"trainer/Q-critic": q_critic, "env_step": env_step})

            if env_step % self.stdout_log_every == 0:
                perc = int(env_step / self.num_steps * 100)
                print(f"Env step {env_step:8d} ({perc:2d}%) Avg Reward {batch[2].mean():10.3f}"
                      f"Ep Reward {mean_reward:10.3f}")