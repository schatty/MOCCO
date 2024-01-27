import numpy as np

from .buffers.mc_episodic_replay_buffer import MCEpisodicReplayBuffer
from .base_trainer import BaseTrainer


class MOCCOTrainer(BaseTrainer):

    def __init__(self, state_shape=None, action_shape=None, env=None, make_env_test=None, algo=None, buffer_size=int(1e6),
                 gamma=0.99, device=None, num_steps=int(1e6), start_steps=int(1e3), batch_size=128,
                 eval_interval=int(2e3), num_eval_episodes=10, log_every=int(1e5), seed=0, logger=None):
        """
        Args:
            state_shape: Shape of the state.
            action_shape: Shape of the action.
            env: Enviornment object.
            env_test: Environment object for evaluation.
            algo: Codename for the algo.
            buffer_size: Buffer size in transitions.
            gamma: Discount factor.
            device: Name of the device.
            num_step: Number of env steps to train.
            start_steps: Number of environment steps not to perform training at the beginning.
            batch_size: Batch-size.
            eval_interval: Number of env step after which perform evaluation.
            log_every: Number of evn steps after which log info to stdout.
            seed: Random seed.
            logger: Logger instance.
        """
        super().__init__(state_shape=state_shape, action_shape=action_shape, env=env, make_env_test=make_env_test,
                         algo=algo, buffer_size=buffer_size, gamma=gamma, device=device, num_steps=num_steps,
                         start_steps=start_steps, batch_size=batch_size, eval_interval=eval_interval,
                         num_eval_episodes=num_eval_episodes, 
                         seed=seed, log_every=log_every, logger=logger)

        self.buffer_mc = MCEpisodicReplayBuffer(
            buffer_size=100000,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            gamma=gamma
        )

    def train(self):
        ep_step = 0
        mean_reward = 0
        state, _ = self.env.reset()

        for env_step in range(self.num_steps + 1):
            ep_step += 1
            if env_step <= self.start_steps:
                action = self.env.sample_action()
                self.algo.accumulate_action_gradient(state)
            else:
                action = self.algo.explore(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.append(state, action, reward, terminated, episode_done=done)
            self.buffer_mc.append(state, action, reward, terminated, episode_done=done,
                                  env=self.env, policy=self.algo)
            if done:
                next_state, _ = self.env.reset(seed=self.seed)
                ep_step = 0
            state = next_state

            if len(self.buffer) < self.batch_size:
                continue

            batch = self.buffer.sample(self.batch_size)
            batch_mc = self.buffer_mc.sample(self.batch_size)
            self.algo.update(batch, batch_mc)

            if env_step % self.eval_interval == 0:
                mean_reward = self.evaluate()
                self.logger.log_scalar("trainer/ep_reward", mean_reward, env_step)
                self.logger.log_scalar("trainer/avg_reward", batch[2].mean(), env_step)
                self.logger.log_scalar("trainer/buffer_transitions", len(self.buffer), env_step)
                self.logger.log_scalar("trainer/buffer_episodes", self.buffer.num_episodes, env_step)
                self.logger.log_scalar("trainer/buffer_last_ep_len", self.buffer.get_last_ep_len(), env_step)

            if env_step % self.log_every == 0:
                perc = int(env_step / self.num_steps * 100)
                print(f"Env step {env_step:8d} ({perc:2d}%) Avg Reward {batch[2].mean():10.3f}"
                      f"Ep Reward {mean_reward:10.3f}")
