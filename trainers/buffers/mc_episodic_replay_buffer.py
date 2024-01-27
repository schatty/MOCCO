import pickle
import numpy as np
import torch


class MCEpisodicReplayBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, gamma, max_episode_len=1000, dtype=torch.float):
        """
        Args:
            buffer_size: Max number of transitions in buffer.
            state_shape: Shape of the state.
            action_shape: Shape of the action.
            device: Device to place buffer.
            gamma: Discount factor for N-step.
            max_episode_len: Max length of the episode to store.
            dtype: Data type.
        """
        self.buffer_size = buffer_size
        self.max_episodes = buffer_size // max_episode_len
        self.max_episode_len = max_episode_len
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

        self.ep_pointer = 0
        self.cur_episodes = 1
        self.cur_size = 0

        self.actions = torch.empty((self.max_episodes, max_episode_len, *action_shape), dtype=dtype, device=device)
        self.rewards = torch.empty((self.max_episodes, max_episode_len, 1), dtype=dtype, device=device)
        self.dones = torch.empty((self.max_episodes, max_episode_len, 1), dtype=dtype, device=device)
        self.states = torch.empty((self.max_episodes, max_episode_len + 1, *state_shape), dtype=dtype, device=device)
        self.qs = torch.empty((self.max_episodes, max_episode_len, 1), dtype=dtype, device=device)
        self.ep_lens = [0] * self.max_episodes

    def append(self, state, action, reward, done, episode_done=None, env=None, policy=None):
        """
        Args:
            state: state.
            action: action.
            reward: reward.
            done: done only if episode ends naturally.
            episode_done: done that can be set to True if time limit is reached.
        """
        self.states[self.ep_pointer, self.ep_lens[self.ep_pointer]].copy_(torch.from_numpy(state))
        self.actions[self.ep_pointer, self.ep_lens[self.ep_pointer]].copy_(torch.from_numpy(action))
        self.rewards[self.ep_pointer, self.ep_lens[self.ep_pointer]] = float(reward)
        self.dones[self.ep_pointer, self.ep_lens[self.ep_pointer]] = float(done)

        self.ep_lens[self.ep_pointer] += 1
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
        if episode_done:
            q_calucated = self._calc_q(self.ep_pointer, state, done, env, policy).unsqueeze(1)
            # print("q_calucated: ", type(q_calucated))
            # print(q_calucated.shape)
            self.qs[self.ep_pointer, :self.ep_lens[self.ep_pointer], :] = q_calucated
            self.ep_pointer = (self.ep_pointer + 1) % self.max_episodes
            self.cur_episodes = min(self.cur_episodes + 1, self.max_episodes)
            self.cur_size -= self.ep_lens[self.ep_pointer]
            self.ep_lens[self.ep_pointer] = 0

    def _calc_q(self, ep_i, state, done, env, policy):
        n = self.ep_lens[ep_i]
        # Study this more closely
        # if not done:
        #     rewards_add = []
        #     s = state
        #     # Here we have an assumtion that if an agent reaches its maximum
        #     # episode length without done=True from natural reasons, it won't have
        #     # same done=True for the next max_len steps...
        #     for i in range(self.max_episode_len):
        #         a = policy.exploit(s)
        #         s, r, teriminated, truncated, _ = env.step(a)
        #         if i < 10:
        #             print("Appenging reward: ", r)
        #         done = teriminated or truncated
        #         rewards_add.append(r)
        #     print("rewards_add: ", rewards_add)
        #     print("Looking at float64: ", torch.float64)
        #     rewards_add_float = torch.tensor(rewards_add, dtype=torch.float64)
        #     print("Converted to float")
        #     rewards = torch.cat((self.rewards[self.ep_pointer].flatten().cpu(), torch.tensor(rewards_add, dtype=torch.float)))
        # else:
        rewards = self.rewards[self.ep_pointer, :n].flatten().cpu()
        discounts = torch.pow(torch.ones(n) * self.gamma, torch.arange(0, n))

        qs = []
        for i in range(n):
            j = min(len(rewards), i + n)
            # print(rewards.shape, discounts.shape, len(rewards[i:j]), i, j)
            qs.append(torch.sum(rewards[i:j] * discounts[:j - i]))
        return torch.tensor(qs).float().to(self.device)

    def _inds_to_episodic(self, inds):
        start_inds = np.cumsum([0] + self.ep_lens[:self.cur_episodes - 1])
        end_inds = start_inds + np.array(self.ep_lens[:self.cur_episodes])
        ep_inds = np.argmin(inds.reshape(-1, 1) >= np.tile(end_inds, (len(inds), 1)), axis=1)
        step_inds = inds - start_inds[ep_inds]

        return ep_inds, step_inds

    def sample(self, batch_size):
        inds = np.random.randint(low=0, high=self.cur_size, size=batch_size)
        ep_inds, step_inds = self._inds_to_episodic(inds)

        return (
            self.states[ep_inds, step_inds],
            self.actions[ep_inds, step_inds],
            self.qs[ep_inds, step_inds],
        )

    @property
    def num_episodes(self):
        return self.cur_episodes

    def get_last_ep_len(self):
        return self.ep_lens[self.ep_pointer]
