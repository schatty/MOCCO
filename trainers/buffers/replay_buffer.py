from collections import deque

import numpy as np
import torch


class NStepBuffer:

    def __init__(self, gamma=0.99, nstep=3):
        self.discounts = [gamma ** i for i in range(nstep)]
        self.nstep = nstep
        self.states = deque(maxlen=self.nstep)
        self.actions = deque(maxlen=self.nstep)
        self.rewards = deque(maxlen=self.nstep)

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self):
        assert len(self.rewards) > 0

        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self._nstep_reward()
        return state, action, reward

    def _nstep_reward(self):
        reward = np.sum([r * d for r, d in zip(self.rewards, self.discounts)])
        self.rewards.popleft()
        return reward

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.nstep

    def __len__(self):
        return len(self.rewards)


class ReplayBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device,
                 gamma, nstep, dtype=torch.float):
        """
        Args:
            buffer_size: Max number of transitions in buffer.
            state_shape: Shape of the state.
            action_shape: Shape of the action.
            device: Device to place buffer.
            gamma: Discount factor for N-step.
            nstep: N-step return.
            dtype: Data type.
        """
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        self.nstep = nstep

        self._pointer = 0
        self._cur_size = 0

        self.actions = torch.empty((buffer_size, *action_shape), dtype=dtype, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=dtype, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=dtype, device=device)

        if nstep != 1:
            self.nstep_buffer = NStepBuffer(gamma, nstep)

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state,
               episode_done=None):

        if self.nstep != 1:
            self.nstep_buffer.append(state, action, reward)

            if self.nstep_buffer.is_full():
                state, action, reward = self.nstep_buffer.get()
                self._append(state, action, reward, done, next_state)

            if done or episode_done:
                while not self.nstep_buffer.is_empty():
                    state, action, reward = self.nstep_buffer.get()
                    self._append(state, action, reward, done, next_state)
        else:
            self._append(state, action, reward, done, next_state)

    def _append(self, state, action, reward, done, next_state):
        self.states[self._pointer].copy_(torch.from_numpy(state))
        self.next_states[self._pointer].copy_(torch.from_numpy(next_state)) 
        self.actions[self._pointer].copy_(torch.from_numpy(action))
        self.rewards[self._pointer] = float(reward)
        self.dones[self._pointer] = float(done)

        self._pointer = (self._pointer + 1) % self.buffer_size
        self._cur_size = min(self._cur_size + 1, self.buffer_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._cur_size, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )