from collections import OrderedDict

import numpy as np
import gymnasium as gym
from dm_control import suite


OPENAI_MUJOCO_PREFIX = [
    "Walker", "HalfCheetah", "Swimmer", "InvertedPendulum", "InvertedDoublePendulum",
    "Hopper", "Humanoid", "Reacher", "Ant"
]


def check_prefix(name, prefixes):
    for pref in prefixes:
        if pref in name:
            return True
    return False


class GymEnv:
    def __init__(self, name):
        self.env = gym.make(name)

    def reset(self, seed: int):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def sample_action(self):
        return self.env.action_space.sample()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


class DMControlEnv:

    def __init__(self, domain_name: str, task_name: str, seed: int):
        self.random_state = np.random.RandomState(seed)
        self.env = suite.load(domain_name, task_name, task_kwargs={"random": self.random_state})

    def reset(self, *args, **kwargs):
        obs = self._flat_obs(self.env.reset().observation)
        return obs, {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flat_obs(time_step.observation)

        terminated = False
        truncated = self.env._step_count >= self.env._step_limit

        return obs, time_step.reward, terminated, truncated, {}

    def sample_action(self):
        spec = self.env.action_spec()
        action = self.random_state.uniform(spec.minimum, spec.maximum, spec.shape) 
        return action

    @property
    def observation_space(self):
        return np.zeros(sum(int(np.prod(v.shape)) for v in self.env.observation_spec().values()))

    @property
    def action_space(self):
        return np.zeros(self.env.action_spec().shape[0])

    def _flat_obs(self, obs: OrderedDict):
        obs_flatten = []
        for _, o in obs.items():
            if len(o.shape) == 0:
                obs_flatten.append(np.array([o]))
            elif len(o.shape) == 2 and o.shape[1] > 1:
                obs_flatten.append(o.flatten())
            else:
                obs_flatten.append(o)
        return np.concatenate(obs_flatten)


def make_env(name: str, seed: int):
    """
    Args:
        name: Environment name.
    """
    if check_prefix(name, OPENAI_MUJOCO_PREFIX):
        env = GymEnv(name=name)
    else:
        domain, task = name.split("-")
        env = DMControlEnv(domain_name=domain, task_name=task, seed=seed)

    return env
