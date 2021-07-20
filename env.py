import gym
import dmc2gym
try:
    import pybulletgym
except ModuleNotFoundError as e:
    print("Failed to import pybullet gym. Install from https://github.com/benelot/pybullet-gym")


OPENAI_MUJOCO_PREFIX = [
    "Walker", "HalfCheetah", "Swimmer", "InvertedPendulum", "InvertedDoublePendulum",
    "Hopper", "Humanoid", "Reacher", "Ant"
]


def check_prefix(name, prefixes):
    for pref in prefixes:
        if pref in name:
            return True
    return False


def make_env(name: str, seed: int = 0):
    """
    Args:
        name: Environment name.
        seed: Random seed.
    """
    env = None
    if check_prefix(name, OPENAI_MUJOCO_PREFIX):
        env = gym.make(name)
        env.seed(seed)
    else:
        domain, task = name.split("-")
        env = dmc2gym.make(domain_name=domain, task_name=task, seed=seed)
    if env is None:
        raise ValueError(f"Non-supported environment: {name}")
    return env
