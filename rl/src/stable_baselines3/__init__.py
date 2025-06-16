import os

from stable_baselines3.common.utils import get_system_info
from stable_baselines3.dqn import DQN

def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )


__all__ = [
    "DQN",
    "get_system_info",
]
