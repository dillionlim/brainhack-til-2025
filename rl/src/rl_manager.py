"""Manages the RL model."""
from importlib import import_module
from scout_manager import ScoutManager


class RLManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        
        self.guard_manager = getattr(import_module("queso-v6.rl_manager"), "RLManager")()
        self.scout_manager = ScoutManager()
        self.manager = None
        pass

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        if not self.manager:
            self.manager = self.scout_manager if observation["scout"] else self.guard_manager

        # Your inference code goes here.

        return self.manager.rl(observation)

