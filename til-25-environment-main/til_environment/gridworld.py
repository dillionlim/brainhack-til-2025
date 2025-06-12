import functools
import logging
import warnings
from functools import partial

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.seeding import np_random
from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper
from supersuit import frame_stack_v2
from til_environment.flatten_dict import FlattenDictWrapper
from til_environment.helpers import (
    convert_tile_to_edge,
    get_bit,
    idx_to_view,
    is_idx_valid,
    is_world_coord_valid,
    manhattan,
    rotate_right,
    supercover_line,
    view_to_idx,
    view_to_world,
    world_to_view,
)
from til_environment.types import Action, Direction, Player, RewardNames, Tile, Wall

NUM_ITERS = 100


DEFAULT_REWARDS_DICT = {
    RewardNames.GUARD_CAPTURES: 50,
    RewardNames.SCOUT_CAPTURED: -50,
    RewardNames.SCOUT_RECON: 1,
    RewardNames.SCOUT_MISSION: 5,
}


def env(
    env_wrappers: list[BaseWrapper] | None = None,
    render_mode: str | None = None,
    **kwargs,
):
    """
    Main entrypoint to the environment, allowing configuration of render mode
    and what wrappers to wrap around the environment. If you write a custom
    wrapper(s), pass them in a list to `env_wrappers`.
    See `flatten_dict.FlattenDictWrapper` for a very simple wrapper example.
    """
    env = raw_env(render_mode=render_mode, **kwargs)
    if env_wrappers is None:
        env_wrappers = [
            FlattenDictWrapper,
            partial(frame_stack_v2, stack_size=4, stack_dim=-1),
        ]
    for wrapper in env_wrappers:
        env = wrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv[AgentID, ObsType, ActionType]):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "til_env",
        "render_fps": 10,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        render_mode=None,
        window_size: int = 768,
        rewards_dict: dict[str, int] | None = None,
        novice: bool = False,
        debug: bool = False,
    ):
        """
        The init method takes in environment arguments and defines the following attributes:
        - `render_mode`: Render mode to be used; one of "human", "rgb_array", or None.
        - `window_size`: The window size (in px) of the environment when being rendered
        - `rewards_dict`: Mapping of reward names to reward values. If None, defaults to DEFAULT_REWARDS_DICT
        - `novice`: Whether to use Novice map generation (fixed seed) or Advanced map generation (random seeds)
        - `debug`: Whether to display the debug window in render, and to log debug information
        """
        # initialize grid
        self.debug = debug
        self.size: int = 16  # size of the square grid
        self.window_size = window_size  # vertical size of the PyGame window
        self.window_width = int(window_size * 1.5) if self.debug else window_size
        self.possible_agents = ["player_" + str(r) for r in range(4)]
        self.viewcone: tuple[int, int, int, int] = (2, 2, 2, 4)
        self._arena = None
        self.logger = logging.getLogger(__name__)
        self.num_moves = 0

        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        # initialize random generation parameters
        self.mission_prob: float = 0.2
        self.wall_prob: float = 0.2
        self.novice = novice

        # viewcone
        self.viewcone_width = self.viewcone[0] + self.viewcone[1] + 1
        self.viewcone_length = self.viewcone[2] + self.viewcone[3] + 1

        # mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.font = None

        # initialize selector for scout
        self._scout_selector = AgentSelector(self.possible_agents)

        # set rewards dictionary
        self.rewards_dict = (
            DEFAULT_REWARDS_DICT if rewards_dict is None else rewards_dict
        )

        # initialize random
        self._np_random, self._np_random_seed = None, None

        # initialize walls
        # walls in this case are ordered pairs of coordinates
        self.walls: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self._maze = Maze()

    # memoize/cache observation and action spaces since they don't change
    # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        return Dict(
            {
                "viewcone": Box(
                    0,
                    2**8 - 1,
                    shape=(
                        self.viewcone_length,
                        self.viewcone_width,
                    ),
                    dtype=np.uint8,
                ),
                "direction": Discrete(len(Direction)),
                "scout": Discrete(2),
                "location": Box(0, self.size, shape=(2,), dtype=np.uint8),
                "step": Discrete(NUM_ITERS),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID):
        return Discrete(len(Action))

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode in self.metadata["render_modes"]:
            return self._render_frame()

    # drawing text in pygame
    def _draw_text(self, text, text_col="black", **kwargs):
        if self.font is not None:
            img = self.font.render(text, True, text_col)
            rect = img.get_rect(**kwargs)
            self.window.blit(img, rect)

    def _draw_gridlines(
        self,
        max_x: int,
        max_y: int,
        square_size: int,
        x_corner: int = 0,
        y_corner: int = 0,
        width: int = 3,
    ):
        for x in range(max_x + 1):
            pygame.draw.line(
                self.window,
                (211, 211, 211),
                (x_corner + square_size * x, y_corner),
                (x_corner + square_size * x, y_corner + square_size * max_y),
                width=width,
            )
        for y in range(max_y + 1):
            pygame.draw.line(
                self.window,
                (211, 211, 211),
                (x_corner, y_corner + square_size * y),
                (x_corner + square_size * max_x, y_corner + square_size * y),
                width=width,
            )

    def _render_frame(self):
        if self.window is None and self.render_mode in self.metadata["render_modes"]:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode(
                    (self.window_width, self.window_size)
                )
                pygame.display.set_caption("TIL-AI 2025 Environment")
            else:
                self.window = pygame.Surface((self.window_width, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if self.font is None:
            try:
                self.font = pygame.font.Font("freesansbold.ttf", 12)
            except:
                warnings.warn("unable to import font")

        self.window.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # add gridlines
        self._draw_gridlines(self.size, self.size, pix_square_size)

        # draw environment tiles
        for x, y in np.ndindex((self.size, self.size)):
            tile = self._state[x, y]
            # draw whether the tile contains points
            Tile(tile % 4).draw(self.window, x, y, pix_square_size)

            # draw walls
            for wall in Wall:
                if not get_bit(tile, wall.value):
                    continue
                wall.draw(self.window, x, y, pix_square_size)

        # draw all the players
        for agent, location in self.agent_locations.items():
            p = Player.SCOUT if agent == self.scout else Player.GUARD
            p.draw(self.window, location[0], location[1], pix_square_size)

            center = (location + 0.5) * pix_square_size
            # draw direction indicator
            pygame.draw.line(
                self.window,
                (0, 255, 0),
                center,
                (
                    location
                    + 0.5
                    + Direction(self.agent_directions[agent]).movement * 0.33
                )
                * pix_square_size,
                3,
            )
            self._draw_text(agent[-1], center=center)

        # draw debug view
        if self.debug:
            # dividing allowable vertical space (0.2 of the vertical window)
            subpix_square_size = int(0.2 * self.window_size / self.viewcone_width)
            x_corner = int(self.window_size * 1.04)
            x_lim = int(self.window_size * 1.47)
            for agent in self.agents:
                agent_id = int(agent[-1])
                observation = self.observe(agent)

                y_corner = int(self.window_size * (0.24 * agent_id + 0.04))

                # draw gridlines
                self._draw_gridlines(
                    self.viewcone_length,
                    self.viewcone_width,
                    subpix_square_size,
                    x_corner,
                    y_corner,
                )

                # draw debug text information
                for i, text in enumerate(
                    [
                        f"id: {agent[-1]}",
                        f"direction: {observation['direction']}",
                        f"scout: {observation['scout']}",
                        f"reward: {self.rewards[agent]:.1f}",
                        f"location: {self.agent_locations[agent]}",
                        f"action {self.num_moves}: {self.actions.get(agent)}",
                    ]
                ):
                    self._draw_text(
                        text,
                        topright=(x_lim, y_corner + i * 15),
                    )

                # plot observation
                for x, y in np.ndindex((self.viewcone_length, self.viewcone_width)):
                    tile = observation["viewcone"][x, y]
                    # draw whether the tile contains points
                    Tile(tile % 4).draw(
                        self.window, x, y, subpix_square_size, x_corner, y_corner, True
                    )
                    for player in Player:
                        if not get_bit(tile, player.value):
                            continue
                        player.draw(
                            self.window, x, y, subpix_square_size, x_corner, y_corner
                        )

                for x, y in np.ndindex((self.viewcone_length, self.viewcone_width)):
                    tile = observation["viewcone"][x, y]
                    # draw walls
                    for wall in Wall:
                        if not get_bit(tile, wall.value):
                            continue
                        wall.draw(
                            self.window, x, y, subpix_square_size, x_corner, y_corner
                        )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def _is_visible(self, agent, end: np.ndarray):
        start = self.agent_locations[agent]
        # if it's a self-to-self check, return true
        if (start == end).all():
            return True

        # get tiles covered by LOS line from start to end
        path = supercover_line(start, end)
        for i in range(len(path) - 1):
            tile, next_tile = path[i], path[i + 1]

            # check if tile is diagonal
            if tile[0] - next_tile[0] != 0 and tile[1] - next_tile[1] != 0:
                horiz0 = tuple(sorted((tile, (next_tile[0], tile[1]))))
                horiz1 = tuple(sorted(((next_tile[0], tile[1]), next_tile)))
                vert0 = tuple(sorted((tile, (tile[0], next_tile[1]))))
                vert1 = tuple(sorted(((tile[0], next_tile[1]), next_tile)))

                # terminate if neither horiz nor vert direction is open
                # allows for (and essentially hardcodes) corner-peeking
                if (horiz0 in self.walls or horiz1 in self.walls) and (
                    vert0 in self.walls or vert1 in self.walls
                ):
                    return False
            else:
                # if not diagonal, check the edge normally
                edge = tuple(sorted((tile, next_tile)))
                if edge in self.walls:
                    return False
        # if you made it to the end, you have visibility
        return True

    def observe(self, agent):
        """
        Returns the observation of the specified agent.
        """
        view = np.zeros((self.viewcone_length, self.viewcone_width), dtype=np.uint8)
        direction = Direction(self.agent_directions[agent])
        location = self.agent_locations[agent]
        for idx in np.ndindex((self.viewcone_length, self.viewcone_width)):
            view_coord = idx_to_view(np.array(idx), self.viewcone)
            world_coord = view_to_world(location, direction, view_coord)
            if not is_world_coord_valid(world_coord, self.size):
                continue
            # check if tile is visible
            if self._is_visible(agent, world_coord):
                # in theory we should filter the state to only include the visible walls, but whatever
                val = self._state[tuple(world_coord)]
                points = val % 4
                # shift orientation of the tile to local position
                view[idx] = (
                    rotate_right(val >> 4, direction.value, bit_width=4) << 4
                ) + points

        # add players
        for _agent, loc in self.agent_locations.items():
            view_coord = world_to_view(location, direction, loc)
            idx = view_to_idx(view_coord, self.viewcone)
            # check only if player is within viewcone, not whether tile is actually visible
            # this lets you "hear" nearby players without seeing them
            if is_idx_valid(idx, self.viewcone_length, self.viewcone_width):
                view[idx] += (
                    np.uint8(Player.SCOUT.power)
                    if _agent == self.scout
                    else np.uint8(Player.GUARD.power)
                )
        return {
            "viewcone": view,
            "direction": self.agent_directions[agent],
            "location": self.agent_locations[agent],
            "scout": 1 if agent == self.scout else 0,
            "step": self.num_moves,
        }

    def close(self):
        """
        Close cleans up the pygame graphical display that is
        no longer needed once you're done with the environment.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _add_wall(self, a: tuple[int, int], b: tuple[int, int]):
        self.walls.add(tuple(sorted([a, b])))

    def _add_walls(self):
        # reset existing walls
        self.walls = set()

        # iterate over arena state and add the walls on each tile
        for x, y in np.ndindex((self.size, self.size)):
            tile = self._arena[x, y]
            for wall in Wall:
                if not get_bit(tile, wall.value):
                    continue
                end_tile = wall.end_tile
                self._add_wall((x, y), (x + end_tile[0], y + end_tile[1]))

    def _random_room(self, min_dim: int = 2, max_dim: int = 5):
        # generate starting point
        start = self._np_random.integers(0, self.size - max_dim, size=2)
        end = np.array(
            (
                self._np_random.integers(start[0] + min_dim, start[0] + max_dim),
                self._np_random.integers(start[1] + min_dim, start[1] + max_dim),
            )
        )
        # randomly flip the axes
        if self._np_random.random() >= 0.5:
            return [tuple(start * 2 + 1), tuple(end * 2 + 1)]
        else:
            return [tuple(np.flip(start) * 2 + 1), tuple(np.flip(end) * 2 + 1)]

    def _new_maze_generator(self, min_rooms: int = 5, max_rooms: int = 10):
        # generate rooms for maze
        # starting 3x3 home base room for scout in the top-left corner
        rooms = [[(1, 1), (5, 5)]]
        for _ in range(self._np_random.integers(min_rooms, max_rooms)):
            rooms.append(self._random_room())
        return DungeonRooms(self.size, self.size, rooms=rooms)

    # randomly generate new arena
    def _generate_arena(self):
        self.logger.debug("generating new arena...")
        # generate the same arena every time for novice
        if self.novice:
            self._init_random(19)
        self._maze.generator = self._new_maze_generator()
        self._maze.generate()
        # randomly knock down some walls to open up new pathways
        # get all indices of the walls, excluding exterior walls
        _grid = self._maze.grid.copy()
        _grid[0, :] = 0  # top edge
        _grid[-1, :] = 0  # bottom edge
        _grid[:, 0] = 0  # left edge
        _grid[:, -1] = 0  # right edge
        # exclude corner bits
        _grid = _grid * (np.indices(_grid.shape).sum(axis=0) % 2)

        walls = np.where(_grid == 1)
        # drop some % of walls to open up more pathways
        idx = self._np_random.choice(
            walls[0].shape[0],
            size=int(walls[0].shape[0] * self.wall_prob),
            replace=False,
        )
        self._maze.grid[walls[0][idx], walls[1][idx]] = 0

        # add recon and mission tiles
        self._arena = self._np_random.choice(
            (np.uint8(Tile.RECON), np.uint8(Tile.MISSION)),
            size=(self.size, self.size),
            p=(1 - self.mission_prob, self.mission_prob),
        )
        convert_tile_to_edge(self._arena, self._maze.grid)
        self._add_walls()

        # select starting directions and locations
        self.starting_directions = self._np_random.integers(0, 4, size=4)
        self.starting_locations = np.array(
            [
                (0, 0),
                (
                    self._np_random.integers(0, self.size // 2),
                    self._np_random.integers(self.size // 2, self.size),
                ),
                (
                    self._np_random.integers(self.size // 2, self.size),
                    self._np_random.integers(0, self.size // 2),
                ),
                self._np_random.integers(self.size // 2, self.size, size=2),
            ],
        )

    # reset state to pregenerated arena
    def _reset_state(self):
        self._state = self._arena.copy()
        _dirs: dict[AgentID, np.int64] = {self.scout: self.starting_directions[0]}
        _locs: dict[AgentID, np.ndarray] = {self.scout: self.starting_locations[0]}
        for i, agent in enumerate([a for a in self.agents if a != self.scout], 1):
            _dirs[agent] = self.starting_directions[i]
            _locs[agent] = self.starting_locations[i]
        self.agent_directions = _dirs
        self.agent_locations = _locs

    def _init_random(self, seed: int | None = None):
        self._np_random, self._np_random_seed = np_random(seed)
        # _maze only accepts seeds up to 2**32 - 1
        self._maze.set_seed(self._np_random_seed % 2**32)
        self.logger.debug(f"seeded with {self._np_random_seed}")

    def reset(self, seed=None, options=None):
        """
        Resets the environment. MUST be called before training to set up the environment.
        Automatically selects the next agent to be the Scout, and generates a new arena for each match.

        Call with `seed` to seed internal numpy RNG.
        `options` dictionary is ignored.
        """
        if self._np_random is None or seed is not None:
            self._init_random(seed)

        # agent_selector utility cyclically steps through agents list
        self.agents = self.possible_agents[:]
        self.agent_selector = AgentSelector(self.agents)
        self.agent_selection = self.agent_selector.next()
        # select the next player to be the scout
        self.scout: AgentID = self._scout_selector.next()
        # generate arena for each match for advanced track
        if self._arena is None or (self._scout_selector.is_first() and not self.novice):
            self._generate_arena()

        self._reset_state()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.actions: dict[AgentID, Action] = {}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.num_moves = 0

        if self.render_mode in self.metadata["render_modes"]:
            self.render()

    def state(self):
        """
        Returns the underlying state of the environment, provided for debugging.
        NOT to be used in training, as it contains information that agents don't know
        """
        return self._state

    # reward actions
    def _capture_scout(self, capturers):
        """
        Given a list of agents who captured the scout, processes those agents' capture of the scout.
        Terminates the game and gives guards and the scout corresponding rewards.
        """
        self.logger.debug(f"{capturers} have captured the scout")
        # scout gets captured, terminate and reward guard
        self.terminations = {agent: True for agent in self.agents}
        for agent in self.agents:
            if agent == self.scout:
                self.rewards[self.scout] += self.rewards_dict.get(
                    RewardNames.SCOUT_CAPTURED, 0
                )
                continue
            self.rewards[agent] += self.rewards_dict.get(RewardNames.GUARD_WINS, 0)
            if agent in capturers:
                self.rewards[agent] += self.rewards_dict.get(
                    RewardNames.GUARD_CAPTURES, 0
                )

    def _handle_agent_collision(self, agent1: AgentID, agent2: AgentID):
        """
        Given two agents, handle agent1 colliding into agent2
        """
        self.logger.debug(f"{agent1} collided with {agent2}")
        self.rewards[agent1] += self.rewards_dict.get(RewardNames.AGENT_COLLIDER, 0)
        self.rewards[agent2] += self.rewards_dict.get(RewardNames.AGENT_COLLIDEE, 0)

    def _handle_wall_collision(self, agent: AgentID):
        self.logger.debug(f"{agent} collided with a wall")
        self.rewards[agent] += self.rewards_dict.get(RewardNames.WALL_COLLISION, 0)

    # game rules
    def _enforce_collisions(
        self, agent: AgentID, direction: Direction
    ) -> tuple[np.ndarray, bool]:
        """
        Given agent and direction of movement, return:
        * resultant movement direction, and
        * the AgentID of any agent collided into, or None if no collision happened
        """
        # check walls on agent's tile
        tile = self._state[tuple(self.agent_locations[agent])]
        # add 4 bc direction+4=wall bit
        if get_bit(tile, direction.value + 4):
            self._handle_wall_collision(agent)
            return np.array([0, 0]), None

        # if good, check reverse wall on destination tile
        next_loc = self.agent_locations[agent] + direction.movement
        next_tile = self._state[tuple(next_loc)]
        if get_bit(next_tile, ((direction.value + 2) % 4) + 4):
            self._handle_wall_collision(agent)
            return np.array([0, 0]), None

        # if that's good too, check for agent collisions
        for _agent, _loc in self.agent_locations.items():
            if _agent == agent:
                continue
            if (next_loc == _loc).all():
                self._handle_agent_collision(agent, _agent)
                # return the collidee
                return np.array([0, 0]), _agent

        # checks out, return original direction
        return direction.movement, None

    def _move_agent(self, agent: AgentID, action: int):
        """
        Updates agent location, accruing rewards along the way
        return the name of the agent collided into, or None
        """
        _action = Action(action)
        if _action in (Action.FORWARD, Action.BACKWARD):
            _direction = Direction(
                self.agent_directions[agent]
                if _action is Action.FORWARD
                else (self.agent_directions[agent] + 2) % 4
            )
            # enforce collisions with walls and other agents
            direction, collision = self._enforce_collisions(agent, _direction)
            # use np.clip to not leave grid
            self.agent_locations[agent] = np.clip(
                self.agent_locations[agent] + direction, 0, self.size - 1
            )
            # update scout rewards
            if agent == self.scout:
                x, y = self.agent_locations[agent]
                tile = self._state[x, y]
                match Tile(tile % 4):
                    case Tile.RECON:
                        self.rewards[self.scout] += self.rewards_dict.get(
                            RewardNames.SCOUT_RECON, 0
                        )
                        self._state[x, y] -= Tile.RECON.value - Tile.EMPTY.value
                    case Tile.MISSION:
                        self.rewards[self.scout] += self.rewards_dict.get(
                            RewardNames.SCOUT_MISSION, 0
                        )
                        self._state[x, y] -= Tile.MISSION.value - Tile.EMPTY.value
            return collision
        if _action in (Action.LEFT, Action.RIGHT):
            # update direction of agent, right = +1 and left = -1 (which is equivalent to +3), mod 4.
            self.agent_directions[agent] = (
                self.agent_directions[agent] + (3 if _action is Action.LEFT else 1)
            ) % 4
        if _action is (Action.STAY):
            # apply stationary penalty
            self.rewards[agent] += self.rewards_dict.get(
                RewardNames.STATIONARY_PENALTY, 0
            )
        return None

    def _handle_actions(self):
        """
        Handles all actions at once, handling guard-scout captures
        """
        capturers = []
        scout_action = self.actions.pop(self.scout)

        # update scout location first
        def_collision = self._move_agent(self.scout, scout_action)
        # if the scout walks into an guard, mark that guard as a capturer
        if def_collision is not None:
            capturers.append(def_collision)

        # now update all the guards
        for agent, action in self.actions.items():
            # update agent location
            agent_collision = self._move_agent(agent, action)
            # if this guard walks into the scout, mark it as a capturer
            if agent_collision == self.scout:
                capturers.append(agent)

        # if there are any capturers, capture the scout
        if len(capturers) > 0:
            self._capture_scout(capturers)

        # put the scout action back into the dict
        self.actions[self.scout] = scout_action

    def get_info(self, agent: AgentID):
        """
        Returns accessory info for training/reward shaping
        """
        return {
            "distance": np.linalg.norm(
                self.agent_locations[agent] - self.agent_locations[self.scout],
                ord=1,
            ),
            "manhattan": manhattan(
                self.agent_locations[agent],
                self.agent_locations[self.scout],
            ),
        }

    def step(self, action: ActionType):
        """
        Takes in an action for the current agent (specified by agent_selection),
        only updating internal environment state when all actions have been received.
        """
        if self.agent_selector.is_first():
            # clear actions from previous round
            self.actions = {}

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent, or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.actions[self.agent_selection] = action

        # handle actions and rewards if it is the last agent to act
        if self.agent_selector.is_last():
            # execute actions
            self._handle_actions()

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            if self.num_moves >= NUM_ITERS:
                self.truncations = {agent: True for agent in self.agents}
                for agent in self.agents:
                    self.rewards[agent] += (
                        self.rewards_dict.get(RewardNames.SCOUT_TRUNCATION, 0)
                        if agent == self.scout
                        else self.rewards_dict.get(RewardNames.GUARD_TRUNCATION, 0)
                    )
            else:
                for agent in self.agents:
                    self.rewards[agent] += (
                        self.rewards_dict.get(RewardNames.SCOUT_STEP, 0)
                        if agent == self.scout
                        else self.rewards_dict.get(RewardNames.GUARD_STEP, 0)
                    )

            # observe the current state and get new infos
            for agent in self.agents:
                self.observations[agent] = self.observe(agent)
                # update infos
                self.infos[agent] = self.get_info(agent)

            # render
            if self.render_mode in self.metadata["render_modes"]:
                self.render()
        else:
            # no rewards are allocated until all players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self.agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
