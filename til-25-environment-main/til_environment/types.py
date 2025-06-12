from enum import IntEnum, StrEnum, auto
from typing import TypedDict

import numpy as np
import pygame


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    @property
    def movement(self) -> np.ndarray:
        match self:
            case Direction.RIGHT:
                return np.array([1, 0])
            case Direction.DOWN:
                return np.array([0, 1])
            case Direction.LEFT:
                return np.array([-1, 0])
            case Direction.UP:
                return np.array([0, -1])


class Action(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


# every tile+player+wall combination is representable by a np.uint8
class Tile(IntEnum):
    NO_VISION = 0
    EMPTY = 1
    RECON = 2
    MISSION = 3

    def draw(
        self,
        canvas: pygame.Surface,
        x: int,
        y: int,
        square_size: int,
        x_corner: int = 0,
        y_corner: int = 0,
        vision=False,
    ):
        match self:
            case Tile.NO_VISION if vision:
                pygame.draw.rect(
                    canvas,
                    (80, 80, 80),
                    pygame.Rect(
                        x_corner + x * square_size,
                        y_corner + y * square_size,
                        square_size,
                        square_size,
                    ),
                )
            case Tile.RECON:
                pygame.draw.circle(
                    canvas,
                    (255, 165, 0),
                    (
                        x_corner + (x + 0.5) * square_size,
                        y_corner + (y + 0.5) * square_size,
                    ),
                    square_size / 10,
                )
            case Tile.MISSION:
                pygame.draw.circle(
                    canvas,
                    (147, 112, 219),
                    (
                        x_corner + (x + 0.5) * square_size,
                        y_corner + (y + 0.5) * square_size,
                    ),
                    square_size / 6,
                )


class Player(IntEnum):
    # do not exist as tiles in state, only in vision
    SCOUT = 2  # 2**2
    GUARD = 3  # 2**3

    @property
    def power(self) -> int:
        return 2**self.value

    @property
    def color(self) -> tuple[int, int, int]:
        match self:
            case Player.GUARD:
                return (255, 0, 0)
            case Player.SCOUT:
                return (0, 0, 255)

    def draw(
        self,
        canvas: pygame.Surface,
        x: int,
        y: int,
        square_size: int,
        x_corner: int = 0,
        y_corner: int = 0,
    ):
        pygame.draw.circle(
            canvas,
            self.color,
            (x_corner + (x + 0.5) * square_size, y_corner + (y + 0.5) * square_size),
            square_size / 3,
        )


class Wall(IntEnum):
    RIGHT = 4  # 2**4 = 16
    BOTTOM = 5  # 2**5 = 32
    LEFT = 6  # 2**6 = 64
    TOP = 7  # 2**7 = 128

    @property
    def power(self) -> int:
        return 2**self.value

    @property
    def orientation(self):
        # returns start x, start y, end x, end y
        # of the line for the wall to be drawn
        match self:
            case Wall.RIGHT:
                return (1, 0, 1, 1)
            case Wall.BOTTOM:
                return (0, 1, 1, 1)
            case Wall.LEFT:
                return (0, 0, 0, 1)
            case Wall.TOP:
                return (0, 0, 1, 0)

    @property
    def end_tile(self):
        # returns x, y of the tile on the other side of the wall
        match self:
            case Wall.RIGHT:
                return (1, 0)
            case Wall.BOTTOM:
                return (0, 1)
            case Wall.LEFT:
                return (-1, 0)
            case Wall.TOP:
                return (0, -1)

    def draw(
        self,
        canvas: pygame.Surface,
        x: int,
        y: int,
        square_size: int,
        x_corner: int = 0,
        y_corner: int = 0,
        width: int = 7,
    ):
        x1, y1, x2, y2 = self.orientation
        pygame.draw.line(
            canvas,
            0,
            (x_corner + square_size * (x + x1), y_corner + square_size * (y + y1)),
            (x_corner + square_size * (x + x2), y_corner + square_size * (y + y2)),
            width=width,
        )


class RewardNames(StrEnum):
    GUARD_WINS = auto()
    GUARD_CAPTURES = auto()
    SCOUT_CAPTURED = auto()
    SCOUT_RECON = auto()
    SCOUT_MISSION = auto()
    WALL_COLLISION = auto()
    AGENT_COLLIDER = auto()
    AGENT_COLLIDEE = auto()
    STATIONARY_PENALTY = auto()
    GUARD_TRUNCATION = auto()
    SCOUT_TRUNCATION = auto()
    GUARD_STEP = auto()
    SCOUT_STEP = auto()


class Observation(TypedDict):
    """
    Class representing an observation dictionary from the environment.

    * `viewcone` (np.ndarray): contents of the tile as a bit flag; 0 if not visible
    * `direction` (int): direction the agent is facing; see the `Direction` enum
    * `scout` (int): 1 if agent is the Scout, 0 otherwise
    * `location` (np.ndarray): x,y coordinate of the agent within the world
    """

    viewcone: np.ndarray
    direction: int
    scout: int
    location: np.ndarray
