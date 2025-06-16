"""Manages the RL model."""
from enum import IntEnum

import numpy as np
from stable_baselines3 import DQN

class Action(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class Tile(IntEnum):
    NO_VISION = 0
    EMPTY = 1
    RECON = 2
    MISSION = 3

class RLManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.model = DQN.load("ensemble_scouts/caracas-8M/caracas_noGuards")
        self.size = 16

        # observation space
        self.VIEW_WINDOW_SIZE = 1 + 15 * 2
        self.BITS_PER_TILE = 10 # walls (4), rewards (2), seen (1), within bounds (1), repeat count (1), guards (1)
        self.APPENDED_BITS = 4 # if last 3 actions were turning (1), location x, y (2), current steps (1)
        self.OBS_DIM_SIZE = self.APPENDED_BITS + self.BITS_PER_TILE * (self.VIEW_WINDOW_SIZE ** 2)

        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_rewards_space = np.full((self.size, self.size), 255, dtype=np.uint8)

        self.obs_guard_space = np.full((self.size, self.size), 16, dtype=np.uint8)
        guard_discrete_levels_list = [0.5 - 0.5 / (1.2 ** i) for i in range(16)]
        guard_discrete_levels_list.append(0.5)
        guard_discrete_levels_list.extend([0.5 + 0.5 / (1.2 ** (15-i)) for i in range(16)])
        self.GUARD_DISCRETE_LEVELS = np.array(guard_discrete_levels_list, dtype=np.float32)

        self.obs_repeat_space = np.zeros((self.size, self.size), dtype=np.float32)
        
        self.consecutive_turns = 0

    def rl(self, observation: dict[str, int | list[int]], prev_action) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        if prev_action in (Action.LEFT, Action.RIGHT,):
            self.consecutive_turns += 1
        elif prev_action != Action.STAY:
            self.consecutive_turns = 0

        self.obs_guard_space[self.obs_guard_space > 16] -= 1
        self.obs_guard_space[self.obs_guard_space < 16] += 1

        new_gridview = np.array(observation["viewcone"], dtype=np.uint8)
        curr_direction = np.array(observation["direction"], dtype=np.int64) # right down left up
        curr_location = np.array(observation["location"], dtype=np.int64)

        self.obs_repeat_space[*curr_location] += np.minimum(0.1, 1 - self.obs_repeat_space[*curr_location])

        # rotate clockwise so absolute north faces up
        new_gridview = np.rot90(new_gridview, k=curr_direction)

        match curr_direction: # location of self in rotated new_gridview
            case Direction.RIGHT: rel_curr_location = (2,2)
            case Direction.DOWN: rel_curr_location = (2,2)
            case Direction.LEFT: rel_curr_location = (4,2)
            case Direction.UP: rel_curr_location = (2,4)

        # update tile by tile, column by column, in global POV
        for i in range(new_gridview.shape[0]):
            new_abs_x = curr_location[0] + i - rel_curr_location[0]
            if new_abs_x < 0 or new_abs_x >= self.size: continue
            for j in range(new_gridview.shape[1]):
                new_abs_y = curr_location[1] + j - rel_curr_location[1]
                if new_abs_y < 0 or new_abs_y >= self.size: continue

                # extract data
                unpacked = np.unpackbits(new_gridview[i, j])

                # update last seen and rewards
                tile_contents = np.packbits(np.concatenate((np.zeros(6, dtype=np.uint8), unpacked[-2:])))[0]
                if tile_contents != Tile.NO_VISION:
                    # store wall
                    # wall is given as relative to agent frame, where agent always faces right
                    # given as top left bottom right
                    wall_bits = list(unpacked[:4])
                    # rotate clockwise
                    for k in range(curr_direction): # direction 0-3 right down left up
                        wall_bits.append(wall_bits.pop(0))
                    self.obs_wall_top_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[0] * 255)
                    self.obs_wall_left_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[1] * 255)
                    self.obs_wall_bottom_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[2] * 255)
                    self.obs_wall_right_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[3] * 255)

                    self.obs_rewards_space[new_abs_x, new_abs_y] = (tile_contents - 1) * np.uint8(5*17)

                # update visible guards
                tile_guard_info = unpacked[4]
                if tile_guard_info == 1:
                    self.obs_guard_space[new_abs_x, new_abs_y] = np.uint8(32)
                else:
                    self.obs_guard_space[new_abs_x, new_abs_y] = np.uint8(0)

        # return observation in relative POV
        # rotate coordinates anticlockwise, such that absolute frame now has the agent's direction pointing up
        rot_x = curr_location[0]
        rot_y = curr_location[1]
        # always maintains top left bottom right
        wall_space_list = [self.obs_wall_top_space, self.obs_wall_left_space, self.obs_wall_bottom_space, self.obs_wall_right_space]
        for i in range(curr_direction+1):
            rot_x, rot_y = rot_y, self.size - 1 - rot_x
            wall_space_list.insert(0, wall_space_list.pop())
        # rotate clockwise until agent direction is up
        wall_space_list = [np.rot90(tmp_space_list, k=3-curr_direction) for tmp_space_list in wall_space_list]
        
        new_rewards_space = np.rot90(self.obs_rewards_space, k=3-curr_direction)
        new_guard_space = np.rot90(self.obs_guard_space, k=3-curr_direction)
        new_repeat_space = np.rot90(self.obs_repeat_space, k=3-curr_direction)

        # START 1D
        # each tile has 8 inputs
        # top, left, bottom, right walls (4)
        # recon, mission points (2)
        # seen before (1)
        # guard (normalised from new_guard_space) (1)
        output_obs = np.zeros((self.OBS_DIM_SIZE,), dtype=np.float32)
        output_ego_loc = ((self.VIEW_WINDOW_SIZE - 1) // 2, (self.VIEW_WINDOW_SIZE - 1) // 2)
        x_shift = output_ego_loc[0] - rot_x
        y_shift = output_ego_loc[1] - rot_y

        # input tile by tile, row by row, from ROTATED global truths
        output_obs_idx = 0
        for dst_y in range(self.VIEW_WINDOW_SIZE):
            ori_y = dst_y - y_shift
            if ori_y < 0 or ori_y >= self.size:
                output_obs_idx += self.BITS_PER_TILE * self.VIEW_WINDOW_SIZE # leave at 0 and skip to next row
                continue
            for dst_x in range(self.VIEW_WINDOW_SIZE):
                ori_x = dst_x - x_shift
                if ori_x < 0 or ori_x >= self.size:
                    output_obs_idx += self.BITS_PER_TILE # leave at 0 and skip to next tile
                    continue
                
                # top left bottom right walls
                for wall_idx in range(4):
                    output_obs[output_obs_idx] = wall_space_list[wall_idx][ori_x, ori_y]/255 # NORMALISE
                    output_obs_idx += 1
                
                # rewards and seen
                match new_rewards_space[ori_x, ori_y]:
                    case 255: # not seen
                        output_obs_idx += 3 # leave rewards, seen as zero
                    case 170: # mission point
                        output_obs_idx += 1 # skip recon point
                        output_obs[output_obs_idx] = 1 # mission point
                        output_obs_idx += 1
                        output_obs[output_obs_idx] = 1 # seen
                        output_obs_idx += 1
                    case 85:
                        output_obs[output_obs_idx] = 1 # recon point
                        output_obs_idx += 2 # skip mission point
                        output_obs[output_obs_idx] = 1 # seen
                        output_obs_idx += 1
                    case 0:
                        output_obs_idx += 2 # skip recon, mission point
                        output_obs[output_obs_idx] = 1 # seen
                        output_obs_idx += 1

                # within bounds
                output_obs[output_obs_idx] = 1
                output_obs_idx += 1
                
                # repeat tiles
                output_obs[output_obs_idx] = new_repeat_space[ori_x, ori_y]
                output_obs_idx += 1
                
                # guards
                output_obs[output_obs_idx] = self.GUARD_DISCRETE_LEVELS[new_guard_space[ori_x, ori_y]]
                output_obs_idx += 1
        
        # last 2 actions were turning
        if self.consecutive_turns >= 2:
            output_obs[output_obs_idx] = 1
        output_obs_idx += 1

        # location x
        output_obs[output_obs_idx] = rot_x / (self.size - 1)
        output_obs_idx += 1

        # location y
        output_obs[output_obs_idx] = rot_y / (self.size - 1)
        output_obs_idx += 1

        # current step number
        output_obs[output_obs_idx] = observation["step"]/100
        
        processed_obs = output_obs

        # Your inference code goes here.
        action, _states = self.model.predict(processed_obs, deterministic=True)
        
        return action
