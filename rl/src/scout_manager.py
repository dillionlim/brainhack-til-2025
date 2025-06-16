"""Manages the RL model."""
from enum import IntEnum
from importlib import import_module

import numpy as np

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

class ScoutManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.collector_model = getattr(import_module("ensemble_fellowship_v2_full.rl_manager"), "RLManager")()
        self.escaper_model = getattr(import_module("ensemble_runhidetell_v10.rl_manager"), "RLManager")()
        
        self.size = 16

        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_rewards_space = np.full((self.size, self.size), 255, dtype=np.uint8)

        self.obs_guard_space = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # Collector/Escaper switching
        self.turns_since_last_guard_sighting = float('inf')
        self.turns_threshold = 2
        
        self.action = 4

        self.prev_locs = []

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
            
        # Increase turn since last guard sighting
        self.turns_since_last_guard_sighting += 1

        guard_locs = []

        new_gridview = np.array(observation["viewcone"], dtype=np.uint8)
        curr_direction = np.array(observation["direction"], dtype=np.int64) # right down left up
        curr_location = np.array(observation["location"], dtype=np.int64)
        
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

                if (new_abs_x, new_abs_y,) in self.prev_locs:
                    self.prev_locs.remove((new_abs_x, new_abs_y,))

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

                    # we know the wall on the other side too
                    if new_abs_y + 1 < self.size: # top wall of the tile below
                        self.obs_wall_top_space[new_abs_x, new_abs_y+1] = np.uint8(wall_bits[2] * 255)
                    if new_abs_x + 1 < self.size: # left wall of the tile to the right
                        self.obs_wall_left_space[new_abs_x+1, new_abs_y] = np.uint8(wall_bits[3] * 255)
                    if new_abs_y - 1 >= 0: # bottom wall of the tile above
                        self.obs_wall_bottom_space[new_abs_x, new_abs_y-1] = np.uint8(wall_bits[0] * 255)
                    if new_abs_x - 1 >= 0: # right wall of the tile to the left
                        self.obs_wall_right_space[new_abs_x-1, new_abs_y] = np.uint8(wall_bits[1] * 255)

                # update visible guards
                tile_guard_info = unpacked[4]
                
                if tile_guard_info == 1:
                    self.obs_guard_space[new_abs_x, new_abs_y] += 1
                    if self.obs_guard_space[new_abs_x, new_abs_y] < 3:
                        self.turns_since_last_guard_sighting = 0
                    guard_locs.append((new_abs_x, new_abs_y))
                else:
                    self.obs_guard_space[new_abs_x, new_abs_y] = 0

        # Your inference code goes here.
        escaper_election_results = self.escaper_model.rl(observation, self.action)
        collector_election_results = self.collector_model.rl(observation, self.action)
        if self.turns_since_last_guard_sighting <= self.turns_threshold:
            # print("RUNNING ESCAPER...")
            election_results = escaper_election_results
        else:
            # print("RUNNING COLLECTOR...")
            election_results = collector_election_results

        god_action_mask = [True for i in range(4)]
        action_mask = [True for i in range(4)]
        match curr_direction: # don't go into walls
            case Direction.RIGHT:
                if self.obs_wall_right_space[*curr_location] > 0: action_mask[0] = False
                if self.obs_wall_left_space[*curr_location] > 0: action_mask[1] = False
            case Direction.DOWN:
                if self.obs_wall_bottom_space[*curr_location] > 0: action_mask[0] = False
                if self.obs_wall_top_space[*curr_location] > 0: action_mask[1] = False
            case Direction.LEFT:
                if self.obs_wall_left_space[*curr_location] > 0: action_mask[0] = False
                if self.obs_wall_right_space[*curr_location] > 0: action_mask[1] = False
            case Direction.UP:
                if self.obs_wall_top_space[*curr_location] > 0: action_mask[0] = False
                if self.obs_wall_bottom_space[*curr_location] > 0: action_mask[1] = False
        
        god_action_mask = [(god_action and action) for god_action, action in zip(god_action_mask, action_mask)]

        # action_mask = [True for i in range(4)]
        # x = curr_location[0]
        # y = curr_location[1]
        # match curr_direction: # don't go into guard
        #     case Direction.RIGHT:
        #         try:
        #             if self.obs_guard_space[x+1, y] == 32: action_mask[0] = False
        #         except:
        #             pass
        #         try:
        #             if self.obs_guard_space[x-1, y] == 32: action_mask[1] = False
        #         except:
        #             pass
        #     case Direction.DOWN:
        #         try:
        #             if self.obs_guard_space[x, y+1] == 32: action_mask[0] = False
        #         except:
        #             pass
        #         try:
        #             if self.obs_guard_space[x, y-1] == 32: action_mask[1] = False
        #         except:
        #             pass
        #     case Direction.LEFT:
        #         try:
        #             if self.obs_guard_space[x-1, y] == 32: action_mask[0] = False
        #         except:
        #             pass
        #         try:
        #             if self.obs_guard_space[x+1, y] == 32: action_mask[1] = False
        #         except:
        #             pass
        #     case Direction.UP:
        #         try:
        #             if self.obs_guard_space[x, y-1] == 32: action_mask[0] = False
        #         except:
        #             pass
        #         try:
        #             if self.obs_guard_space[x, y+1] == 32: action_mask[1] = False
        #         except:
        #             pass
    
        # tmp_god_action_mask = [(god_action and action) for god_action, action in zip(god_action_mask, action_mask)]
        # if sum(tmp_god_action_mask) > 0:
        #     god_action_mask = tmp_god_action_mask
        
        action_mask = [True for i in range(4)]
        x = curr_location[0]
        y = curr_location[1]
        verboten_tiles = []
        if len(guard_locs) == 1:
            for guard_loc in guard_locs:
                if self.obs_wall_bottom_space[*guard_loc] == 0 and guard_loc[1] + 1 < self.size and tuple(curr_location) != (guard_loc[0], guard_loc[1]+1):
                    verboten_tiles.append((guard_loc[0], guard_loc[1]+1))
                    if self.obs_wall_bottom_space[guard_loc[0], guard_loc[1]+1] == 0 and guard_loc[1] + 2 < self.size and tuple(curr_location) != (guard_loc[0], guard_loc[1]+2):
                        verboten_tiles.append((guard_loc[0], guard_loc[1]+2))

                if self.obs_wall_top_space[*guard_loc] == 0 and guard_loc[1] - 1 >= 0 and tuple(curr_location) != (guard_loc[0], guard_loc[1]-1):
                    verboten_tiles.append((guard_loc[0], guard_loc[1]-1))
                    if self.obs_wall_top_space[guard_loc[0], guard_loc[1]-1] == 0 and guard_loc[1] - 2 >= 0 and tuple(curr_location) != (guard_loc[0], guard_loc[1]-2):
                        verboten_tiles.append((guard_loc[0], guard_loc[1]-2))

                if self.obs_wall_right_space[*guard_loc] == 0 and guard_loc[0] + 1 < self.size and tuple(curr_location) != (guard_loc[0]+1, guard_loc[1]):
                    verboten_tiles.append((guard_loc[0]+1, guard_loc[1]))
                    if self.obs_wall_right_space[guard_loc[0]+1, guard_loc[1]] == 0 and guard_loc[0] + 2 < self.size and tuple(curr_location) != (guard_loc[0]+2, guard_loc[1]):
                        verboten_tiles.append((guard_loc[0]+2, guard_loc[1]))

                if self.obs_wall_left_space[*guard_loc] == 0 and guard_loc[0] - 1 >= 0 and tuple(curr_location) != (guard_loc[0]-1, guard_loc[1]):
                    verboten_tiles.append((guard_loc[0]-1, guard_loc[1]))
                    if self.obs_wall_left_space[guard_loc[0]-1, guard_loc[1]] == 0 and guard_loc[0] - 2 >= 0 and tuple(curr_location) != (guard_loc[0]-2, guard_loc[1]):
                        verboten_tiles.append((guard_loc[0]-2, guard_loc[1]))
        
        # for guard_loc in self.prev_locs:
        #     if self.obs_wall_bottom_space[*guard_loc] == 0 and guard_loc[1] + 1 < self.size:
        #         verboten_tiles.append((guard_loc[0], guard_loc[1]+1))
        #         if self.obs_wall_bottom_space[guard_loc[0], guard_loc[1]+1] == 0 and guard_loc[1] + 2 < self.size:
        #             verboten_tiles.append((guard_loc[0], guard_loc[1]+2))

        #     if self.obs_wall_top_space[*guard_loc] == 0 and guard_loc[1] - 1 >= 0:
        #         verboten_tiles.append((guard_loc[0], guard_loc[1]-1))
        #         if self.obs_wall_top_space[guard_loc[0], guard_loc[1]-1] == 0 and guard_loc[1] - 2 >= 0:
        #             verboten_tiles.append((guard_loc[0], guard_loc[1]-2))

        #     if self.obs_wall_right_space[*guard_loc] == 0 and guard_loc[0] + 1 < self.size:
        #         verboten_tiles.append((guard_loc[0]+1, guard_loc[1]))
        #         if self.obs_wall_right_space[guard_loc[0]+1, guard_loc[1]] == 0 and guard_loc[0] + 2 < self.size:
        #             verboten_tiles.append((guard_loc[0]+2, guard_loc[1]))

        #     if self.obs_wall_left_space[*guard_loc] == 0 and guard_loc[0] - 1 >= 0:
        #         verboten_tiles.append((guard_loc[0]-1, guard_loc[1]))
        #         if self.obs_wall_left_space[guard_loc[0]-1, guard_loc[1]] == 0 and guard_loc[0] - 2 >= 0:
        #             verboten_tiles.append((guard_loc[0]-2, guard_loc[1]))
        
        self.prev_locs = guard_locs
        
        match curr_direction: # don't go into guard + 2 moves
            case Direction.RIGHT:
                try:
                    if (x+1, y) in verboten_tiles: action_mask[0] = False
                except:
                    pass
                try:
                    if (x-1, y) in verboten_tiles: action_mask[1] = False
                except:
                    pass
            case Direction.DOWN:
                try:
                    if (x, y+1) in verboten_tiles: action_mask[0] = False
                except:
                    pass
                try:
                    if (x, y-1) in verboten_tiles: action_mask[1] = False
                except:
                    pass
            case Direction.LEFT:
                try:
                    if (x-1, y) in verboten_tiles: action_mask[0] = False
                except:
                    pass
                try:
                    if (x+1, y) in verboten_tiles: action_mask[1] = False
                except:
                    pass
            case Direction.UP:
                try:
                    if (x, y-1) in verboten_tiles: action_mask[0] = False
                except:
                    pass
                try:
                    if (x, y+1) in verboten_tiles: action_mask[1] = False
                except:
                    pass
            
        tmp_god_action_mask = [(god_action and action) for god_action, action in zip(god_action_mask, action_mask)]
        if sum(tmp_god_action_mask) > 0:
            god_action_mask = tmp_god_action_mask

        curr_max = None
        best_action = 4
        for action_id in range(len(election_results)):
            if god_action_mask[action_id] and (curr_max == None or election_results[action_id] > curr_max):
                curr_max = election_results[action_id]
                best_action = action_id
        # print(god_action_mask)
        # self.action = np.argmax(election_results)
        self.action = best_action
        
        return self.action
