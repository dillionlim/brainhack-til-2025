import heapq
from typing import List, Tuple, Dict

import numpy as np
"""
32x16 grid
[:16,:] represents facing up/down (moving vertically)
[16:,:] represents facing left/right (moving horizontally)
"""
def create_node(position: Tuple[int, int], g: float = float('inf'), 
                h: float = 0.0, parent: Dict = None) -> Dict:
    """
    Create a node for the A* algorithm.
    
    Args:
        position: (x, y) coordinates of the node
        g: Cost from start to this node (default: infinity)
        h: Estimated cost from this node to goal (default: 0)
        parent: Parent node (default: None)
    
    Returns:
        Dictionary containing node information
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }
def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate the estimated distance between two points using Euclidean distance.
    Using Manhattan instead, and takes into account horizontal/vertical spaces.
    """
    diff_dims = 0
    x1, y1 = pos1
    x2, y2 = pos2

    if (x1 > 15 and x2 <= 15) or (x2 > 15 and x1 <= 15): diff_dims = 1

    x1 = x1 - 15 if x1 > 15 else x1
    x2 = x2 - 15 if x2 > 15 else x2
    # return sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return abs(x2-x1) + abs(y2-y1) + diff_dims
def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get all valid neighboring positions in the grid.
    
    Args:
        grid: 3D numpy array of shape (32,16,4)
        position: Current position (x, y)
    
    Returns:
        List of valid neighboring positions
    """
    x, y = position
    rows, cols = grid.shape[:2]

    # walls top left bottom right
    fourwalls = grid[x,y]
    
    # All possible moves (including diagonals)
    possible_moves = []
    # switching dimension
    if x < 16: # vertical
        possible_moves.append((x+16,y)) # switch dims
        if fourwalls[0] == 0: possible_moves.append((x,y-1)) # up
        if fourwalls[2] == 0: possible_moves.append((x,y+1)) # down
    else: # horizontal
        possible_moves.append((x-16,y)) # switch dims
        if fourwalls[1] == 0: possible_moves.append((x-1,y)) # left
        if fourwalls[3] == 0: possible_moves.append((x+1,y)) # right


    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols  # Within grid bounds
        # and grid[nx, ny] == 0                # Not an obstacle
    ]
def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = []
    current = goal_node
    
    while current is not None:
        path.append(current['position'])
        current = current['parent']
        
    return path[::-1]  # Reverse to get path from start to goal
def find_path(grid: np.ndarray, start: Tuple[int, int], 
              goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find the optimal path using A* algorithm.
    
    Args:
        grid: 3D numpy array of shape (32,16,4)
        start: Starting position (x, y)
        goal: Goal position (x, y)
    
    Returns:
        List of positions representing the optimal path
    """
    # Initialize start node
    start_node = create_node(
        position=start,
        g=0,
        h=calculate_heuristic(start, goal)
    )
    
    # Initialize open and closed sets
    open_list = [(start_node['f'], start)]  # Priority queue
    open_dict = {start: start_node}         # For quick node lookup
    closed_set = set()                      # Explored nodes
    
    while open_list:
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]
        
        # Check if we've reached the goal
        if current_pos == goal or (current_pos[0] - 16, current_pos[1]) == goal:
            return reconstruct_path(current_node)
            
        closed_set.add(current_pos)
        
        # Explore neighbors
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            # Skip if already explored
            if neighbor_pos in closed_set:
                continue
                
            # Calculate new path cost
            # tentative_g = current_node['g'] + calculate_heuristic(current_pos, neighbor_pos)
            tentative_g = current_node['g'] + 1
            
            # Create or update neighbor
            if neighbor_pos not in open_dict:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                # Found a better path to the neighbor
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node
    
    return []  # No path found