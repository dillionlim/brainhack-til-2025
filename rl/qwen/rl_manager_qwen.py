import enum
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LogitsProcessor
import random
import torch

class States(enum.Enum):
    UNKNOWN = 0
    EMPTY = 1
    RECON = 2
    MISSION = 3
    WALL = 4

class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    A processor where after a maximum number of tokens are generated,
    a </think> token is added at the end to stop the thinking generation,
    and then it will continue to generate the response.
    """
    def __init__(self, tokenizer, max_thinking_tokens=None):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.nl_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = float('-inf')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.tokens_generated += 1
        if self.max_thinking_tokens == 0 and not self.stopped_thinking and self.tokens_generated > 0:
            scores[:] = self.neg_inf
            scores[0][self.nl_token] = 0
            scores[0][self.think_end_token] = 0
            self.stopped_thinking = True
            return scores

        if self.max_thinking_tokens is not None and not self.stopped_thinking:
            if (self.tokens_generated / self.max_thinking_tokens) > .95:
                scores[0][self.nl_token] = scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.max_thinking_tokens))
                scores[0][self.think_end_token] = (
                    scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.max_thinking_tokens))
                )

            if self.tokens_generated >= (self.max_thinking_tokens - 1):
                if self.tokens_generated == self.max_thinking_tokens-1:
                    scores[:] = self.neg_inf
                    scores[0][self.nl_token] = 0
                else:
                    scores[:] = self.neg_inf
                    scores[0][self.think_end_token] = 0
                    self.stopped_thinking = True

        return scores

class RLManager:
    def __init__(self, map_size = 16 * 2 + 1):
        self.map_size = map_size
        self.global_map = [[States.UNKNOWN for _ in range(map_size)] for _ in range(map_size)]
        model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = ThinkingTokenBudgetProcessor(self.tokenizer, max_thinking_tokens=4096)
        self.pipe = pipeline(
            "text-generation", model=model_name, max_new_tokens=4200, logits_processor=[self.processor], device=0
        )

        for i in range(map_size):
            self.global_map[0][i] = States.WALL                   # Top row
            self.global_map[map_size - 1][i] = States.WALL        # Bottom row
            self.global_map[i][0] = States.WALL                   # Left column
            self.global_map[i][map_size - 1] = States.WALL        # Right column
        
        for i in range(0, map_size, 2):
            for j in range(0, map_size, 2):
                self.global_map[i][j] = States.WALL

    def parse_viewcone_tile(self, viewcone_tile: int, dir: int) -> dict:
        def rotate_right(arr, x):
            x = x % len(arr)  # handle x > len(arr)
            return arr[-x:] + arr[:-x]

        last2 = viewcone_tile & 0b11
        state = States.UNKNOWN
        if last2 == 0b00:
            state = States.UNKNOWN
        elif last2 == 0b01:
            state = States.EMPTY
        elif last2 == 0b10:
            state = States.RECON
        elif last2 == 0b11:
            state = States.MISSION
        # these are wall directions relative to agent's direction
        # so we need to offset them by the agent's direction
        #           right                          bottom                   left                        top
        walls = [viewcone_tile & 0b10000, viewcone_tile & 0b100000, viewcone_tile & 0b1000000, viewcone_tile & 0b10000000]
        no_rotations = dir % 4
        walls = rotate_right(walls, no_rotations)
        walls = [bool(i) for i in walls]

        return {
            "state": state,
            "scout": True if (viewcone_tile & 0b100) else False,
            "guard": True if (viewcone_tile & 0b1000) else False,
            "walls": walls,
        }
    
    def update_global_map(self, new_tile, x, y):
        walls = new_tile["walls"]
        if walls[0]: # right wall
            self.global_map[x + 1][y] = States.WALL 
        elif self.global_map[x + 1][y] == States.UNKNOWN:
            self.global_map[x + 1][y] = States.EMPTY 
        if walls[1]: # down wall
            self.global_map[x][y + 1] = States.WALL 
        elif self.global_map[x][y + 1] == States.UNKNOWN:
            self.global_map[x][y + 1] = States.EMPTY 
        if walls[2]: # left wall
            self.global_map[x - 1][y] = States.WALL 
        elif self.global_map[x - 1][y] == States.UNKNOWN:
            self.global_map[x - 1][y] = States.EMPTY 
        if walls[3]: # up wall
            self.global_map[x][y - 1] = States.WALL 
        elif self.global_map[x][y - 1] == States.UNKNOWN:
            self.global_map[x][y - 1] = States.EMPTY 
        self.global_map[x][y] = new_tile["state"]

    def _viewcone_to_map_coords(self, vc_row, vc_col, agent_location, agent_direction):
        """
        Translates viewcone coordinates relative to the agent into global map coordinates.

        Args:
            vc_row (int): Row index in the 7x5 viewcone (0-6).
            vc_col (int): Column index in the 7x5 viewcone (0-4).
            agent_location (list): Agent's global location [x, y].
            agent_direction (int): Agent's direction (0:E, 1:S, 2:W, 3:N).

        Returns:
            tuple: Global map coordinates (map_x, map_y) or None if outside map bounds.
        """
        # Agent is at viewcone[2, 2]
        relative_row = vc_row - 2
        relative_col = vc_col - 2

        if agent_direction == 3: # North -> (0,0) is bottom left relative to agent
            delta_x = relative_col
            delta_y = -relative_row
        elif agent_direction == 0: # East -> (0,0) is top left relative to agent
            delta_x = relative_row
            delta_y = relative_col
        elif agent_direction == 1: # South -> (0,0) is top right relative to agent
            delta_x = -relative_col
            delta_y = relative_row
        elif agent_direction == 2: # West -> (0,0) is bottom right relative to agent
            delta_x = -relative_row
            delta_y = -relative_col
        else:
            raise ValueError(f"Invalid agent direction: {agent_direction}")

        # Calculate global map coordinates
        map_x = agent_location[0] + delta_x
        map_y = agent_location[1] + delta_y

        # Check if coordinates are within map bounds
        if 0 <= map_x < 7 and 0 <= map_y < 5:
            return (map_x, map_y)
        else:
            return None # Outside map bounds
    
    def print_map(self, scout, agents=[]):
        for i in range(self.map_size):
            for j in range(self.map_size):
                if scout == (i, j):
                    print("S", end="")
                    continue
                elif (i, j) in agents:
                    print("A", end="")
                    continue
                cell = self.global_map[i][j]
                if cell == States.EMPTY:
                    print(" ", end="")
                elif cell == States.RECON:
                    print("R", end="")
                elif cell == States.MISSION:
                    print("M", end="")
                elif cell == States.WALL:
                    print("#", end="")
                else:
                    print("?", end="")
            print()
    
    def serialize_map(self, scout, agents=[]):
        lines = []
        for i in range(self.map_size):
            line = ""
            for j in range(self.map_size):
                if scout == (i, j):
                    line += "S"
                elif (i, j) in agents:
                    line += "A"
                else:
                    cell = self.global_map[i][j]
                    if cell == States.EMPTY:
                        line += " "
                    elif cell == States.RECON:
                        line += "R"
                    elif cell == States.MISSION:
                        line += "M"
                    elif cell == States.WALL:
                        line += "#"
                    else:
                        line += "?"
            lines.append(line)
        return "\n".join(lines)


    def chat(self, prompt: str) -> int:
        messages = [
            {"role": "system", "content": """You are a master strategist. You are placed in a 16 x 16 map, with the following annotations:
- " " for EMPTY
- "R" for RECON (Worth 1 point)
- "M" for MISSION (Worth 5 points)
- "#" for WALL, which you cannot walk into
- "S" for Scout, which is your current location
- "A" for Guard, which you want to avoid walking into. Getting caught by a guard will result in a harsh penalty of 50 points.
- "?" for UNKNOWN, which are unexplored cells.

A cell is represented by a 3 x 3 grid:
#Y#
YXY
#Y#

Y could either be a wall, or an empty passageway. 

You can only move move forward, backwards, turn left or right, or stay in your current position. Your aim is to get as many points as possible.
             
Reason your answer by considering the 5 possible moves only. Limit your thinking to 500 words.

Do not include your reasoning in the final answer. Only include the final answer as an integer from 0 to 4.

0: Move forward
1: Move backward
2: Turn left
3: Turn right
4: Stay          
"""},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )

        try:
            answer = self.pipe(messages)[0]["generated_text"][-1]["content"]
            delimiter = "</think>"
            if delimiter in answer:
                content = answer.split(delimiter, 1)[1].strip()
            else:
                content = answer  # fallback if </think> is not found

            print(content)
            return int(content)
        except:
            return random.randint(0, 4)

    def query_llm_scout(self, serialized_map, agent_info):
        prompt = f"""
Current Global Map:
{serialized_map}

Agent Info:
- Position: {agent_info['location'][0] * 2 + 1, agent_info['location'][1] * 2 + 1}
- Direction: {agent_info['direction']} (0 = Facing right, 1 = Facing down, 2 = Facing left, 3 = Facing up)
- Step: {agent_info['step']}

What is the best next move for the agent? Consider future possibilities in deciding. Return only an integer from 0 to 4, representing the direction.

0: Move forward
1: Move backward
2: Turn left
3: Turn right
4: Stay
        """
        # print(prompt)
        return self.chat(prompt)

    def rl(self, observation_json: dict) -> int:
        instance = observation_json

        viewcone = instance["viewcone"]
        direction = instance["direction"]
        location = instance["location"]
        scout = instance["scout"]
        step = instance["step"]

        viewcone_info = [
            [self.parse_viewcone_tile(int(tile), int(direction)) for tile in row]
            for row in viewcone
        ]
        scout = None
        guards = []
        for vc_row in range(7):
            for vc_col in range(5):
                res = self._viewcone_to_map_coords(vc_row, vc_col, location, direction)
                if res:
                    gx, gy = res
                    # Each tile is centered in a 2x scaled grid
                    self.update_global_map(viewcone_info[vc_row][vc_col], gx * 2 + 1, gy * 2 + 1)
                if viewcone_info[vc_row][vc_col]["scout"]:
                    scout = (gx * 2 + 1, gy * 2 + 1)
                elif viewcone_info[vc_row][vc_col]["guard"]:
                    guards.append((gx * 2 + 1, gy * 2 + 1))

        print(self.serialize_map(scout, guards))

        serialized = self.serialize_map(scout, guards)
        return self.query_llm_scout(serialized, {
            "location": location,
            "direction": direction,
            "step": step,
            "scout": scout
        })
