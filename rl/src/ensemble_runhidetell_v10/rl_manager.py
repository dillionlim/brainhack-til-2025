from importlib import import_module

import numpy as np

class RLManager:
    def __init__(self):
        voter_register = [
            "chitose-guard2-8M",
            "chitose-guard-2M",
            "chitose-guard-8M",
            "cologne-guard-8M",
        ]
        weights = [0.3148798209765864, 0.22951525587785473, 0.41574585272483505, 0.03985907042072377]
        assert len(voter_register) == len(weights)

        i = 0
        while i < len(weights):
            if weights[i] == 0:
                voter_register.pop(i)
                weights.pop(i)
            else:
                i += 1

        self.voters = [getattr(import_module(f"ensemble_scouts.{voter_name}.rl_manager"), "RLManager")() for voter_name in voter_register]
        self.weights = weights

        self.action = 4

    def rl(self, observation: dict[str, int | list[int]], prev_action) -> int:
        self.action = prev_action
        ballot_box = [weight * voter.rl(observation, self.action) for voter, weight in zip(self.voters, self.weights)]
        
        election_results = np.zeros((4,))
        for vote in ballot_box:
            assert len(vote) == 4
            election_results += vote
        
        # self.action = np.argmax(election_results)

        return election_results
    