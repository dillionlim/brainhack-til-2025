from importlib import import_module

import numpy as np

class RLManager:
    def __init__(self):
        voter_register = [
            "atlanta-8M-deadend-8M",
            "avignon-8M",
            "avignon-ariane4-8M",
            "avignon-ariane20-8M",
            "avignon-ariane24-4M",
            "avignon-ariane24-normal-2M4",
            "avignon-ariane25-4M",
            "avignon-ariane25-normal-2M8",
            "caracas-8M",
            "caracas-ariane25-4M",
            "chitose-6M",
            "chitose-8M",
        ]
        weights = [0.01927453437238683, 0.08712583043616347, 0.10645170543735268, 0.07834380025969802, 0.025097352057912217, 0.2861685264023054, 0.11636224518181616, 0.2324002282631162, 0.03190663490402947, 0.009631867651736917, 0.000906384139520319, 0.006330890893962332, ]
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
    