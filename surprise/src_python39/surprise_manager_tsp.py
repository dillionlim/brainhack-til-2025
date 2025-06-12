import numpy as np
import cv2
from collections import defaultdict
from metrics import calculate_ssim_batch_vs_batch
from numba import jit
from typing import List, Tuple

class SurpriseManager:
    # SSIM constants
    K1: float = 0.01
    K2: float = 0.05
    L:  float = 255.0
    
    def __init__(self, threshold: int = 128,
                 α: float = 0.6, β: float = 0.3, γ: float = 0.1,
                 beam_width: int = 5):
        """
        :param threshold: pixel threshold for binarization
        :param α,β,γ: weights for (raw-SSIM, bin-SSIM, cross-corr)
        :param beam_width: how many partial sequences to keep
        """
        self.threshold  = threshold
        self.α, self.β, self.γ = α, β, γ
        self.beam_width = beam_width
        
        try:
            warmup_matrix = np.load("warmup_matrix.npy")
            _, _ = self.find_max_weight_hamiltonian_cycle(warmup_matrix, 15)
            print("JIT warmup completed successfully")
        except Exception as e:
            print("Error during JIT warmup: {e}")
    
    def threshold_image(self, img: np.ndarray) -> np.ndarray:
        _, binarized = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
        return binarized

    def ssim_1d(self, x: np.ndarray, y: np.ndarray) -> float:
        μx, μy = x.mean(), y.mean()
        σx2, σy2 = x.var(ddof=0), y.var(ddof=0)
        cov   = ((x-μx)*(y-μy)).mean()
        C1, C2 = (self.K1*self.L)**2, (self.K2*self.L)**2
        num    = (2*μx*μy + C1)*(2*cov + C2)
        den    = (μx*μx + μy*μy + C1)*(σx2 + σy2 + C2)
        return float(num/den) if den!=0 else 1.0

    def cross_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        """Normalized cross‐correlation of two 1D signals."""
        x0, y0 = x - x.mean(), y - y.mean()
        num  = (x0*y0).sum()
        den  = np.sqrt((x0*x0).sum() * (y0*y0).sum())
        return float(num/den) if den!=0 else 0.0
    
    def beam_search(self, raws: List[np.ndarray]) -> List[int]:
        # --- 1. Decode & prepare edges ---
        bins = [self.threshold_image(r) for r in raws]
        N    = len(bins)

        raw_left  = np.stack([r[:,   0] for r in raws])  # (N,H)
        raw_right = np.stack([r[:,  -1] for r in raws])
        bin_left  = np.stack([b[:,   0] for b in bins])
        bin_right = np.stack([b[:,  -1] for b in bins])

        # --- 2. Build composite similarity matrix ---
        sim = np.full((N,N), -np.inf, dtype=float)
        for i in range(N):
            for j in range(N):
                if i==j: 
                    continue
                s1 = self.ssim_1d(raw_right[i], raw_left[j])
                s2 = self.ssim_1d(bin_right[i], bin_left[j])
                c  = self.cross_corr(raw_right[i], raw_left[j])
                sim[i,j] = self.α*s1 + self.β*s2 + self.γ*c

        # --- 3. Choose left‐most start by bin‐whiteness ---
        start = int(np.argmax(bin_left.sum(axis=1)))

        # --- 4. Beam search for best path ---
        BeamItem = Tuple[float, List[int], int]  # (score, path, used_mask)
        beam: List[BeamItem] = [(0.0, [start], 1<<start)]

        for _ in range(1, N):
            candidates: List[BeamItem] = []
            for score, path, mask in beam:
                last = path[-1]
                for nxt in range(N):
                    if mask & (1<<nxt): 
                        continue
                    new_score = score + sim[last, nxt]
                    candidates.append((new_score, path + [nxt], mask | (1<<nxt)))
            # keep top‐K by score
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:self.beam_width]

        # any of the final beam items has length N; pick the highest score
        best = max(beam, key=lambda x: x[0])
        return best[1]
    
    def surprise(self, slices: list[bytes]) -> list[int]:
        """
        Reconstructs shredded document from vertical slices.

        Args:
            slices: list of byte arrays, each representing a JPEG-encoded vertical slice of the input document

        Returns:
            Predicted permutation of input slices to correctly reassemble the document.
        """

        np_imgs = [cv2.imdecode(np.frombuffer(slice_, np.uint8), cv2.IMREAD_GRAYSCALE) for slice_ in slices]
        num_slices = len(np_imgs)
        
        if num_slices >= 20:
            return self.beam_search(np_imgs)
        if num_slices == 1:
            return [0]
        
        np_imgs_thr = np.array(np_imgs) # no thresholding
        
        left_edges = np_imgs_thr[:, :, 0]
        right_edges = np_imgs_thr[:, :, -1]
        
        # Add white strip to be starting right edge
        white_strip = np.ones((1, left_edges.shape[1])) * 255
        left_edges  = np.concat([left_edges, white_strip])
        right_edges = np.concat([right_edges, white_strip])
        
        ssim_matrix = calculate_ssim_batch_vs_batch(right_edges, left_edges)
        
        max_total_sum, path = self.find_max_weight_hamiltonian_cycle(ssim_matrix, start_node=np_imgs_thr.shape[0])
        
        return path[1:]
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def find_max_weight_hamiltonian_path(weights: np.ndarray, start_node: int = None):
        """
        Finds the Hamiltonian path with the maximum sum of weights in a complete
        weighted directed graph using dynamic programming with bitmasking.

        Args:
            weights (np.ndarray): An N x N NumPy array where weights[i][j]
                                  represents the weight of the directed edge from
                                  node i to node j.
            start_node (int, optional): The index of the node where the path must start.
                                        If None, the algorithm finds the maximum path
                                        starting from any possible node. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - float: The maximum sum of weights found.
                - list: The sequence of nodes representing the path.
                        Returns an empty list if N is 0, or [0] if N is 1.
        """
        n = weights.shape[0]

        if n == 0:
            return 0, None
        if n == 1:
            return 0, [0]

        # Validate start_node if provided
        if start_node is not None and (start_node < 0 or start_node >= n):
            raise ValueError(f"start_node must be between 0 and {n-1}, but got {start_node}")

        dp = np.full((1 << n, n), -np.inf)
        path_tracker = np.full((1 << n, n), -1, dtype=np.int64)

        # Base cases: path with a single node
        if start_node is None:
            # If no specific start_node, any node can be a starting point
            for i in range(n):
                dp[1 << i][i] = 0
        else:
            # If a specific start_node is given, only that node can start the path
            dp[1 << start_node][start_node] = 0

        # Iterate through all masks
        for mask in range(1, 1 << n):
            # Iterate through all possible last nodes (j)
            for j in range(n):
                if (mask >> j) & 1:
                    prev_mask = mask ^ (1 << j)

                    # Iterate through all possible previous nodes (i)
                    for i in range(n):
                        if (prev_mask >> i) & 1:
                            if dp[prev_mask][i] != -np.inf:
                                current_path_sum = dp[prev_mask][i] + weights[i, j]

                                if current_path_sum > dp[mask][j]:
                                    dp[mask][j] = current_path_sum
                                    path_tracker[mask][j] = i

        # Find the maximum total sum and its last node
        max_total_sum = -np.inf
        last_node_in_max_path = -1
        final_mask = (1 << n) - 1

        for j in range(n):
            # Ensure the path found is a Hamiltonian path (visits all nodes)
            if dp[final_mask][j] > max_total_sum:
                max_total_sum = dp[final_mask][j]
                last_node_in_max_path = j

        # Reconstruct the path
        path = []
        if last_node_in_max_path != -1:
            current_mask = final_mask
            current_node = last_node_in_max_path

            while current_mask > 0:
                path.append(current_node)
                prev_node = path_tracker[current_mask][current_node]
                current_mask ^= (1 << current_node)
                current_node = prev_node

            path.reverse()

        return max_total_sum, path

    @staticmethod
    @jit(nopython=True, cache=True)
    def find_max_weight_hamiltonian_cycle(weights: np.ndarray, start_node: int):
        """
        Finds the Hamiltonian cycle with the maximum sum of weights in a complete
        weighted directed graph using dynamic programming with bitmasking.
        The cycle must start and end at the specified start_node, visiting all
        other nodes exactly once.

        Args:
            weights (np.ndarray): An N x N NumPy array where weights[i][j]
                                  represents the weight of the directed edge from
                                  node i to node j.
            start_node (int): The index of the node where the cycle must start and end.

        Returns:
            tuple: A tuple containing:
                - float: The maximum sum of weights found.
                - list: The sequence of nodes representing the cycle.
                        Returns -np.inf and an empty list if no valid cycle is found
                        (e.g., n < 3 or no path exists).
        """
        n = weights.shape[0]

        # A Hamiltonian cycle requires at least 3 nodes
        if n < 3:
            return -np.inf, None

        # Validate start_node
        if start_node < 0 or start_node >= n:
            raise ValueError(f"start_node must be between 0 and {n-1}, but got {start_node}")

        dp = np.full((1 << n, n), -np.inf, dtype=np.float64)
        path_tracker = np.full((1 << n, n), -1, dtype=np.int64)

        # Base case for the cycle: start at the specified node with sum 0
        dp[1 << start_node][start_node] = 0.0

        # Iterate through all masks
        for mask in range(1, 1 << n):
            # Iterate through all possible last nodes (j)
            for j in range(n):
                # Ensure j is part of the current mask and not the start_node for intermediate steps
                if (mask >> j) & 1:
                    prev_mask = mask ^ (1 << j)

                    # Iterate through all possible previous nodes (i)
                    for i in range(n):
                        if (prev_mask >> i) & 1:
                            # Exclude cases where i is the start_node and mask is not just start_node and j
                            if i == start_node and mask != (1 << start_node | 1 << j):
                                continue # This ensures start_node is only the first node in path segment

                            if dp[prev_mask][i] != -np.inf:
                                current_path_sum = dp[prev_mask][i] + weights[i, j]

                                if current_path_sum > dp[mask][j]:
                                    dp[mask][j] = current_path_sum
                                    path_tracker[mask, j] = i

        # Find the maximum total sum for a cycle
        max_cycle_sum = -np.inf
        best_last_node_before_return = -1
        final_mask = (1 << n) - 1

        # Check all possible last nodes (j) before returning to start_node
        for j in range(n):
            if j == start_node: # Cannot be the last node before returning to itself
                continue

            # Ensure a valid path to j exists, and an edge from j back to start_node exists
            if dp[final_mask][j] != -np.inf:
                current_cycle_sum = dp[final_mask][j] + weights[j, start_node]
                if current_cycle_sum > max_cycle_sum:
                    max_cycle_sum = current_cycle_sum
                    best_last_node_before_return = j

        # Reconstruct the cycle path
        cycle_path = []
        if best_last_node_before_return != -1:
            current_mask = final_mask
            current_node = best_last_node_before_return

            # Backtrack from the node before the start_node
            while current_mask > 0:
                cycle_path.append(current_node)
                prev_node = path_tracker[current_mask, current_node]
                current_mask ^= (1 << current_node)
                current_node = prev_node

            cycle_path.reverse()

        return max_cycle_sum, cycle_path