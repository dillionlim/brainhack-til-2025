import numpy as np
import cv2
# from numpy.linalg import norm

class SurpriseManager:
    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        pass
    
    @staticmethod
    def threshold_np(arr, threshold=128):
        arrcopy = arr.copy()
        arrcopy[arr >= threshold] = 255
        arrcopy[arr < threshold] = 0
        return arrcopy
    
    @staticmethod
    def mse(A, B):
        return np.mean(np.square(A - B))
    
    @staticmethod
    def calculate_ssim_edges(edge1: np.ndarray, edge2: np.ndarray, L: float = 255.0) -> float:
        # Constants for SSIM calculation, preventing division by zero
        # K1 and K2 are small constants, typically 0.01 and 0.03
        K1 = 0.01
        K2 = 0.05
        C1 = (K1 * L)**2
        C2 = (K2 * L)**2

        # Calculate the mean of each edge array
        mu1 = np.mean(edge1)
        mu2 = np.mean(edge2)

        # Calculate the variance of each edge array (population variance, ddof=0 for consistency with SSIM formula)
        sigma1_sq = np.var(edge1, ddof=0)
        sigma2_sq = np.var(edge2, ddof=0)

        # Calculate the covariance between the two edge arrays (population covariance)
        sigma12 = np.mean((edge1 - mu1) * (edge2 - mu2))

        # Apply the SSIM formula
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

        # Handle the case where the denominator might be zero (e.g., if both edges are perfectly uniform and identical)
        # In such cases, SSIM is considered 1.0 (perfect similarity).
        if denominator == 0:
            return 1.0

        ssim_score = numerator / denominator
        return ssim_score
    
    def surprise(self, slices: list[bytes]) -> list[int]:
        """
        Reconstructs shredded document from vertical slices.

        Args:
            slices: list of byte arrays, each representing a JPEG-encoded vertical slice of the input document

        Returns:
            Predicted permutation of input slices to correctly reassemble the document.
        """

        np_imgs = [cv2.imdecode(np.frombuffer(slice_, np.uint8), cv2.IMREAD_GRAYSCALE) for slice_ in slices]
        np_imgs_thr = [self.threshold_np(arr) for arr in np_imgs]
        
        result = []
        chosen = np.zeros((len(np_imgs_thr)))

        max_leftsum = -np.inf
        max_rightsum = -np.inf
        for i, slice_ in enumerate(np_imgs_thr):
            left_edge = slice_[:, 0]
            right_edge = slice_[:, -1]

            leftsum = left_edge.sum()
            if leftsum > max_leftsum:
                max_leftsum = leftsum
                left_index = i
            rightsum = right_edge.sum()
            if rightsum > max_rightsum:
                max_rightsum = rightsum
                right_index = i

        # assert left_index != right_index
        result.append(left_index)
        chosen[left_index] = 1
        # chosen[right_index] = 1
        
        for i in range(1, len(np_imgs_thr)):
            prev_slice = np_imgs_thr[result[-1]]
            prev_slice_right = prev_slice[:, -1]

            max_ssim = -np.inf
            # min_mse = np.inf
            best_match = None
            for i, slice_ in enumerate(np_imgs_thr):
                if chosen[i]: continue
                left_edge = slice_[:, 0]
                ssim = self.calculate_ssim_edges(left_edge, prev_slice_right)
                # mse_ = mse(left_edge, prev_slice_right)
                if ssim > max_ssim:
                    best_match = i
                    max_ssim = ssim
                # if mse_ < min_mse:
                #     best_match = i
                #     min_mse = mse_
            assert best_match is not None
            result.append(best_match)
            chosen[best_match] = 1
        # if left_index != right_index:
        #     result.append(right_index)
        return result
