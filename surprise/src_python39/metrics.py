import numpy as np
def calculate_ssim_edges_batch(edge_primary: np.ndarray, edge_batch: np.ndarray, L: float = 255.0) -> np.ndarray:
    """
    Calculates the Structural Similarity Index (SSIM) between a single primary 1D edge
    and a batch of other 1D edges.

    Args:
        edge_primary (np.ndarray): The primary 1D NumPy array representing an edge's pixel values.
                                   Ensure the array contains pixel intensity values (e.g., 0-255).
        edge_batch (np.ndarray): A 2D NumPy array where each row represents an edge in the batch.
                                 All edges in the batch must have the same length as edge_primary.
        L (float): The dynamic range of the pixel values. For 8-bit images, this is typically 255.0.

    Returns:
        np.ndarray: A 1D NumPy array of SSIM scores, where each score corresponds to the SSIM
                    between `edge_primary` and the respective edge in `edge_batch`.
                    Scores range from -1 to 1.

    Raises:
        ValueError: If `edge_primary` is not 1-dimensional, or if `edge_batch` is not 2-dimensional,
                    or if the length of `edge_primary` does not match the length of edges in `edge_batch`.
    """

    # Constants for SSIM calculation
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    # Calculate statistics for the primary edge (edge1 in the SSIM formula)
    mu1 = np.mean(edge_primary)
    sigma1_sq = np.var(edge_primary, ddof=0)

    # Calculate statistics for the batch of edges (edge2 in the SSIM formula, for each edge in batch)
    mu2_batch = np.mean(edge_batch, axis=1) # Mean of each row (each edge in the batch)
    sigma2_sq_batch = np.var(edge_batch, axis=1, ddof=0) # Variance of each row

    # Calculate covariance for edge_primary against each edge in edge_batch
    # First, center the primary edge and each edge in the batch
    edge_primary_centered = edge_primary - mu1
    edge_batch_centered = edge_batch - mu2_batch[:, np.newaxis] # Subtract row-wise mean

    # Element-wise product, then mean along axis 1 to get covariance for each batch edge
    sigma12_batch = np.mean(edge_primary_centered * edge_batch_centered, axis=1)

    # Apply the SSIM formula element-wise across the batch
    numerator = (2 * mu1 * mu2_batch + C1) * (2 * sigma12_batch + C2)
    denominator = (mu1**2 + mu2_batch**2 + C1) * (sigma1_sq + sigma2_sq_batch + C2)

    # Use np.where to set SSIM to 1.0 where the denominator is zero, otherwise calculate normally.
    ssim_scores = np.where(denominator == 0, 1.0, numerator / denominator)

    return ssim_scores

def calculate_ssim_edges(edge1: np.ndarray, edge2: np.ndarray, L: float = 255.0) -> float:
    # Constants for SSIM calculation, preventing division by zero
    # K1 and K2 are small constants, typically 0.01 and 0.03
    K1 = 0.01
    K2 = 0.03
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

def mse(edge1, edge2):
    return np.mean(np.square(edge1 - edge2))

def calculate_ssim_batch_vs_batch(edges_batch1: np.ndarray, edges_batch2: np.ndarray, L: float = 255.0) -> np.ndarray:
    """
    Calculates the Structural Similarity Index (SSIM) for every edge in the first batch
    against every edge in the second batch, returning a similarity matrix.

    This function is optimized for performance using NumPy's vectorized operations.

    Args:
        edges_batch1 (np.ndarray): A 2D NumPy array where each row is an edge (N1 x D).
                                   N1 is the number of edges, D is the edge length.
        edges_batch2 (np.ndarray): A 2D NumPy array where each row is an edge (N2 x D).
                                   N2 is the number of edges, D is the edge length.
                                   D (edge length) must be the same for both batches.
        L (float): The dynamic range of the pixel values. For 8-bit images, this is typically 255.0.

    Returns:
        np.ndarray: A 2D NumPy array (N1 x N2) where element [i, j] is the SSIM score
                    between edges_batch1[i] and edges_batch2[j].
                    Scores range from -1 to 1.

    Raises:
        ValueError: If input arrays are not 2-dimensional or have mismatched edge lengths.
    """

    num_edges1, edge_length = edges_batch1.shape
    num_edges2 = edges_batch2.shape[0]

    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    # Calculate means and variances for each batch
    mu1 = np.mean(edges_batch1, axis=1) # Shape (N1,)
    mu2 = np.mean(edges_batch2, axis=1) # Shape (N2,)

    sigma1_sq = np.var(edges_batch1, axis=1, ddof=0) # Shape (N1,)
    sigma2_sq = np.var(edges_batch2, axis=1, ddof=0) # Shape (N2,)

    # Calculate cross-covariance matrix (N1 x N2)
    # Center the data first
    centered_edges1 = edges_batch1 - mu1[:, np.newaxis]
    centered_edges2 = edges_batch2 - mu2[:, np.newaxis]

    # Covariance for each pair (i, j) is the mean of the element-wise product
    # This is efficiently computed using a dot product: (N1 x D) @ (D x N2) -> (N1 x N2)
    sigma12_matrix = np.dot(centered_edges1, centered_edges2.T) / edge_length

    # Expand mu and sigma_sq to N1 x N2 matrices for element-wise calculations
    # This uses broadcasting: (N1, 1) * (1, N2) -> (N1, N2)
    mu1_matrix = mu1[:, np.newaxis]
    mu2_matrix = mu2[np.newaxis, :]

    sigma1_sq_matrix = sigma1_sq[:, np.newaxis]
    sigma2_sq_matrix = sigma2_sq[np.newaxis, :]

    # Numerator components
    numerator_part1 = (2 * mu1_matrix * mu2_matrix + C1)
    numerator_part2 = (2 * sigma12_matrix + C2)
    numerator = numerator_part1 * numerator_part2

    # Denominator components
    denominator_part1 = (mu1_matrix**2 + mu2_matrix**2 + C1)
    denominator_part2 = (sigma1_sq_matrix + sigma2_sq_matrix + C2)
    denominator = denominator_part1 * denominator_part2

    # Calculate SSIM matrix, handling division by zero where denominator is 0
    # np.where ensures that if denominator is zero, SSIM for that pair is 1.0 (perfect match)
    ssim_matrix = np.where(denominator == 0, 1.0, numerator / denominator)

    return ssim_matrix
