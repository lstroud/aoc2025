"""PyTorch GPU strategy - distance computation on GPU.

Uses torch.cdist for GPU-accelerated pairwise distances,
then falls back to scipy for connected components (no good
GPU implementation without pulling in extra dependencies).

The real question: can GPU speedup overcome data transfer overhead
for only 1000 points?
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def solve(coords: np.ndarray, k: int = 1000, device: str = "cpu") -> int:
    """
    Solve Part 1 using PyTorch for distance computation.

    Args:
        coords: n×3 array of junction box coordinates
        k: Number of connection attempts
        device: PyTorch device ("cpu", "cuda", or "mps")

    Returns:
        Product of 3 largest circuit sizes
    """
    n = len(coords)
    torch_device = torch.device(device)

    # Move coordinates to GPU
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=torch_device)

    # Compute full distance matrix on GPU - O(n²) but parallel
    dist_matrix = torch.cdist(coords_tensor, coords_tensor, p=2)

    # Extract upper triangle (avoid self-distances and duplicates)
    # triu_indices gives us the condensed form indices
    triu_idx = torch.triu_indices(n, n, offset=1, device=torch_device)
    distances = dist_matrix[triu_idx[0], triu_idx[1]]

    # Find k-th smallest distance on GPU
    # kthvalue is O(n) on average, better than full sort
    threshold = torch.kthvalue(distances, k).values

    # Build adjacency mask on GPU
    adjacency_mask = (dist_matrix <= threshold) & (dist_matrix > 0)

    # Transfer to CPU for scipy connected_components
    # (GPU connected components would need cuGraph or custom kernel)
    rows, cols = torch.where(adjacency_mask)
    rows = rows.cpu().numpy()
    cols = cols.cpu().numpy()

    # Sync GPU before timing ends
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    # Build sparse matrix and find components
    data = np.ones(len(rows), dtype=np.int8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adjacency, directed=False)

    # Get component sizes
    sizes = np.bincount(labels)

    # Product of three largest
    largest_three = np.sort(sizes)[-3:]
    return int(np.prod(largest_three))
