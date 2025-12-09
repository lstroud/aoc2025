"""Sparse graph strategy using scipy.sparse.csgraph.

Build a sparse adjacency matrix from the k closest pairs,
then use connected_components - the tool literally designed
for this exact problem.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform


def solve(coords: np.ndarray, k: int = 1000) -> int:
    """
    Solve Part 1 using sparse graph connected components.

    Args:
        coords: n×3 array of junction box coordinates
        k: Number of connection attempts

    Returns:
        Product of 3 largest circuit sizes
    """
    n = len(coords)

    # Compute all pairwise distances
    distances = pdist(coords, metric='euclidean')

    # Find threshold for k-th smallest distance
    threshold = np.partition(distances, k - 1)[k - 1]

    # Build sparse adjacency matrix for edges within threshold
    dist_matrix = squareform(distances)
    rows, cols = np.where((dist_matrix <= threshold) & (dist_matrix > 0))

    # Create sparse adjacency matrix
    data = np.ones(len(rows), dtype=np.int8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Find connected components
    n_components, labels = connected_components(adjacency, directed=False)

    # Get component sizes
    sizes = np.bincount(labels)

    # Product of three largest
    largest_three = np.sort(sizes)[-3:]
    return int(np.prod(largest_three))
