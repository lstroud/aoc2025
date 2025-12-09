"""Hierarchical clustering strategy using scipy.

Single-linkage clustering is Union-Find with better marketing.
The linkage matrix tracks every merge, making it trivial to
count circuits after k connection attempts.
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


def solve(coords: np.ndarray, k: int = 1000) -> int:
    """
    Solve Part 1 using scipy hierarchical clustering.

    Args:
        coords: n×3 array of junction box coordinates
        k: Number of connection attempts

    Returns:
        Product of 3 largest circuit sizes
    """
    n = len(coords)

    # Compute all pairwise distances
    distances = pdist(coords, metric='euclidean')

    # Single-linkage: merge closest clusters first
    Z = linkage(distances, method='single')

    # Find threshold distance for k-th closest pair
    sorted_distances = np.sort(distances)
    threshold = sorted_distances[k - 1]

    # Count actual merges (some attempts are redundant)
    actual_merges = np.sum(Z[:, 2] <= threshold)
    num_circuits = n - actual_merges

    # Get cluster labels and sizes
    labels = fcluster(Z, num_circuits, criterion='maxclust')
    sizes = np.bincount(labels)

    # Product of three largest
    largest_three = np.sort(sizes)[-3:]
    return int(np.prod(largest_three))
