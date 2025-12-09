"""KD-Tree strategy using scipy.spatial.

Instead of computing all O(n²) distances, use spatial indexing
to find neighbors within a radius. For sparse connections this
can be much faster - but we need to know the radius upfront.
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist


def solve(coords: np.ndarray, k: int = 1000) -> int:
    """
    Solve Part 1 using KD-Tree radius queries.

    Args:
        coords: n×3 array of junction box coordinates
        k: Number of connection attempts

    Returns:
        Product of 3 largest circuit sizes
    """
    n = len(coords)

    # Still need pdist to find the k-th distance threshold
    # (KD-Tree shines when you know the radius, not when finding it)
    distances = pdist(coords, metric='euclidean')
    threshold = np.partition(distances, k - 1)[k - 1]

    # Build KD-Tree and query pairs within threshold
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=threshold, output_type='ndarray')

    # Build sparse adjacency matrix
    if len(pairs) > 0:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(rows), dtype=np.int8)
        adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        adjacency = csr_matrix((n, n), dtype=np.int8)

    # Find connected components
    n_components, labels = connected_components(adjacency, directed=False)

    # Get component sizes
    sizes = np.bincount(labels)

    # Product of three largest
    largest_three = np.sort(sizes)[-3:]
    return int(np.prod(largest_three))
