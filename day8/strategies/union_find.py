"""Union-Find strategy - the classic approach.

Sort pairs by distance, union first k. Path compression and
union by rank keep it nearly O(n) per operation.
"""
import numpy as np
from scipy.spatial.distance import pdist


class UnionFind:
    """Union-Find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if a merge happened."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already in same circuit

        # Attach smaller tree under larger
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def get_component_sizes(self) -> np.ndarray:
        """Return array of component sizes."""
        roots = set(self.find(i) for i in range(len(self.parent)))
        return np.array([self.size[r] for r in roots])


def solve(coords: np.ndarray, k: int = 1000) -> int:
    """
    Solve Part 1 using manual Union-Find.

    Args:
        coords: n×3 array of junction box coordinates
        k: Number of connection attempts

    Returns:
        Product of 3 largest circuit sizes
    """
    n = len(coords)
    uf = UnionFind(n)

    # Compute pairwise distances
    distances = pdist(coords, metric='euclidean')

    # Build (distance, i, j) tuples for sorting
    pairs = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((distances[idx], i, j))
            idx += 1

    # Sort by distance and attempt first k connections
    pairs.sort()
    for dist, i, j in pairs[:k]:
        uf.union(i, j)

    # Get component sizes
    sizes = uf.get_component_sizes()

    # Product of three largest
    largest_three = np.sort(sizes)[-3:]
    return int(np.prod(largest_three))
