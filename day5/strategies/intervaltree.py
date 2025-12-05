import numpy as np
from intervaltree import IntervalTree

from .base import MatchingStrategy


class IntervalTreeStrategy(MatchingStrategy):
    """
    Interval tree for O(log m) per-query lookups.

    Builds a balanced tree structure for efficient point-in-interval queries.
    Best when querying many values against a fixed interval set.

    Complexity: O(m log m) build + O(n log m) queries
    """

    @property
    def name(self) -> str:
        return "Interval Tree"

    def find_matches(self, range_tuples: list[tuple[int, int]], values: np.ndarray) -> np.ndarray:
        tree = IntervalTree()
        for left, right in range_tuples:
            tree.addi(left, right + 1)  # +1 for inclusive end
        return np.array([tree.overlaps(v) for v in values])
