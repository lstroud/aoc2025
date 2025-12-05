import numpy as np

from .base import MatchingStrategy


class BroadcastStrategy(MatchingStrategy):
    """
    Vectorized interval containment using numpy broadcasting.

    Creates an (n_values, n_intervals) matrix comparing all values against
    all interval bounds simultaneously. Memory intensive but fast for
    moderate sizes.

    Complexity: O(n × m) time and space
    """

    @property
    def name(self) -> str:
        return "Broadcasting"

    def find_matches(self, range_tuples: list[tuple[int, int]], values: np.ndarray) -> np.ndarray:
        left = np.array([r[0] for r in range_tuples])
        right = np.array([r[1] for r in range_tuples])
        containment = (values[:, None] >= left) & (values[:, None] <= right)
        return containment.any(axis=1)
