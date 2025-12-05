import numpy as np
from pandas import IntervalIndex

from .base import MatchingStrategy


class IntervalIndexStrategy(MatchingStrategy):
    """
    Pandas IntervalIndex using sorted endpoint binary search.

    Builds an IntervalIndex from tuples and uses vectorized containment
    checks via broadcasting against sorted endpoints.

    Complexity: O(m log m) build + O(n log m) queries
    """

    @property
    def name(self) -> str:
        return "IntervalIndex"

    def find_matches(self, range_tuples: list[tuple[int, int]], values: np.ndarray) -> np.ndarray:
        idx = IntervalIndex.from_tuples(range_tuples, closed='both')
        left = np.array([iv.left for iv in idx])
        right = np.array([iv.right for iv in idx])
        containment = (values[:, None] >= left) & (values[:, None] <= right)
        return containment.any(axis=1)
