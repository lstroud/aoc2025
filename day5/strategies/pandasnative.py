import numpy as np
import pandas as pd
from pandas import IntervalIndex

from .base import MatchingStrategy


class PandasNativeStrategy(MatchingStrategy):
    """
    Pandas-native approach using IntervalIndex.get_indexer().

    Uses binary search on sorted interval endpoints to find which interval
    (if any) contains each value. Returns -1 for non-matches.

    Complexity: O(m log m) build + O(n log m) queries via binary search
    """

    @property
    def name(self) -> str:
        return "Pandas Native"

    def find_matches(self, range_tuples: list[tuple[int, int]], values: np.ndarray) -> np.ndarray:
        idx = IntervalIndex.from_tuples(range_tuples, closed='both')
        # get_indexer_non_unique returns (indexer, missing)
        # missing contains indices of values with no interval match
        _, missing = idx.get_indexer_non_unique(values)
        result = np.ones(len(values), dtype=bool)
        result[missing] = False
        return result
