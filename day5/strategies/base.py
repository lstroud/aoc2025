from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class MatchingStrategy(ABC):
    """Abstract base class for interval matching strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the strategy."""

    @abstractmethod
    def find_matches(self, range_tuples: list[tuple[int, int]], values: np.ndarray) -> np.ndarray:
        """
        Find which values fall within any interval.

        Returns a boolean mask where True indicates the value matches.
        """

    def count_coverage(self, range_tuples: list[tuple[int, int]]) -> int:
        """
        Count unique discrete integer values across all intervals.

        Default implementation uses interval merging. Subclasses may override
        with alternative algorithms.
        """
        if len(range_tuples) == 0:
            return 0

        intervals = sorted(range_tuples)
        merged = [intervals[0]]

        for left, right in intervals[1:]:
            prev_left, prev_right = merged[-1]
            if left <= prev_right + 1:
                merged[-1] = (prev_left, max(prev_right, right))
            else:
                merged.append((left, right))

        return sum(right - left + 1 for left, right in merged)


@dataclass
class BenchmarkResult:
    """Results from running a strategy benchmark."""
    strategy_name: str
    match_ms: float
    coverage_ms: float
    peak_memory_kb: float
    matching_count: int
    matching_total: int
    coverage: int
