import numpy as np

from .base import MatchingStrategy


class SweepLineStrategy(MatchingStrategy):
    """
    Sweep line algorithm using event-based processing.

    Creates +1/-1 events at interval boundaries, sorts all positions,
    and uses cumulative sum to track coverage depth.

    Complexity: O((n + m) log(n + m)) dominated by sorting
    """

    @property
    def name(self) -> str:
        return "Sweep Line"

    def find_matches(self, range_tuples: list[tuple[int, int]], values: np.ndarray) -> np.ndarray:
        left = np.array([r[0] for r in range_tuples])
        right = np.array([r[1] for r in range_tuples])
        n_intervals = len(left)
        n_values = len(values)

        positions = np.concatenate([left, right + 1, values])
        event_types = np.concatenate([
            np.zeros(n_intervals),      # starts: type 0
            np.full(n_intervals, 2),    # ends: type 2
            np.ones(n_values)           # queries: type 1
        ])
        deltas = np.concatenate([
            np.ones(n_intervals),
            -np.ones(n_intervals),
            np.zeros(n_values)
        ])
        original_idx = np.concatenate([
            np.full(2 * n_intervals, -1),
            np.arange(n_values)
        ])

        sort_order = np.lexsort((event_types, positions))
        deltas_sorted = deltas[sort_order]
        original_idx_sorted = original_idx[sort_order]

        depth = np.cumsum(deltas_sorted)

        is_query_sorted = original_idx_sorted >= 0
        result = np.zeros(n_values, dtype=bool)
        result[original_idx_sorted[is_query_sorted].astype(int)] = depth[is_query_sorted] > 0

        return result

    def count_coverage(self, range_tuples: list[tuple[int, int]]) -> int:
        """
        Count coverage using vectorized sweep line algorithm.

        Creates +1/-1 events at interval boundaries, sorts positions,
        groups deltas, and sums gaps where depth > 0.
        """
        if len(range_tuples) == 0:
            return 0

        left = np.array([r[0] for r in range_tuples])
        right = np.array([r[1] for r in range_tuples])

        positions = np.concatenate([left, right + 1])
        deltas = np.concatenate([np.ones(len(left)), -np.ones(len(left))])

        order = np.argsort(positions)
        positions = positions[order]
        deltas = deltas[order]

        unique_pos, indices = np.unique(positions, return_index=True)
        grouped_deltas = np.add.reduceat(deltas, indices)

        depth = np.cumsum(grouped_deltas)
        gaps = np.diff(unique_pos)
        covered = depth[:-1] > 0

        return int((gaps * covered).sum())
