from pathlib import Path
import re

import numpy as np
from pandas import IntervalIndex
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.console import Group
from rich.text import Text



class Section:
    RANGE_PATTERN = re.compile(r'^\d+-\d+$')

    def __init__(self, data: str):
        self.lines = data.strip().split('\n')

    @classmethod
    def from_file(cls, file_path: str) -> list['Section']:
        """Load sections from a file, split by blank lines."""
        current_dir = Path(__file__).parent
        full_path = current_dir / file_path
        with open(full_path) as f:
            return [cls(s) for s in f.read().split('\n\n')]

    def is_range_section(self):
        return all(self.RANGE_PATTERN.match(l) for l in self.lines)

    def get_as_range_tuples(self):
        if not self.is_range_section():
            raise ValueError("Not a Range Section")
        return [tuple(map(int, r.split('-'))) for r in self.lines]

    @property
    def values(self):
        return np.array([int(v) for v in self.lines])

    def __str__(self):
        return "\n".join(self.lines)

    def __repr__(self):
        return f"Section({len(self.lines)} lines)"


def parse_sections(sections: list[Section]) -> tuple[IntervalIndex, np.ndarray]:
    """Extract intervals and values from parsed sections."""
    range_tuples = []
    values_list = []
    for s in sections:
        if s.is_range_section():
            range_tuples.extend(s.get_as_range_tuples())
        else:
            values_list.append(s.values)
    idx = IntervalIndex.from_tuples(range_tuples, closed='both')
    values = np.concatenate(values_list)
    return idx, values


def find_interval_matches(idx: IntervalIndex, values: np.ndarray) -> np.ndarray:
    """Vectorized interval containment check using broadcasting."""
    left = np.array([iv.left for iv in idx])
    right = np.array([iv.right for iv in idx])
    return (values[:, None] >= left) & (values[:, None] <= right)


def count_interval_values(idx: IntervalIndex) -> int:
    """Count unique discrete integer values across all intervals."""
    if len(idx) == 0:
        return 0

    intervals = sorted((iv.left, iv.right) for iv in idx)
    merged = [intervals[0]]

    for left, right in intervals[1:]:
        prev_left, prev_right = merged[-1]
        if left <= prev_right + 1:
            # Overlap or adjacent, extend interval
            merged[-1] = (prev_left, max(prev_right, right))
        else:
            # No overlap, add new interval
            merged.append((left, right))

    return sum(right - left + 1 for left, right in merged)


def display_results(console: Console, total_possible: int, matching_values: np.ndarray, total: int):
    """Display puzzle results using rich formatting."""
    console.print(Padding("\n[bold]Puzzle Summary[/bold]", (0, 0, 1, 4)))
    console.print(Padding(
        Panel(
            Group(
                Text(f"Interval coverage : {total_possible:,}", style="cyan"),
                Text(f"\nMatching count    : {len(matching_values):,}", style="yellow"),
                Text(f"\nMatching total    : {total:,}", style="bold red"),
            ),
            title="Results",
            border_style="blue",
            expand=False,
            padding=(1, 1, 1, 1)
        ),
        (0, 0, 0, 4)
    ))


if __name__ == "__main__":
    sections = Section.from_file('data.dat')
    idx, values = parse_sections(sections)

    total_possible = count_interval_values(idx)
    has_match = find_interval_matches(idx, values).any(axis=1)
    matching_values = values[has_match]

    console = Console()
    display_results(console, total_possible, matching_values, matching_values.sum())