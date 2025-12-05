from pathlib import Path
import re
import time
import tracemalloc

import numpy as np
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.table import Table

from strategies import (
    MatchingStrategy,
    BenchmarkResult,
    BroadcastStrategy,
    IntervalTreeStrategy,
    SweepLineStrategy,
)


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


def parse_sections(sections: list[Section]) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Extract interval tuples and values from parsed sections."""
    range_tuples = []
    values_list = []
    for s in sections:
        if s.is_range_section():
            range_tuples.extend(s.get_as_range_tuples())
        else:
            values_list.append(s.values)
    values = np.concatenate(values_list)
    return range_tuples, values


def benchmark_strategy(
    strategy: MatchingStrategy,
    range_tuples: list[tuple[int, int]],
    values: np.ndarray
) -> BenchmarkResult:
    """Run a strategy and capture timing and memory metrics."""
    tracemalloc.start()

    # Benchmark find_matches
    start = time.perf_counter()
    has_match = strategy.find_matches(range_tuples, values)
    match_elapsed = time.perf_counter() - start

    matching_values = values[has_match]
    matching_total = matching_values.sum()

    # Benchmark count_coverage
    start = time.perf_counter()
    coverage = strategy.count_coverage(range_tuples)
    coverage_elapsed = time.perf_counter() - start

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        strategy_name=strategy.name,
        match_ms=match_elapsed * 1000,
        coverage_ms=coverage_elapsed * 1000,
        peak_memory_kb=peak / 1024,
        matching_count=len(matching_values),
        matching_total=matching_total,
        coverage=coverage
    )


def verify_results(results: list[BenchmarkResult]) -> list[str]:
    """Verify all strategies produce identical results. Returns list of discrepancies."""
    if len(results) < 2:
        return []

    first = results[0]
    discrepancies = []

    for r in results[1:]:
        if r.matching_count != first.matching_count:
            discrepancies.append(
                f"{r.strategy_name} matching_count={r.matching_count} != {first.strategy_name} {first.matching_count}"
            )
        if r.matching_total != first.matching_total:
            discrepancies.append(
                f"{r.strategy_name} matching_total={r.matching_total} != {first.strategy_name} {first.matching_total}"
            )
        if r.coverage != first.coverage:
            discrepancies.append(
                f"{r.strategy_name} coverage={r.coverage} != {first.strategy_name} {first.coverage}"
            )

    return discrepancies


def display_results(console: Console, results: list[BenchmarkResult]):
    """Display puzzle results and benchmark comparison."""
    console.print(Padding("\n[bold]Puzzle Summary[/bold]", (0, 0, 1, 4)))

    discrepancies = verify_results(results)
    if discrepancies:
        console.print(Padding(
            Panel(
                "\n".join(discrepancies),
                title="Strategy Mismatch",
                border_style="red",
                expand=False,
            ),
            (0, 0, 1, 4)
        ))

    first = results[0]
    console.print(Padding(
        Panel(
            Group(
                Text(f"Interval coverage : {first.coverage:,}", style="cyan"),
                Text(f"\nMatching count    : {first.matching_count:,}", style="yellow"),
                Text(f"\nMatching total    : {first.matching_total:,}", style="bold red"),
            ),
            title="Results",
            border_style="blue",
            expand=False,
            padding=(1, 1, 1, 1)
        ),
        (0, 0, 0, 4)
    ))

    table = Table(title="Strategy Comparison", border_style="blue")
    table.add_column("Strategy", style="cyan")
    table.add_column("Match (ms)", justify="right", style="yellow")
    table.add_column("Coverage (ms)", justify="right", style="yellow")
    table.add_column("Peak Memory (KB)", justify="right", style="magenta")

    for r in results:
        table.add_row(r.strategy_name, f"{r.match_ms:.2f}", f"{r.coverage_ms:.2f}", f"{r.peak_memory_kb:.1f}")

    console.print(Padding(table, (0, 0, 1, 4)))


if __name__ == "__main__":
    sections = Section.from_file('large.dat')
    range_tuples, values = parse_sections(sections)

    strategies: list[MatchingStrategy] = [
        BroadcastStrategy(),
        IntervalTreeStrategy(),
        SweepLineStrategy(),
    ]

    results = [benchmark_strategy(s, range_tuples, values) for s in strategies]

    console = Console()
    display_results(console, results)
