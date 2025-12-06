"""
Day 6 Part 2: Vertical Digit Pivot

Parse column-aligned data where whitespace encodes digit positions. Numbers are
right-aligned within logical columns. Read digit positions vertically (right-to-left),
skip spaces, apply operators (* or +) to each column's result, and sum all results.

Example:
    123 328     Column 0: 123, 45, 6 → read vertically → [356, 24, 1]
     45 64      With operator *, result = 356 × 24 × 1 = 8544
      6 98
    *   +
"""

from dataclasses import dataclass
from pathlib import Path
import time
import tracemalloc
import numpy as np
from rich.console import Console
from rich.padding import Padding
from rich.table import Table


@dataclass
class ColumnSpec:
    """Defines a logical column's character range and aggregation operator."""
    start: int
    end: int
    operator: str


def load_file(file_path: str) -> tuple[np.ndarray, str]:
    """Load file as numpy char grid and operator line."""
    current_dir = Path(__file__).parent
    with open(current_dir / file_path) as f:
        lines = [line.rstrip('\n') for line in f]

    op_line = lines[-1]
    data_lines = lines[:-1]

    max_len = max(len(line) for line in data_lines)
    char_grid = np.array([list(line.ljust(max_len)) for line in data_lines], dtype='U1')

    return char_grid, op_line


def find_column_boundaries(op_line: str) -> list[ColumnSpec]:
    """Find logical column boundaries from operator positions in the last row."""
    op_chars = np.array(list(op_line))
    op_mask = np.isin(op_chars, ['*', '+'])
    positions = np.where(op_mask)[0]
    operators = op_chars[op_mask]

    starts = positions
    ends = np.append(positions[1:], len(op_line))
    return [ColumnSpec(s, e, o) for s, e, o in zip(starts, ends, operators)]


def read_vertical_numbers(char_grid: np.ndarray) -> list[int]:
    """
    Read digit columns right-to-left, forming numbers from non-space characters.

    For each character column (starting from rightmost), collect digits top-to-bottom,
    skip spaces, and form an integer. This effectively reads the ones place first,
    then tens, then hundreds across all rows.
    """
    _, n_cols = char_grid.shape
    numbers = []

    for col_idx in range(n_cols - 1, -1, -1):
        col_chars = char_grid[:, col_idx]
        digits = [c for c in col_chars if c != ' ']
        if digits:
            numbers.append(int(''.join(digits)))

    return numbers


def read_vertical_numbers_vectorized(char_grid: np.ndarray) -> list[int]:
    """
    Vectorized version.

    Converts digits to numeric values, computes place values based on
    cumulative count from bottom, then sums each column.
    """
    mask = char_grid != ' '

    # Convert char digits to int: '0'->0, '1'->1, etc. (ord('0') = 48)
    numeric = np.where(mask, char_grid.view(np.uint32) - 48, 0)

    # Cumulative count from bottom gives position within valid digits
    cumsum_from_bottom = np.cumsum(mask[::-1], axis=0)[::-1]

    # Place value: 10^(position - 1), clamp to 0 minimum to avoid negative exponent
    exponents = np.maximum(cumsum_from_bottom - 1, 0)
    place_values = np.where(mask, 10 ** exponents, 0)

    # Multiply digits by place values and sum down columns
    column_numbers = (numeric * place_values).sum(axis=0)

    # Filter empty columns and reverse (right-to-left)
    return column_numbers[mask.any(axis=0)][::-1].tolist()


def solve(file_path: str) -> tuple[int, list[ColumnSpec]]:
    """Solve the puzzle: pivot columns vertically, apply operators, sum results."""
    char_grid, op_line = load_file(file_path)
    specs = find_column_boundaries(op_line)

    total = 0
    for spec in specs:
        col_chars = char_grid[:, spec.start:spec.end]
        numbers = read_vertical_numbers(col_chars)

        if spec.operator == '*':
            total += int(np.prod(numbers))
        elif spec.operator == '+':
            total += int(np.sum(numbers))
        else:
            raise ValueError(f"Unknown operator: {spec.operator}")

    return total, specs


def benchmark(char_grid: np.ndarray, iterations: int = 1000) -> dict:
    """Benchmark both implementations for time and memory."""
    funcs = [('loop', read_vertical_numbers), ('vectorized', read_vertical_numbers_vectorized)]

    # Verify outputs match
    outputs = [func(char_grid) for _, func in funcs]
    if outputs[0] != outputs[1]:
        raise ValueError(f"Output mismatch: loop={outputs[0]}, vectorized={outputs[1]}")

    results = {}
    for name, func in funcs:
        # Time benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            func(char_grid)
        elapsed = (time.perf_counter() - start) / iterations * 1_000_000  # microseconds

        # Memory benchmark (single call)
        tracemalloc.start()
        func(char_grid)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[name] = {'time_us': elapsed, 'memory_bytes': peak}

    return results


if __name__ == "__main__":
    console = Console()
    total, specs = solve('sample.dat')

    mult_count = sum(1 for s in specs if s.operator == '*')
    add_count = sum(1 for s in specs if s.operator == '+')

    console.print(Padding("\n[bold]Day 6: Vertical Digit Pivot[/bold]", (0, 0, 1, 4)))
    console.print(Padding(f"[dim]Columns:[/dim] {len(specs)} ([red]*[/red] {mult_count}, [green]+[/green] {add_count})", (0, 0, 0, 4)))
    console.print(Padding(f"[bold green]Total[/bold green]: {total:,}\n", (0, 0, 0, 4)))

    # Benchmark comparison
    char_grid, _ = load_file('sample.dat')
    bench = benchmark(char_grid)

    table = Table(title="Performance Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Time (μs)", justify="right")
    table.add_column("Memory (bytes)", justify="right")

    for name, stats in bench.items():
        table.add_row(name, f"{stats['time_us']:.2f}", f"{stats['memory_bytes']:,}")

    console.print(Padding(table, (0, 0, 1, 4)))
