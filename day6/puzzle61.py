from pathlib import Path
import time
import tracemalloc
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.padding import Padding


def parse_file(file_path: str) -> pd.DataFrame:
    """Load numbers from CSV file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None, delim_whitespace=True)


def pandas_agg(df: pd.DataFrame) -> float:
    """Pandas agg() with dict mapping columns to operations."""
    operators = df.iloc[-1].tolist()
    data = df.iloc[:-1].apply(pd.to_numeric)
    op_map = {'*': 'prod', '+': 'sum'}
    agg_dict = {col: op_map[op] for col, op in zip(data.columns, operators)}
    result = data.agg(agg_dict)
    return result.sum()


def numpy_mask(df: pd.DataFrame) -> float:
    """Numpy with boolean masking for vectorized operations."""
    ops = df.iloc[-1].values
    data = df.iloc[:-1].to_numpy(dtype=float)

    is_multiply = (ops == '*')
    is_sum = (ops == '+')

    mult_results = data[:, is_multiply].prod(axis=0)
    sum_results = data[:, is_sum].sum(axis=0)

    return mult_results.sum() + sum_results.sum()


def benchmark(name: str, func, df: pd.DataFrame) -> dict:
    """Run a function and measure time and memory."""
    tracemalloc.start()
    start = time.perf_counter()

    total = func(df)

    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'name': name,
        'total': total,
        'time_ms': elapsed_ms,
        'memory_kb': peak_memory / 1024
    }


if __name__ == "__main__":
    console = Console()
    console.print(Padding("\n[bold]Day 6: Column Operations[/bold]", (0, 0, 1, 4)))

    df = parse_file('data.dat')
    rows, cols = df.shape[0] - 1, df.shape[1]
    console.print(Padding(f"[cyan]Data shape:[/cyan] {rows} rows x {cols} columns", (0, 0, 1, 4)))

    # Run benchmarks
    results = [
        benchmark("Pandas Agg", pandas_agg, df),
        benchmark("Numpy Mask", numpy_mask, df),
    ]

    # Display results
    table = Table(title=f"Strategy Comparison ({rows} rows x {cols} columns)")
    table.add_column("Strategy", style="cyan")
    table.add_column("Total", justify="right", style="green")
    table.add_column("Time (ms)", justify="right", style="magenta")
    table.add_column("Memory (KB)", justify="right", style="blue")

    for r in results:
        table.add_row(
            r['name'],
            f"{r['total']:,.0f}",
            f"{r['time_ms']:.3f}",
            f"{r['memory_kb']:.1f}"
        )

    console.print(Padding(table, (1, 0, 1, 4)))
