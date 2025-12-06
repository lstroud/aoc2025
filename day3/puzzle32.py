"""
Advent of Code 2025 - Day 3, Part 2: Largest N-Digit Selection

Problem: From each number, select 12 digits (in order) that form the
largest possible 12-digit number. Sum all results.

This file demonstrates two approaches:
1. Greedy argmax: At each step, pick the largest digit that leaves enough remaining
2. Monotonic stack: Remove smallest digits until we have exactly N left

Both have O(n) complexity but different constant factors.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from rich.console import Console


def parse_file(file_path: str) -> pd.DataFrame:
    """Load numbers from CSV file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)


def get_max_n_digit_argmax(s: str, n: int) -> int:
    """
    Select n digits forming the largest number using greedy argmax.

    At each position i, choose the largest digit from arr[start:end]
    where end ensures enough digits remain for positions i+1 to n-1.
    """
    arr = np.array(list(s), dtype=np.int8)
    result = np.empty(n, dtype=np.int8)
    start = 0

    for i in range(n):
        remaining = n - i
        end = len(arr) - remaining + 1
        best_idx = start + np.argmax(arr[start:end])
        result[i] = arr[best_idx]
        start = best_idx + 1

    return int(''.join(result.astype(str)))


def get_max_n_digit_stack(s: str, n: int) -> int:
    """
    Select n digits forming the largest number using monotonic stack.

    Process digits left to right. When a larger digit arrives, pop smaller
    digits from the stack (up to the number we need to drop). This maintains
    a decreasing stack that produces the largest result.
    """
    to_drop = len(s) - n
    stack = []

    for d in s:
        while to_drop and stack and d > stack[-1]:
            stack.pop()
            to_drop -= 1
        stack.append(d)

    return int(''.join(stack[:n]))


def benchmark_approaches(df: pd.DataFrame, n: int) -> dict:
    """Run both approaches and return timing results."""
    results = {}

    start = time.perf_counter()
    df['max_digits_argmax'] = df['value'].apply(lambda x: get_max_n_digit_argmax(str(x), n))
    results['argmax_ms'] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    df['max_digits_stack'] = df['value'].apply(lambda x: get_max_n_digit_stack(str(x), n))
    results['stack_ms'] = (time.perf_counter() - start) * 1000

    return results


def display_results(console: Console, total: int, timing: dict):
    """Display puzzle results and timing comparison."""
    console.print("\n[bold]Puzzle Summary[/bold]")
    console.print(f"  [bold green]Total[/bold green]        : {int(total):,}")
    console.print(f"\n  [cyan]Argmax time[/cyan]  : {timing['argmax_ms']:.1f} ms")
    console.print(f"  [cyan]Stack time[/cyan]   : {timing['stack_ms']:.1f} ms\n")


if __name__ == "__main__":
    df = parse_file('data.dat')
    df = pd.DataFrame(df.values.flatten(), columns=['value']).dropna().reset_index(drop=True)

    timing = benchmark_approaches(df, n=12)
    total = df['max_digits_argmax'].sum()

    console = Console()
    display_results(console, total, timing)
