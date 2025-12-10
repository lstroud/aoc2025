"""Day 9 Part 2 - Strategy comparison and benchmarking."""
import time
import tracemalloc
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.padding import Padding

from strategies import (
    load_tile_locations,
    shapely_contains,
    convex_hull,
    rasterize,
)


def benchmark(name: str, func, coords: list[tuple[int, int]]) -> dict:
    """Run a strategy and measure time and memory."""
    tracemalloc.start()
    start = time.perf_counter()

    result = func(coords)

    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'name': name,
        'result': result,
        'time_ms': elapsed_ms,
        'memory_kb': peak_memory / 1024,
    }


def display_results(console: Console, results: list[dict], n_tiles: int):
    """Display benchmark results in a festive table."""
    table = Table(title=f"🎄 Carpet Strategy Comparison ({n_tiles:,} red tiles) 🎅")
    table.add_column("Strategy", style="cyan")
    table.add_column("Biggest Carpet", justify="right", style="green")
    table.add_column("Time (ms)", justify="right", style="magenta")
    table.add_column("Memory (KB)", justify="right", style="blue")

    for r in results:
        result_str = f"{r['result']:,}" if r['result'] is not None else "[red]ERROR[/red]"
        table.add_row(
            r['name'],
            result_str,
            f"{r['time_ms']:.3f}",
            f"{r['memory_kb']:.1f}",
        )

    console.print(Padding(table, (1, 0, 1, 4)))


def run_benchmarks(console: Console, coords: list[tuple[int, int]], strategies: dict):
    """Run all benchmarks and display results."""
    console.print(Padding(
        f"[cyan]Red tiles:[/cyan] {len(coords):,}",
        (1, 0, 0, 4)
    ))
    console.print(Padding(
        f"[cyan]Pairs to check:[/cyan] {len(coords) * (len(coords) - 1) // 2:,}",
        (0, 0, 1, 4)
    ))

    results = []
    for name, func in strategies.items():
        try:
            results.append(benchmark(name, func, coords))
        except NotImplementedError:
            results.append({
                'name': name,
                'result': None,
                'time_ms': 0,
                'memory_kb': 0,
            })
            console.print(f"[yellow]  ⏭ Skipping {name} (not implemented)[/yellow]")

    display_results(console, results, len(coords))


if __name__ == "__main__":
    console = Console()
    console.print(Panel(
        "[bold]Movie Theater Renovation[/bold]\n"
        "[dim]Finding the biggest carpet that fits the elves' questionable floor plan[/dim]",
        title="[green]Day 9 Part 2[/green]",
        border_style="red",
        expand=False
    ))

    # Load data
    coords = load_tile_locations('sample.dat')

    # Define strategies to benchmark
    # Note: Rasterize works on sample but memory-explodes on large data
    strategies = {
        'Shapely Contains': shapely_contains,
        'Convex Hull': convex_hull,
        # 'Rasterize + Mask': rasterize,
    }

    run_benchmarks(console, coords, strategies)
