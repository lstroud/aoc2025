"""Compare benchmark results across multiple datasets to show crossover point."""
from puzzle52 import Section, parse_sections, benchmark_strategy
from strategies import (
    BroadcastStrategy,
    IntervalIndexStrategy,
    IntervalTreeStrategy,
    PandasNativeStrategy,
    SweepLineStrategy,
)
from rich.console import Console
from rich.table import Table

console = Console()

datasets = ['sample.dat', 'data.dat', 'large.dat']
strategies = [
    BroadcastStrategy(),
    IntervalIndexStrategy(),
    IntervalTreeStrategy(),
    PandasNativeStrategy(),
    SweepLineStrategy(),
]

for filename in datasets:
    sections = Section.from_file(filename)
    range_tuples, values = parse_sections(sections)

    table = Table(title=f"{filename} ({len(range_tuples)} intervals, {len(values)} values)")
    table.add_column("Strategy", style="cyan")
    table.add_column("Match (ms)", justify="right", style="yellow")
    table.add_column("Coverage (ms)", justify="right", style="yellow")
    table.add_column("Peak Memory (KB)", justify="right", style="magenta")

    for strategy in strategies:
        r = benchmark_strategy(strategy, range_tuples, values)
        table.add_row(r.strategy_name, f"{r.match_ms:.2f}", f"{r.coverage_ms:.2f}", f"{r.peak_memory_kb:.1f}")

    console.print(table)
    console.print()
