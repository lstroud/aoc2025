"""
Day 12: Christmas Tree Farm - Part 1

Now, let me tell you something about bin packing problems. The elves need to
load presents into bays under Christmas trees. Each bay has a grid of cells,
and we need to figure out if all the weird-shaped presents can actually fit
without overlapping.

Here's what a few hours of implementing backtracking, DLX, and Z3 solvers
taught me: sometimes the clever solution is to count. If the total cells
needed by all presents exceeds the available cells in the bay, they won't
fit. For this particular puzzle input, that's the whole answer.

As Mark Twain might have observed, I didn't have time to write a simple
solution, so I wrote three complicated ones first.

Available strategies (swap import to try different ones):
- hybrid: Area check + Z3 for small cases (THE WORKING SOLUTION)
- area_check: O(n) cell counting - the shortcut that worked for this input
- backtrack: Recursive brute force - hangs on UNSAT cases (sample only)
- dlx: Dancing Links exact cover - correct but slow (sample only)
- z3_sat: SAT solver - industrial strength but hangs on large cases (sample only)
"""

import threading
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table
from rich.text import Text

from strategies import (
    load_manifest,
    precompute_orientations,
    # THE WORKING SOLUTION - swap this import to try other strategies:
    hybrid as load_presents_strategy,
    # area_check as load_presents_strategy,  # Fast but wrong on sample bay 3
    # z3_sat as load_presents_strategy,      # Correct but hangs on data.dat
    # dlx as load_presents_strategy,         # Correct but slow
    # backtrack as load_presents_strategy,   # Hangs on UNSAT cases
)

console = Console()

# Which strategy are we using? (for display)
STRATEGY_NAME = "hybrid"


def load_presents_into_bay(bay, shapes, orientations) -> bool:
    """
    Can all the required presents be loaded into this bay?

    Delegates to the selected packing strategy. Swap the import at the top
    to try different strategies.
    """
    return load_presents_strategy(bay, shapes=shapes, orientations=orientations)


def count_packable_bays(from_file: str) -> int:
    """
    Count how many bays can be loaded with all their assigned presents.

    The elves will be disappointed about the bays that don't work out, but
    at least we can tell them which ones to prioritize for overflow storage.
    """
    manifest = load_manifest(from_file=from_file)
    shapes = manifest.present_shapes
    orientations = precompute_orientations(for_shapes=shapes)

    packable_count = sum(
        1 for bay in manifest.bays
        if load_presents_into_bay(bay, shapes=shapes, orientations=orientations)
    )

    return packable_count


def run_with_live_display(filename: str, expected: int = None) -> int:
    """Run solver with a festive live-updating display."""
    from rich.live import Live

    manifest = load_manifest(from_file=filename)
    shapes = manifest.present_shapes
    orientations = precompute_orientations(for_shapes=shapes)

    packable_count = 0
    total_bays = len(manifest.bays)
    last_fit_info = ""
    is_complete = False
    current_bay = 0

    progress = Progress(
        SpinnerColumn("christmas", finished_text="🎄"),
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=30, complete_style="green", finished_style="bright_green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        expand=False,
    )
    task = progress.add_task("Loading presents", total=total_bays)

    def make_display():
        stats = Table.grid(padding=(0, 2))
        stats.add_column(justify="right", style="dim")
        stats.add_column(justify="left")
        stats.add_row("Strategy:", f"[yellow]{STRATEGY_NAME}[/]")
        stats.add_row("Bays checked:", f"[cyan]{current_bay}[/] of [cyan]{total_bays}[/]")
        stats.add_row("Packable:", f"[bright_green]{packable_count}[/]")
        stats.add_row("Not packable:", f"[red]{current_bay - packable_count}[/]")

        if is_complete:
            result = Text()
            result.append(f"\nAnswer: ", style="dim")
            result.append(f"{packable_count}", style="bold bright_green")
            if expected is not None:
                if packable_count == expected:
                    result.append(f"  (expected {expected})", style="green")
                else:
                    result.append(f"  (expected {expected})", style="red")
        elif last_fit_info:
            result = Text(last_fit_info, style="dim italic")
        else:
            result = Text("Starting...", style="dim italic")

        return Panel(
            Group(progress, Text(""), stats, Text(""), result),
            title=f"[bold red]Day 12[/] [green]Part 1[/] [dim]({filename})[/]",
            subtitle="[dim]Christmas Tree Farm[/]",
            border_style="bright_green",
            padding=(1, 2),
        )

    stop_refresh = threading.Event()

    def refresh_loop(live):
        while not stop_refresh.is_set():
            live.update(make_display())
            stop_refresh.wait(0.05)

    with Live(make_display(), console=console, refresh_per_second=20) as live:
        refresh_thread = threading.Thread(target=refresh_loop, args=(live,), daemon=True)
        refresh_thread.start()

        for i, bay in enumerate(manifest.bays):
            fits = load_presents_into_bay(bay, shapes=shapes, orientations=orientations)

            current_bay = i + 1
            if fits:
                packable_count += 1
                last_fit_info = f"Bay {bay.grid_size}: {sum(bay.presents_to_load)} presents fit!"
            else:
                last_fit_info = f"Bay {bay.grid_size}: won't fit"

            progress.update(task, completed=current_bay)
            live.update(make_display())

        is_complete = True
        stop_refresh.set()
        live.update(make_display())

    return packable_count


if __name__ == "__main__":
    run_with_live_display("sample.dat", expected=2)
    console.print()
    run_with_live_display("data.dat")
