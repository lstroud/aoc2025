"""Day 11: Reactor - Part 1

The toroidal reactor's wiring is a mess. Data flows through devices
like gossip through the North Pole breakroom - one direction only,
but with many possible paths to the coffee machine (or in this case, 'out').

Algorithm: DFS with memoization. Each device remembers how many paths
lead from it to the reactor output, because asking the same elf twice
is just inefficient.
"""
import functools
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.live import Live

console = Console()


def untangle_cable_spaghetti(file_name: str) -> dict[str, list[str]]:
    """Parse the device wiring list that some elf scribbled on a napkin."""
    manual_path = Path(__file__).parent / file_name
    with open(manual_path) as f:
        return {k: v.split() for k, v in (line.split(":") for line in f)}


def trace_data_paths(file_name: str) -> dict:
    """Count all paths from 'you' to 'out' - the reactor needs to know."""
    wiring = untangle_cable_spaghetti(file_name)

    # Gather stats while we're at it
    num_devices = len(wiring)
    num_connections = sum(len(v) for v in wiring.values())
    max_branch = max(len(v) for v in wiring.values())

    @functools.cache
    def paths_to_reactor(device: str) -> int:
        """Recursively count paths. Memoized because elves hate redundant work."""
        if device == "out":
            return 1  # Found the reactor! That's one path.

        downstream = wiring.get(device)
        if downstream is None:
            return 0  # Dead end. Someone forgot to plug this in.

        # Sum paths through all downstream devices
        return sum(paths_to_reactor(next_device) for next_device in downstream)

    return {
        "paths": paths_to_reactor("you"),
        "devices": num_devices,
        "connections": num_connections,
        "max_branch": max_branch,
    }


REACTOR_FRAMES = [
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ [/][yellow]o[/][dim] · · · · · │
    │ · · · · · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · [/][yellow]o[/][dim] · · · · │
    │ · · · · · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · [/][yellow]o[/][dim] · · · │
    │ · · · · · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · · [/][yellow]o[/][dim] · · │
    │ · · · · · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · · · [/][yellow]o[/][dim] · │
    │ · · · · · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · · · · [/][yellow]o[/][dim] │
    │ · · · · · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · · · · · │
    │ [/][yellow]o[/][dim] · · · · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · · · · · │
    │ · [/][yellow]o[/][dim] · · · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · · · · · │
    │ · · [/][yellow]o[/][dim] · · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · · · · · │
    │ · · · [/][yellow]o[/][dim] · · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · · · · · │
    │ · · · · [/][yellow]o[/][dim] · │  ← {paths:,} paths
    └─────────────┘[/]""",
    """\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ · · · · · · │
    │ · · · · · [/][yellow]o[/][dim] │  ← {paths:,} paths
    └─────────────┘[/]""",
]


def make_panel(content: str) -> Panel:
    """Wrap content in the standard reactor panel."""
    return Panel(
        content,
        title="[bold]🎅 Day 11 Part 1 🎅[/]",
        border_style="red",
        padding=(1, 2),
    )


def display_results(result: dict, cycles: int = 2):
    """Animate data flowing through reactor, then show final state."""
    stats = f"\n[dim]📊 {result['devices']} devices, {result['connections']} connections, max branching: {result['max_branch']}[/]"

    # Animate within the panel
    frames = [f.format(**result) + stats for f in REACTOR_FRAMES]
    with Live(make_panel(frames[0]), console=console, refresh_per_second=8) as live:
        for _ in range(cycles):
            for frame in frames:
                live.update(make_panel(frame))
                time.sleep(0.12)

        # Final frame: all lights on
        final_art = f"""\
     🎄 REACTOR 🎄
[dim]    ┌─────────────┐
    │ [/][green]o o o o o o[/][dim] │
    │ [/][green]o o o o o o[/][dim] │  ← [bold green]{result['paths']:,}[/] paths
    └─────────────┘[/]
{stats}"""
        live.update(make_panel(final_art))


if __name__ == "__main__":
    sample = trace_data_paths("sample.dat")
    print(f"Sample: {sample['paths']} (expected 5)")

    result = trace_data_paths("data.dat")
    display_results(result)
