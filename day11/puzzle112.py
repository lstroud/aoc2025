"""Day 11: Reactor - Part 2

The elves narrowed it down: the bug travels through both the DAC
(digital-to-analog converter) and the FFT (fast Fourier transform).
Apparently even Christmas runs on signal processing.

Algorithm: Same DFS, but now we track checkpoints. Like Santa checking
if you've been naughty AND nice - both boxes must be ticked.
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


def trace_suspicious_paths(file_name: str) -> dict:
    """Count paths from server to reactor that visit both troublemakers."""
    wiring = untangle_cable_spaghetti(file_name)

    # Gather stats
    num_devices = len(wiring)
    num_connections = sum(len(v) for v in wiring.values())
    max_branch = max(len(v) for v in wiring.values())

    @functools.cache
    def paths_through_checkpoints(device: str, saw_dac: bool, saw_fft: bool) -> int:
        """
        DFS with checkpoint tracking. A path only counts if it passed
        through both the DAC and FFT - the dynamic duo of data corruption.
        """
        if device == "out":
            # Only count paths that visited both suspects
            return 1 if (saw_dac and saw_fft) else 0

        downstream = wiring.get(device)
        if downstream is None:
            return 0  # Dead end, no reactor for you

        # Recurse, updating checkpoint flags as we go
        return sum(
            paths_through_checkpoints(
                next_device,
                saw_dac or next_device == "dac",
                saw_fft or next_device == "fft"
            )
            for next_device in downstream
        )

    return {
        "paths": paths_through_checkpoints("svr", False, False),
        "devices": num_devices,
        "connections": num_connections,
        "max_branch": max_branch,
    }


# Animation frames showing data flowing through checkpoints
PATH_FRAMES = [
    "[dim]svr[/] ─── [dim]···[/] ─── [dim]dac[/] ─── [dim]···[/] ─── [dim]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[yellow]svr[/] ─── [dim]···[/] ─── [dim]dac[/] ─── [dim]···[/] ─── [dim]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─[yellow]──[/] [dim]···[/] ─── [dim]dac[/] ─── [dim]···[/] ─── [dim]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [yellow]···[/] ─── [dim]dac[/] ─── [dim]···[/] ─── [dim]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─[yellow]──[/] [dim]dac[/] ─── [dim]···[/] ─── [dim]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─── [bold red]dac[/] ─── [dim]···[/] ─── [dim]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─── [green]dac[/] ─[yellow]──[/] [dim]···[/] ─── [dim]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─── [green]dac[/] ─── [yellow]···[/] ─── [dim]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─── [green]dac[/] ─── [dim]···[/] ─[yellow]──[/] [dim]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─── [green]dac[/] ─── [dim]···[/] ─── [bold red]fft[/] ─── [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─── [green]dac[/] ─── [dim]···[/] ─── [green]fft[/] ─[yellow]──[/] [dim]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─── [green]dac[/] ─── [dim]···[/] ─── [green]fft[/] ─── [yellow]···[/] ─── [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─── [green]dac[/] ─── [dim]···[/] ─── [green]fft[/] ─── [dim]···[/] ─[yellow]──[/] [dim]out[/]",
    "[dim]svr[/] ─── [dim]···[/] ─── [green]dac[/] ─── [dim]···[/] ─── [green]fft[/] ─── [dim]···[/] ─── [bold green]out[/]",
]


def make_panel(content: str) -> Panel:
    """Wrap content in the standard checkpoint panel."""
    return Panel(
        content,
        title="[bold]🎅 Day 11 Part 2 🎅[/]",
        border_style="red",
        padding=(1, 2),
    )


def display_results(result: dict, cycles: int = 1):
    """Animate data flowing through checkpoints, then show final state."""
    stats = f"""
   [bold green]{result['paths']:,}[/] suspicious paths pass through both checkpoints

   [dim]📊 {result['devices']} devices, {result['connections']} connections, max branching: {result['max_branch']}[/]"""

    # Build animation frames with stats
    anim_frames = [f"🎄 {frame} 🎄{stats}" for frame in PATH_FRAMES]

    with Live(make_panel(anim_frames[0]), console=console, refresh_per_second=10) as live:
        for _ in range(cycles):
            for frame in anim_frames:
                live.update(make_panel(frame))
                time.sleep(0.1)

        # Final frame with checkpoint labels
        final_viz = f"""\
🎄 [dim]svr[/] ─── [dim]···[/] ─── [green]dac[/] ─── [dim]···[/] ─── [green]fft[/] ─── [dim]···[/] ─── [bold green]out[/] 🎄
                    [green]^^^[/]           [green]^^^[/]
               [dim]checkpoint 1[/]   [dim]checkpoint 2[/]
{stats}"""
        live.update(make_panel(final_viz))


if __name__ == "__main__":
    sample = trace_suspicious_paths("sample2.dat")
    print(f"Sample: {sample['paths']} (expected 2)")

    result = trace_suspicious_paths("data.dat")
    display_results(result)
