"""Wave picking simulation using 2D convolution for neighbor counting.

Each wave picks items with <4 neighbors. Like Tetris in reverse - clear
the edges first, work your way in. Scipy does the heavy lifting.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import convolve
from rich.console import Console
from rich.text import Text
from rich.padding import Padding
from rich.panel import Panel
from rich.console import Group


# Items with <4 neighbors are reachable. The warehouse elves hate crowds.
ACCESSIBILITY_THRESHOLD = 4

# 8-directional neighbor kernel - the 0 in the center means "don't count yourself"
# Convolve this bad boy and you've got neighbor counts for the whole grid
NEIGHBOR_KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

GRID_CELL_STYLES = {
    "x": "green",   # Picked items
    "@": "red",     # Remaining items
    ".": "dim"      # Empty positions
}

@dataclass
class WavePickResult:
    items_picked_count: int
    picked_mask: np.ndarray
    updated_grid: np.ndarray

class DisplayVerbosity(Enum):
    FULL = "full"         # Every wave
    BOOKENDS = "bookends" # First + last wave
    SUMMARY = "summary"   # Just the numbers

@dataclass
class WaveDisplayData:
    wave_number: int
    items_picked_count: int
    remaining_accessible_count: int
    picked_mask: np.ndarray
    grid_before_pick: np.ndarray

@dataclass
class SimulationResult:
    total_items_picked: int
    wave_picks: list[int]
    wave_data: list[WaveDisplayData]

def load_inventory_grid(file_path: str) -> np.ndarray:
    """
    Load warehouse layout. '@' = item, everything else = empty.

    Args:
        file_path: Path to layout file (relative to this module)

    Returns:
        Binary grid where 1 = item, 0 = empty
    """
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    df = pd.read_csv(full_path, header=None)
    grid = df[0].apply(list).to_list()
    grid = np.array(grid)
    return (grid == '@').astype(int)

def calculate_accessible_positions(grid: np.ndarray) -> np.ndarray:
    """
    Find items the elves can reach - fewer than 4 neighbors means accessible.

    Args:
        grid: Binary grid (1 = item, 0 = empty)

    Returns:
        Boolean mask where True = reachable this wave
    """
    # Convolve once, get neighbor counts everywhere. No loops needed.
    neighbor_counts = convolve(grid, NEIGHBOR_KERNEL, mode='constant', cval=0)
    return (grid == 1) & (neighbor_counts < ACCESSIBILITY_THRESHOLD)

def render_grid_rows(picked_mask: np.ndarray, grid_before_pick: np.ndarray) -> list[Text]:
    """
    Turn grid state into pretty colored output.

    Args:
        picked_mask: Where we grabbed items this wave
        grid_before_pick: Grid state before picking

    Returns:
        Rich Text rows ready for display
    """
    display_grid = np.where(picked_mask, 'x', np.where(grid_before_pick == 1, '@', '.'))
    styled_rows = []
    for row in display_grid:
        row_text = Text()
        for cell_char in row:
            row_text.append(cell_char, style=GRID_CELL_STYLES.get(cell_char, ""))
        styled_rows.append(row_text)
    return styled_rows

def display_wave_result(console: Console, wave_number: int, items_picked_count: int,
                        remaining_accessible_count: int, picked_mask: np.ndarray,
                        grid_before_pick: np.ndarray):
    """
    Show what happened in a single wave - grid state plus stats.

    Args:
        console: Rich console for output
        wave_number: Which wave (1-indexed)
        items_picked_count: Items grabbed this wave
        remaining_accessible_count: What's exposed for next wave
        picked_mask: Where we picked
        grid_before_pick: Grid state before picking
    """
    styled_rows = render_grid_rows(picked_mask, grid_before_pick)
    console.print(
        Padding(
            Panel(Group(*styled_rows,
                        Text(f"\nPicked     : {items_picked_count}\n", style="yellow"),
                        Text(f"Accessible : {remaining_accessible_count}\n", style="cyan")),
                  title=f"Wave #{wave_number}",
                  border_style="blue",
                  expand=False,
                  padding=(1, 1, 1, 1)),
            (0, 0, 0, 4))
        )
    


def perform_wave_pick(current_grid: np.ndarray) -> WavePickResult:
    """
    Execute a single wave pick using convolution for neighbor counting.

    Items are accessible if they have fewer than ACCESSIBILITY_THRESHOLD neighbors.
    Like peeling an onion - we clear the exposed edges each wave.

    Args:
        current_grid: Binary grid (1 = item, 0 = empty)

    Returns:
        WavePickResult with count, picked mask, and updated grid
    """
    neighbor_counts = convolve(current_grid, NEIGHBOR_KERNEL, mode='constant', cval=0)
    is_accessible = (current_grid == 1) & (neighbor_counts < ACCESSIBILITY_THRESHOLD)

    if is_accessible.sum() == 0:
        return WavePickResult(0, np.zeros_like(current_grid, dtype=bool), current_grid)

    # Yoink! Remove picked items from the grid
    updated_grid = current_grid * ~is_accessible
    return WavePickResult(
        items_picked_count=is_accessible.sum(),
        picked_mask=current_grid & ~updated_grid,
        updated_grid=updated_grid
    )

def execute_simulation(file_path: str) -> SimulationResult:
    """
    Run the wave picking simulation until the warehouse is empty.

    Args:
        file_path: Path to inventory layout file

    Returns:
        SimulationResult with totals and per-wave data
    """
    inventory_grid = load_inventory_grid(file_path)
    total_items_picked = 0
    wave_number = 1
    wave_picks: list[int] = []
    wave_data: list[WaveDisplayData] = []

    # Keep picking waves until nothing's reachable anymore
    while True:
        result = perform_wave_pick(inventory_grid)
        total_items_picked += result.items_picked_count
        wave_picks.append(result.items_picked_count)

        accessible_positions = calculate_accessible_positions(result.updated_grid)
        remaining_accessible_count = accessible_positions.sum()

        wave_data.append(WaveDisplayData(
            wave_number=wave_number,
            items_picked_count=result.items_picked_count,
            remaining_accessible_count=remaining_accessible_count,
            picked_mask=result.picked_mask.copy(),
            grid_before_pick=inventory_grid.copy()
        ))

        if remaining_accessible_count == 0:
            break

        inventory_grid = result.updated_grid
        wave_number += 1

    return SimulationResult(total_items_picked, wave_picks, wave_data)

def display_summary_panel(console: Console, wave_picks: list[int], total_items_picked: int):
    """
    Show the final scorecard.

    Args:
        console: Rich console for output
        wave_picks: Items picked each wave
        total_items_picked: Grand total
    """
    console.print(Padding(
        Panel(
            Group(
                Text(f"Waves          : {len(wave_picks)}\n", style="cyan"),
                Text(f"Picks per wave : {', '.join(str(p) for p in wave_picks)}\n", style="yellow"),
                Text(f"Total picked   : {total_items_picked}\n", style="bold red")
            ),
            title="Summary",
            border_style="blue",
            expand=False,
            padding=(1, 1, 1, 1)
        ),
        (0, 0, 0, 4)
    ))

def display_simulation_results(console: Console, verbosity: DisplayVerbosity,
                               result: SimulationResult):
    """
    Show simulation results at the requested detail level.

    Args:
        console: Rich console for output
        verbosity: FULL = every wave, BOOKENDS = first/last, SUMMARY = just totals
        result: Complete simulation results
    """
    console.print(Padding("\n[bold]Puzzle Summary[/bold]", (0, 0, 1, 4)))

    if verbosity == DisplayVerbosity.FULL:
        for wave in result.wave_data:
            display_wave_result(console, wave.wave_number, wave.items_picked_count,
                                wave.remaining_accessible_count, wave.picked_mask,
                                wave.grid_before_pick)
        console.print(Padding(
            f"[bold red]Total Picked[/bold red]: {result.total_items_picked}\n",
            (0, 0, 0, 4)
        ))
    elif verbosity == DisplayVerbosity.BOOKENDS and len(result.wave_data) > 0:
        first_wave = result.wave_data[0]
        last_wave = result.wave_data[-1]
        display_wave_result(console, first_wave.wave_number, first_wave.items_picked_count,
                            first_wave.remaining_accessible_count, first_wave.picked_mask,
                            first_wave.grid_before_pick)
        if last_wave.wave_number != first_wave.wave_number:
            display_wave_result(console, last_wave.wave_number, last_wave.items_picked_count,
                                last_wave.remaining_accessible_count, last_wave.picked_mask,
                                last_wave.grid_before_pick)
        display_summary_panel(console, result.wave_picks, result.total_items_picked)
    else:
        display_summary_panel(console, result.wave_picks, result.total_items_picked)

if __name__ == "__main__":
    console = Console()
    result = execute_simulation('data.dat')
    display_simulation_results(console, DisplayVerbosity.BOOKENDS, result)



