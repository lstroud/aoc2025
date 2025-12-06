"""
Collapse Picking - Order-Dependent Item Recovery

A variant of puzzle42 where items need structural support from neighbors.
When an item loses too many neighbors, it collapses and becomes unrecoverable.

The challenge: Maximize items picked before cascade collapses destroy them.

Key insight: Same convolution machinery, but with inverted logic:
- Puzzle42: accessible if neighbors < 4 (edge items pickable)
- Collapse: unstable if neighbors < K (items need support)

This creates an order-dependent optimization where greedy can be catastrophic.
"""
import time
import tracemalloc
import numpy as np
from scipy.ndimage import convolve
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.padding import Padding
from rich.panel import Panel
from rich.console import Group
from rich.text import Text

from puzzle42 import load_inventory_grid, NEIGHBOR_KERNEL


@dataclass
class CollapseResult:
    """Result of a collapse picking simulation."""
    strategy_name: str
    items_picked: int
    items_collapsed: int
    pick_order: list[tuple[int, int]]
    elapsed_ms: float = 0.0
    peak_memory_kb: float = 0.0


def get_neighbor_counts(grid: np.ndarray) -> np.ndarray:
    """Calculate neighbor count for each cell using convolution."""
    return convolve(grid, NEIGHBOR_KERNEL, mode='constant', cval=0)


def find_accessible(grid: np.ndarray, access_threshold: int = 4) -> np.ndarray:
    """Find items that can be picked (fewer than threshold neighbors)."""
    neighbor_counts = get_neighbor_counts(grid)
    return (grid == 1) & (neighbor_counts < access_threshold)


def find_unstable(grid: np.ndarray, stability_threshold: int) -> np.ndarray:
    """Find items that will collapse (fewer than threshold neighbors)."""
    neighbor_counts = get_neighbor_counts(grid)
    return (grid == 1) & (neighbor_counts < stability_threshold)


def apply_cascading_collapse(grid: np.ndarray, stability_threshold: int) -> tuple[np.ndarray, int]:
    """
    Remove unstable items until grid stabilizes.

    Returns updated grid and count of collapsed items.
    """
    total_collapsed = 0
    current_grid = grid.copy()

    while True:
        unstable = find_unstable(current_grid, stability_threshold)
        collapse_count = unstable.sum()

        if collapse_count == 0:
            break

        total_collapsed += collapse_count
        current_grid = current_grid * ~unstable

    return current_grid, total_collapsed


def pick_item(grid: np.ndarray, position: tuple[int, int]) -> np.ndarray:
    """Remove a single item from the grid."""
    new_grid = grid.copy()
    new_grid[position] = 0
    return new_grid


# =============================================================================
# Strategy 1: Random Order (Baseline)
# =============================================================================

def random_order_picking(
    grid: np.ndarray,
    stability_threshold: int,
    access_threshold: int = 4,
    seed: int = 42
) -> CollapseResult:
    """
    Pick accessible items in random order.

    Baseline strategy - demonstrates how bad random can be.
    """
    np.random.seed(seed)
    current_grid = grid.copy()
    items_picked = 0
    total_collapsed = 0
    pick_order = []

    while True:
        accessible = find_accessible(current_grid, access_threshold)
        accessible_positions = np.argwhere(accessible)

        if len(accessible_positions) == 0:
            break

        # Pick random accessible item
        idx = np.random.randint(len(accessible_positions))
        pos = tuple(accessible_positions[idx])

        current_grid = pick_item(current_grid, pos)
        items_picked += 1
        pick_order.append(pos)

        # Apply cascade collapses
        current_grid, collapsed = apply_cascading_collapse(current_grid, stability_threshold)
        total_collapsed += collapsed

    return CollapseResult(
        strategy_name="Random Order",
        items_picked=items_picked,
        items_collapsed=total_collapsed,
        pick_order=pick_order
    )


# =============================================================================
# Strategy 2: Most Neighbors First (Interior → Exterior)
# =============================================================================

def most_neighbors_first(
    grid: np.ndarray,
    stability_threshold: int,
    access_threshold: int = 4
) -> CollapseResult:
    """
    Pick accessible items with most neighbors first.

    Intuition: Interior items are well-supported, removing them won't
    cause cascades. Work from inside out.
    """
    current_grid = grid.copy()
    items_picked = 0
    total_collapsed = 0
    pick_order = []

    while True:
        accessible = find_accessible(current_grid, access_threshold)
        accessible_positions = np.argwhere(accessible)

        if len(accessible_positions) == 0:
            break

        # Get neighbor counts for accessible items
        neighbor_counts = get_neighbor_counts(current_grid)
        accessible_neighbors = [neighbor_counts[tuple(p)] for p in accessible_positions]

        # Pick item with most neighbors
        best_idx = np.argmax(accessible_neighbors)
        pos = tuple(accessible_positions[best_idx])

        current_grid = pick_item(current_grid, pos)
        items_picked += 1
        pick_order.append(pos)

        current_grid, collapsed = apply_cascading_collapse(current_grid, stability_threshold)
        total_collapsed += collapsed

    return CollapseResult(
        strategy_name="Most Neighbors First",
        items_picked=items_picked,
        items_collapsed=total_collapsed,
        pick_order=pick_order
    )


# =============================================================================
# Strategy 3: Fewest Neighbors First (Exterior → Interior)
# =============================================================================

def fewest_neighbors_first(
    grid: np.ndarray,
    stability_threshold: int,
    access_threshold: int = 4
) -> CollapseResult:
    """
    Pick accessible items with fewest neighbors first.

    Intuition: Edge items have few neighbors, removing them doesn't
    destabilize much. Like puzzle42's natural wave pattern.
    """
    current_grid = grid.copy()
    items_picked = 0
    total_collapsed = 0
    pick_order = []

    while True:
        accessible = find_accessible(current_grid, access_threshold)
        accessible_positions = np.argwhere(accessible)

        if len(accessible_positions) == 0:
            break

        neighbor_counts = get_neighbor_counts(current_grid)
        accessible_neighbors = [neighbor_counts[tuple(p)] for p in accessible_positions]

        # Pick item with fewest neighbors
        best_idx = np.argmin(accessible_neighbors)
        pos = tuple(accessible_positions[best_idx])

        current_grid = pick_item(current_grid, pos)
        items_picked += 1
        pick_order.append(pos)

        current_grid, collapsed = apply_cascading_collapse(current_grid, stability_threshold)
        total_collapsed += collapsed

    return CollapseResult(
        strategy_name="Fewest Neighbors First",
        items_picked=items_picked,
        items_collapsed=total_collapsed,
        pick_order=pick_order
    )


# =============================================================================
# Strategy 4: Stability Margin (Pick items with most buffer)
# =============================================================================

def stability_margin_picking(
    grid: np.ndarray,
    stability_threshold: int,
    access_threshold: int = 4
) -> CollapseResult:
    """
    Pick accessible items with highest stability margin first.

    Stability margin = neighbor_count - stability_threshold
    Items with higher margin have more "buffer" before they'd collapse.
    O(n) - just sorting by a different criterion.
    """
    current_grid = grid.copy()
    items_picked = 0
    total_collapsed = 0
    pick_order = []

    while True:
        accessible = find_accessible(current_grid, access_threshold)
        accessible_positions = np.argwhere(accessible)

        if len(accessible_positions) == 0:
            break

        neighbor_counts = get_neighbor_counts(current_grid)
        # Stability margin: how far above the collapse threshold
        margins = [neighbor_counts[tuple(p)] - stability_threshold for p in accessible_positions]

        # Pick item with highest margin (most buffer)
        best_idx = np.argmax(margins)
        pos = tuple(accessible_positions[best_idx])

        current_grid = pick_item(current_grid, pos)
        items_picked += 1
        pick_order.append(pos)

        current_grid, collapsed = apply_cascading_collapse(current_grid, stability_threshold)
        total_collapsed += collapsed

    return CollapseResult(
        strategy_name="Stability Margin",
        items_picked=items_picked,
        items_collapsed=total_collapsed,
        pick_order=pick_order
    )


# =============================================================================
# Strategy 5: Batch Wave (Pick all safe items simultaneously)
# =============================================================================

def batch_wave_picking(
    grid: np.ndarray,
    stability_threshold: int,
    access_threshold: int = 4
) -> CollapseResult:
    """
    Pick all "safe" items in parallel waves.

    An item is safe to pick if it's accessible AND has enough neighbors
    that removing it won't immediately destabilize its neighbors.
    This mimics puzzle42's wave pattern but with stability awareness.

    O(n) - uses vectorized operations per wave.
    """
    current_grid = grid.copy()
    items_picked = 0
    total_collapsed = 0
    pick_order = []

    while True:
        accessible = find_accessible(current_grid, access_threshold)
        if accessible.sum() == 0:
            break

        neighbor_counts = get_neighbor_counts(current_grid)

        # Safe items: accessible AND have enough neighbors that picking won't cascade
        # An item is "safe" if its neighbors will still have >= stability_threshold
        # after it's removed. Conservative heuristic: pick items with high neighbor counts.
        safe_margin = stability_threshold + 1
        safe_to_pick = accessible & (neighbor_counts >= safe_margin)

        if safe_to_pick.sum() == 0:
            # No safe items - fall back to picking one with highest margin
            accessible_positions = np.argwhere(accessible)
            margins = [neighbor_counts[tuple(p)] for p in accessible_positions]
            best_idx = np.argmax(margins)
            pos = tuple(accessible_positions[best_idx])

            current_grid = pick_item(current_grid, pos)
            items_picked += 1
            pick_order.append(pos)
        else:
            # Pick all safe items in this wave
            picked_positions = np.argwhere(safe_to_pick)
            for pos in picked_positions:
                pick_order.append(tuple(pos))
            items_picked += len(picked_positions)
            current_grid = current_grid * ~safe_to_pick

        current_grid, collapsed = apply_cascading_collapse(current_grid, stability_threshold)
        total_collapsed += collapsed

    return CollapseResult(
        strategy_name="Batch Wave",
        items_picked=items_picked,
        items_collapsed=total_collapsed,
        pick_order=pick_order
    )


# =============================================================================
# Strategy 6: Cascade-Aware (Simulate Before Picking)
# =============================================================================

def cascade_aware_picking(
    grid: np.ndarray,
    stability_threshold: int,
    access_threshold: int = 4
) -> CollapseResult:
    """
    Simulate each possible pick and choose the one with minimal cascade.

    O(n²) but optimal at each step - true greedy with lookahead.
    """
    current_grid = grid.copy()
    items_picked = 0
    total_collapsed = 0
    pick_order = []

    while True:
        accessible = find_accessible(current_grid, access_threshold)
        accessible_positions = np.argwhere(accessible)

        if len(accessible_positions) == 0:
            break

        # Simulate each pick and find one with minimum cascade
        best_pos = None
        min_cascade = float('inf')

        for pos in accessible_positions:
            pos = tuple(pos)
            test_grid = pick_item(current_grid, pos)
            _, cascade = apply_cascading_collapse(test_grid, stability_threshold)

            if cascade < min_cascade:
                min_cascade = cascade
                best_pos = pos

        current_grid = pick_item(current_grid, best_pos)
        items_picked += 1
        pick_order.append(best_pos)

        current_grid, collapsed = apply_cascading_collapse(current_grid, stability_threshold)
        total_collapsed += collapsed

    return CollapseResult(
        strategy_name="Cascade-Aware",
        items_picked=items_picked,
        items_collapsed=total_collapsed,
        pick_order=pick_order
    )


# =============================================================================
# Strategy 5: Maximize Remaining (Greedy by Grid Size)
# =============================================================================

def maximize_remaining(
    grid: np.ndarray,
    stability_threshold: int,
    access_threshold: int = 4
) -> CollapseResult:
    """
    Pick the item that leaves the most items remaining after cascade.

    Different from cascade-aware: considers that cascaded items
    might have been unreachable anyway.
    """
    current_grid = grid.copy()
    items_picked = 0
    total_collapsed = 0
    pick_order = []

    while True:
        accessible = find_accessible(current_grid, access_threshold)
        accessible_positions = np.argwhere(accessible)

        if len(accessible_positions) == 0:
            break

        best_pos = None
        max_remaining = -1

        for pos in accessible_positions:
            pos = tuple(pos)
            test_grid = pick_item(current_grid, pos)
            stable_grid, _ = apply_cascading_collapse(test_grid, stability_threshold)
            remaining = stable_grid.sum()

            if remaining > max_remaining:
                max_remaining = remaining
                best_pos = pos

        current_grid = pick_item(current_grid, best_pos)
        items_picked += 1
        pick_order.append(best_pos)

        current_grid, collapsed = apply_cascading_collapse(current_grid, stability_threshold)
        total_collapsed += collapsed

    return CollapseResult(
        strategy_name="Maximize Remaining",
        items_picked=items_picked,
        items_collapsed=total_collapsed,
        pick_order=pick_order
    )


# =============================================================================
# Comparison Runner
# =============================================================================

def timed_run(strategy_fn, *args, **kwargs) -> CollapseResult:
    """Run a strategy function and record elapsed time and memory usage."""
    tracemalloc.start()
    start = time.perf_counter()

    result = strategy_fn(*args, **kwargs)

    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result.elapsed_ms = elapsed_ms
    result.peak_memory_kb = peak_memory / 1024
    return result


def compare_strategies(
    grid: np.ndarray,
    stability_threshold: int,
    access_threshold: int = 4,
    max_items_for_expensive: int = 500
) -> list[CollapseResult]:
    """
    Run all strategies and compare results.

    Args:
        max_items_for_expensive: Skip O(n²) strategies if grid has more items than this
    """
    results = []
    total_items = grid.sum()

    # O(n) strategies - always run
    results.append(timed_run(random_order_picking, grid, stability_threshold, access_threshold))
    results.append(timed_run(fewest_neighbors_first, grid, stability_threshold, access_threshold))
    results.append(timed_run(most_neighbors_first, grid, stability_threshold, access_threshold))
    results.append(timed_run(stability_margin_picking, grid, stability_threshold, access_threshold))
    results.append(timed_run(batch_wave_picking, grid, stability_threshold, access_threshold))

    # O(n²) strategies - skip on large grids
    if total_items <= max_items_for_expensive:
        results.append(timed_run(cascade_aware_picking, grid, stability_threshold, access_threshold))
        results.append(timed_run(maximize_remaining, grid, stability_threshold, access_threshold))

    return results


def display_comparison(console: Console, results: list[CollapseResult], total_items: int):
    """Display comparison table of strategy results."""
    table = Table(title=f"Collapse Picking Comparison ({total_items} total items)")
    table.add_column("Strategy", style="cyan")
    table.add_column("Picked", justify="right", style="green")
    table.add_column("Collapsed", justify="right", style="red")
    table.add_column("Recovery %", justify="right", style="yellow")
    table.add_column("Time (ms)", justify="right", style="magenta")
    table.add_column("Memory (KB)", justify="right", style="blue")

    for r in results:
        recovery_pct = (r.items_picked / total_items * 100) if total_items > 0 else 0
        table.add_row(
            r.strategy_name,
            str(r.items_picked),
            str(r.items_collapsed),
            f"{recovery_pct:.1f}%",
            f"{r.elapsed_ms:.1f}",
            f"{r.peak_memory_kb:.1f}"
        )

    console.print(Padding(table, (1, 0, 1, 4)))


def visualize_grid_state(
    console: Console,
    original_grid: np.ndarray,
    final_picked: list[tuple[int, int]],
    title: str
):
    """Show grid with picked items marked."""
    display = np.where(original_grid == 1, '@', '.')
    for pos in final_picked:
        display[pos] = 'x'

    rows = []
    for row in display:
        text = Text()
        for char in row:
            style = {"x": "green", "@": "red", ".": "dim"}.get(char, "")
            text.append(char, style=style)
        rows.append(text)

    console.print(Padding(
        Panel(Group(*rows), title=title, border_style="blue", expand=False, padding=(1, 1, 1, 1)),
        (0, 0, 0, 4)
    ))


if __name__ == "__main__":
    console = Console()
    console.print(Padding("\n[bold]Collapse Picking Optimization[/bold]", (0, 0, 1, 4)))

    # Load grid
    grid = load_inventory_grid('data.dat')
    binary_grid = (grid == '@').astype(int) if grid.dtype.kind in ['U', 'S'] else grid
    total_items = binary_grid.sum()

    console.print(Padding(f"[cyan]Total items:[/cyan] {total_items}", (0, 0, 0, 4)))
    console.print(Padding(f"[cyan]Grid size:[/cyan] {binary_grid.shape}", (0, 0, 1, 4)))

    # Compare with different stability thresholds
    for stability in [1, 2, 3]:
        console.print(Padding(
            f"[bold yellow]Stability Threshold: {stability}[/bold yellow] "
            f"(items collapse if neighbors < {stability})",
            (0, 0, 0, 4)
        ))

        results = compare_strategies(binary_grid, stability_threshold=stability)
        display_comparison(console, results, total_items)

        if total_items > 500:
            console.print(Padding(
                "[dim]Note: O(n²) strategies (Cascade-Aware, Maximize Remaining) skipped for large grid[/dim]",
                (0, 0, 0, 4)
            ))

        # Show best strategy's final state for threshold=2
        if stability == 2:
            best = max(results, key=lambda r: r.items_picked)
            visualize_grid_state(
                console, binary_grid, best.pick_order,
                f"Best Strategy: {best.strategy_name}"
            )
