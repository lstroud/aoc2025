"""
Warehouse Picking Optimization - Exploring Multi-Picker Routing

Building on puzzle42's wave picking simulation, this module explores optimization
when picker travel time matters. We compare several approaches:

1. Greedy (Nearest Neighbor) - Simple, fast, decent results
2. Clustering (K-Means) - Assign pickers to regions, then route within
3. Linear Assignment - Optimal one-to-one matching for single picks
4. Simulated Annealing - Metaheuristic for route optimization

Problem: Given N pickers at starting positions and M accessible items,
minimize total time to pick all items (travel + pick time).
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.padding import Padding

from puzzle42 import load_inventory_grid, calculate_accessible_positions, NEIGHBOR_KERNEL


@dataclass
class PickerState:
    """Current state of a picker."""
    picker_id: int
    position: tuple[int, int]
    items_picked: int
    total_distance: float


@dataclass
class PickingResult:
    """Result of a picking optimization run."""
    strategy_name: str
    total_distance: float
    max_picker_distance: float  # Bottleneck picker determines wave time
    assignments: list[list[tuple[int, int]]]  # Route per picker


def get_accessible_positions(grid: np.ndarray) -> np.ndarray:
    """Get (row, col) coordinates of all accessible items."""
    accessible_mask = calculate_accessible_positions(grid)
    positions = np.argwhere(accessible_mask)
    return positions


def euclidean_distances(positions: np.ndarray, picker_positions: np.ndarray) -> np.ndarray:
    """Compute distance matrix between items and pickers."""
    return cdist(positions, picker_positions, metric='euclidean')


def manhattan_distances(positions: np.ndarray, picker_positions: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance matrix (more realistic for grid movement)."""
    return cdist(positions, picker_positions, metric='cityblock')


# =============================================================================
# Strategy 1: Greedy Nearest Neighbor
# =============================================================================

def greedy_nearest_neighbor(
    item_positions: np.ndarray,
    picker_starts: np.ndarray,
    distance_fn=manhattan_distances
) -> PickingResult:
    """
    Each picker repeatedly picks the nearest unassigned item.

    Simple O(n*m) greedy approach. Fast but can create unbalanced workloads.
    """
    n_pickers = len(picker_starts)
    n_items = len(item_positions)

    picker_positions = picker_starts.copy().astype(float)
    picker_distances = np.zeros(n_pickers)
    assignments = [[] for _ in range(n_pickers)]
    picked = np.zeros(n_items, dtype=bool)

    while not picked.all():
        # Find nearest unpicked item for each picker
        unpicked_indices = np.where(~picked)[0]
        unpicked_items = item_positions[unpicked_indices]

        distances = distance_fn(unpicked_items, picker_positions)

        # Assign items greedily - picker with shortest distance to any item goes first
        for _ in range(min(n_pickers, len(unpicked_indices))):
            if picked.all():
                break

            # Find global minimum (which picker-item pair is closest)
            min_idx = np.unravel_index(distances.argmin(), distances.shape)
            item_local_idx, picker_idx = min_idx
            item_global_idx = unpicked_indices[item_local_idx]

            # Assign item to picker
            item_pos = tuple(item_positions[item_global_idx])
            assignments[picker_idx].append(item_pos)
            picker_distances[picker_idx] += distances[item_local_idx, picker_idx]
            picker_positions[picker_idx] = item_positions[item_global_idx]
            picked[item_global_idx] = True

            # Mark this item as unavailable (infinite distance)
            distances[item_local_idx, :] = np.inf

    return PickingResult(
        strategy_name="Greedy Nearest",
        total_distance=picker_distances.sum(),
        max_picker_distance=picker_distances.max(),
        assignments=assignments
    )


# =============================================================================
# Strategy 2: Clustering + Local Routing
# =============================================================================

def cluster_then_route(
    item_positions: np.ndarray,
    picker_starts: np.ndarray,
    distance_fn=manhattan_distances
) -> PickingResult:
    """
    Use K-Means to assign items to picker regions, then route within each.

    Better load balancing than pure greedy. Two-phase approach:
    1. Cluster items into N groups (one per picker)
    2. Route within each cluster using nearest neighbor
    """
    n_pickers = len(picker_starts)
    n_items = len(item_positions)

    if n_items <= n_pickers:
        # Fewer items than pickers - use linear assignment
        return linear_assignment_strategy(item_positions, picker_starts, distance_fn)

    # Phase 1: Cluster items
    kmeans = KMeans(n_clusters=n_pickers, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(item_positions)

    # Assign clusters to pickers based on centroid proximity
    centroids = kmeans.cluster_centers_
    cluster_to_picker_distances = distance_fn(centroids, picker_starts)
    _, picker_assignment = linear_sum_assignment(cluster_to_picker_distances)

    # Phase 2: Route within each cluster
    assignments = [[] for _ in range(n_pickers)]
    picker_distances = np.zeros(n_pickers)

    for cluster_id in range(n_pickers):
        picker_id = picker_assignment[cluster_id]
        cluster_items = item_positions[cluster_labels == cluster_id]

        if len(cluster_items) == 0:
            continue

        # Nearest neighbor within cluster
        current_pos = picker_starts[picker_id].astype(float)
        remaining = list(range(len(cluster_items)))

        while remaining:
            distances = distance_fn(
                cluster_items[remaining],
                current_pos.reshape(1, -1)
            ).flatten()
            nearest_idx = remaining[distances.argmin()]

            item_pos = tuple(cluster_items[nearest_idx])
            assignments[picker_id].append(item_pos)
            picker_distances[picker_id] += distances.min()
            current_pos = cluster_items[nearest_idx]
            remaining.remove(nearest_idx)

    return PickingResult(
        strategy_name="Cluster + Route",
        total_distance=picker_distances.sum(),
        max_picker_distance=picker_distances.max(),
        assignments=assignments
    )


# =============================================================================
# Strategy 3: Linear Assignment (Hungarian Algorithm)
# =============================================================================

def linear_assignment_strategy(
    item_positions: np.ndarray,
    picker_starts: np.ndarray,
    distance_fn=manhattan_distances
) -> PickingResult:
    """
    Optimal one-to-one assignment using Hungarian algorithm.

    Only works when n_items <= n_pickers (one item per picker max).
    For multiple items, this gives optimal FIRST pick assignment.
    """
    n_pickers = len(picker_starts)
    n_items = len(item_positions)

    # Build cost matrix
    cost_matrix = distance_fn(item_positions, picker_starts).T  # pickers x items

    # Solve assignment problem
    picker_indices, item_indices = linear_sum_assignment(cost_matrix)

    assignments = [[] for _ in range(n_pickers)]
    picker_distances = np.zeros(n_pickers)

    for picker_idx, item_idx in zip(picker_indices, item_indices):
        if item_idx < n_items:  # Valid assignment
            item_pos = tuple(item_positions[item_idx])
            assignments[picker_idx].append(item_pos)
            picker_distances[picker_idx] = cost_matrix[picker_idx, item_idx]

    return PickingResult(
        strategy_name="Linear Assignment",
        total_distance=picker_distances.sum(),
        max_picker_distance=picker_distances.max(),
        assignments=assignments
    )


# =============================================================================
# Strategy 4: Simulated Annealing for Route Optimization
# =============================================================================

def simulated_annealing(
    item_positions: np.ndarray,
    picker_starts: np.ndarray,
    distance_fn=manhattan_distances,
    initial_temp: float = 1000.0,
    cooling_rate: float = 0.995,
    iterations: int = 10000
) -> PickingResult:
    """
    Metaheuristic optimization starting from greedy solution.

    Explores solution space by swapping items between pickers,
    accepting worse solutions with decreasing probability.
    """
    n_pickers = len(picker_starts)
    n_items = len(item_positions)

    # Start with greedy solution
    initial = greedy_nearest_neighbor(item_positions, picker_starts, distance_fn)

    # Convert to mutable assignment array
    item_to_picker = np.zeros(n_items, dtype=int)
    for picker_id, route in enumerate(initial.assignments):
        for item_pos in route:
            item_idx = np.where((item_positions == item_pos).all(axis=1))[0][0]
            item_to_picker[item_idx] = picker_id

    def calculate_cost(assignment):
        """Calculate max picker distance (bottleneck)."""
        picker_distances = np.zeros(n_pickers)
        for picker_id in range(n_pickers):
            picker_items = item_positions[assignment == picker_id]
            if len(picker_items) == 0:
                continue
            # Simple approximation: sum of distances from start through all items
            current = picker_starts[picker_id]
            for item in picker_items:
                picker_distances[picker_id] += np.abs(item - current).sum()
                current = item
        return picker_distances.max()

    current_assignment = item_to_picker.copy()
    current_cost = calculate_cost(current_assignment)
    best_assignment = current_assignment.copy()
    best_cost = current_cost

    temp = initial_temp

    for _ in range(iterations):
        # Random swap: move one item to different picker
        item_idx = np.random.randint(n_items)
        old_picker = current_assignment[item_idx]
        new_picker = np.random.randint(n_pickers)

        if new_picker == old_picker:
            continue

        # Try the swap
        current_assignment[item_idx] = new_picker
        new_cost = calculate_cost(current_assignment)

        # Accept or reject
        delta = new_cost - current_cost
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_assignment = current_assignment.copy()
        else:
            current_assignment[item_idx] = old_picker  # Revert

        temp *= cooling_rate

    # Reconstruct assignments from best solution
    assignments = [[] for _ in range(n_pickers)]
    for item_idx, picker_id in enumerate(best_assignment):
        assignments[picker_id].append(tuple(item_positions[item_idx]))

    # Recalculate actual distances
    picker_distances = np.zeros(n_pickers)
    for picker_id in range(n_pickers):
        current = picker_starts[picker_id].astype(float)
        for item_pos in assignments[picker_id]:
            picker_distances[picker_id] += np.abs(np.array(item_pos) - current).sum()
            current = np.array(item_pos)

    return PickingResult(
        strategy_name="Simulated Annealing",
        total_distance=picker_distances.sum(),
        max_picker_distance=picker_distances.max(),
        assignments=assignments
    )


# =============================================================================
# Comparison Runner
# =============================================================================

def compare_strategies(
    grid: np.ndarray,
    n_pickers: int = 4,
    picker_edge: str = "left"
) -> list[PickingResult]:
    """
    Run all strategies on the same problem and compare results.

    Args:
        grid: Binary warehouse grid
        n_pickers: Number of pickers
        picker_edge: Where pickers start ("left", "top", "corners")
    """
    item_positions = get_accessible_positions(grid)
    n_items = len(item_positions)

    if n_items == 0:
        return []

    # Set picker starting positions
    rows, cols = grid.shape
    if picker_edge == "left":
        picker_starts = np.array([
            [i * rows // n_pickers, 0] for i in range(n_pickers)
        ])
    elif picker_edge == "top":
        picker_starts = np.array([
            [0, i * cols // n_pickers] for i in range(n_pickers)
        ])
    elif picker_edge == "corners":
        picker_starts = np.array([
            [0, 0], [0, cols-1], [rows-1, 0], [rows-1, cols-1]
        ])[:n_pickers]
    else:
        picker_starts = np.array([[0, 0]] * n_pickers)

    results = []

    # Run each strategy
    results.append(greedy_nearest_neighbor(item_positions, picker_starts))
    results.append(cluster_then_route(item_positions, picker_starts))

    if n_items <= n_pickers:
        results.append(linear_assignment_strategy(item_positions, picker_starts))

    results.append(simulated_annealing(item_positions, picker_starts, iterations=5000))

    return results


def display_comparison(console: Console, results: list[PickingResult], n_items: int):
    """Display comparison table of strategy results."""
    table = Table(title=f"Picking Optimization Comparison ({n_items} items)")
    table.add_column("Strategy", style="cyan")
    table.add_column("Total Distance", justify="right", style="yellow")
    table.add_column("Max Picker Dist", justify="right", style="magenta")
    table.add_column("Load Balance", justify="right", style="green")

    for r in results:
        # Calculate load balance (lower is better, 1.0 = perfect)
        distances = [sum(np.abs(np.diff(route, axis=0)).sum() if len(route) > 1 else 0
                        for route in [r.assignments[i]])
                    for i in range(len(r.assignments))]
        items_per_picker = [len(route) for route in r.assignments]
        balance = max(items_per_picker) / max(1, np.mean(items_per_picker)) if items_per_picker else 1.0

        table.add_row(
            r.strategy_name,
            f"{r.total_distance:.1f}",
            f"{r.max_picker_distance:.1f}",
            f"{balance:.2f}x"
        )

    console.print(Padding(table, (1, 0, 1, 4)))


if __name__ == "__main__":
    console = Console()
    console.print(Padding("\n[bold]Warehouse Picking Optimization[/bold]", (0, 0, 1, 4)))

    # Load grid and find accessible items
    grid = load_inventory_grid('data.dat')
    binary_grid = (grid == '@').astype(int) if grid.dtype.kind in ['U', 'S'] else grid

    item_positions = get_accessible_positions(binary_grid)
    console.print(Padding(f"[cyan]Accessible items:[/cyan] {len(item_positions)}", (0, 0, 0, 4)))
    console.print(Padding(f"[cyan]Grid size:[/cyan] {binary_grid.shape}", (0, 0, 1, 4)))

    # Compare with different picker counts
    for n_pickers in [2, 4, 8]:
        console.print(Padding(f"[bold yellow]{n_pickers} Pickers[/bold yellow]", (0, 0, 0, 4)))
        results = compare_strategies(binary_grid, n_pickers=n_pickers, picker_edge="left")
        display_comparison(console, results, len(item_positions))
