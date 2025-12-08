"""Junction box circuits using single-linkage hierarchical clustering.

The elves could have used Union-Find, but scipy's clustering is basically
Union-Find with a PhD and better marketing.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from rich.console import Console


def read_junction_box_locations(file_name: str) -> np.ndarray:
    """Load junction box 3D coordinates from file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_name
    return np.loadtxt(full_path, delimiter=',', dtype=int)


# Each box suspended in 3D space, waiting to be wired up
junction_boxes = read_junction_box_locations('data.dat')

# O(n²) pairs, but pdist handles it like a champ
pairwise_distances = pdist(junction_boxes, metric='euclidean')

# Single-linkage: merges with whoever's closest
# The algorithm that treats social distancing as a suggestion
Z = linkage(pairwise_distances, method='single')

# Sort distances, grab the k-th smallest as our cutoff
sorted_distances = np.sort(pairwise_distances)
connection_attempts = 1000
distance_threshold = sorted_distances[connection_attempts - 1]

# Some connection attempts are redundant - the elves try to wire
# boxes that are already in the same circuit. Classic elf move.
actual_connections = np.sum(Z[:, 2] <= distance_threshold)
num_circuits = len(junction_boxes) - actual_connections

# fcluster: "here's which group you belong to, don't @ me"
circuit_labels = fcluster(Z, num_circuits, criterion='maxclust')
circuit_sizes = np.bincount(circuit_labels)

# Product of three largest - because one big circuit is never enough
largest_three = np.sort(circuit_sizes)[-3:]
answer = np.prod(largest_three)

# Results
console = Console()
console.print(f"\n[cyan]Junction boxes[/cyan]      : {len(junction_boxes):,}")
console.print(f"[cyan]Connection attempts[/cyan] : {connection_attempts:,}")
console.print(f"[dim]Redundant wiring[/dim]     : {connection_attempts - actual_connections:,} [dim italic](elves gonna elf)[/dim italic]")
console.print(f"[yellow]Actual merges[/yellow]       : {actual_connections:,}")
console.print(f"[yellow]Circuits formed[/yellow]     : {num_circuits:,}")
console.print(f"[green]Largest three[/green]       : {list(largest_three)}")
console.print(f"[bold green]Answer[/bold green]              : {answer:,}\n")
