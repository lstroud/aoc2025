"""Junction box circuits using single-linkage hierarchical clustering.

The elves could have used Union-Find, but scipy's clustering is basically
Union-Find with a PhD and better marketing.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def read_junction_box_locations(file_name: str) -> np.ndarray:
    """Load junction box 3D coordinates from file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_name
    return np.loadtxt(full_path, delimiter=',', dtype=int)


# Each box suspended in 3D space, waiting to be wired up
junction_boxes = read_junction_box_locations('sample.dat')
connection_attempts = 10

# O(n²) pairs, but pdist handles it like a champ
pairwise_distances = pdist(junction_boxes, metric='euclidean')

# Single-linkage: merges with whoever's closest
# The algorithm that treats social distancing as a suggestion
Z = linkage(pairwise_distances, method='single')

# Sort distances, grab the k-th smallest as our cutoff
sorted_distances = np.sort(pairwise_distances)
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

# The final string of lights that unites all circuits
final_string_distance = Z[-1, 2]
dist_matrix = squareform(pairwise_distances)
i, j = np.where(dist_matrix == final_string_distance)
box_a = junction_boxes[i[0]]
box_b = junction_boxes[j[0]]
cable_length = box_a[0] * box_b[0]

# Results
console = Console()

part1 = Text()
part1.append(f"Junction boxes      : {len(junction_boxes):,}\n", style="green")
part1.append(f"Connection attempts : {connection_attempts:,}\n", style="green")
part1.append(f"Redundant wiring    : {connection_attempts - actual_connections:,} ", style="dim")
part1.append("(elves gonna elf) 🧝\n", style="dim italic")
part1.append(f"Actual merges       : {actual_connections:,}\n", style="yellow")
part1.append(f"Circuits formed     : {num_circuits:,}\n", style="yellow")
part1.append(f"Largest three       : {[int(x) for x in largest_three]}\n", style="red")
part1.append(f"🍬 Answer           : {int(answer):,}", style="bold green")

part2 = Text()
coord_a = tuple(int(x) for x in box_a)
coord_b = tuple(int(x) for x in box_b)
part2.append(f"Box A               : {coord_a}\n", style="red")
part2.append(f"Box B               : {coord_b}\n", style="red")
part2.append(f"🎁 Cable length     : {int(cable_length):,}", style="bold green")

console.print()
console.print(Panel(part1, title="🎄 Part 1: Three Largest Circuits 🎅", expand=False, border_style="red"))
console.print(Panel(part2, title="🦌 Part 2: One Circuit to Rule Them All ☃️ ", expand=False, border_style="green"))
console.print()