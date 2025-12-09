"""Junction box circuits - optimization variants.

The elves' greedy approach (connect k closest pairs) isn't optimal.
What if we wanted to minimize total wire length instead?

Fair comparisons:
1. Same connectivity goal: Greedy vs MST to reach 1 circuit
2. Same connection budget: Greedy k vs MST-pruned k
"""
import time
import tracemalloc
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from rich.console import Console
from rich.table import Table
from rich.padding import Padding

from strategies import load_junction_boxes


class UnionFind:
    """Union-Find for tracking components."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.num_components = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.num_components -= 1
        return True


def benchmark(name: str, func) -> dict:
    """Run a function and measure time and memory."""
    tracemalloc.start()
    start = time.perf_counter()

    result = func()

    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'name': name,
        'result': result,
        'time_ms': elapsed_ms,
        'memory_kb': peak_memory / 1024,
    }


def greedy_until_connected(dist_matrix: np.ndarray) -> dict:
    """Greedy: connect closest pairs until 1 circuit."""
    n = dist_matrix.shape[0]
    uf = UnionFind(n)

    # Get all pairs sorted by distance
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((dist_matrix[i, j], i, j))
    pairs.sort()

    total_wire = 0.0
    connections = 0

    for dist, i, j in pairs:
        if uf.union(i, j):
            total_wire += dist
            connections += 1
            if uf.num_components == 1:
                break

    return {'wire': total_wire, 'connections': connections, 'circuits': 1}


def greedy_k_connections(dist_matrix: np.ndarray, k: int) -> dict:
    """Greedy: connect k closest pairs."""
    n = dist_matrix.shape[0]
    uf = UnionFind(n)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((dist_matrix[i, j], i, j))
    pairs.sort()

    total_wire = 0.0
    for dist, i, j in pairs[:k]:
        uf.union(i, j)
        total_wire += dist

    return {'wire': total_wire, 'connections': k, 'circuits': uf.num_components}


def mst_full(dist_matrix: np.ndarray) -> dict:
    """MST: minimum wire to connect all boxes."""
    sparse_dist = csr_matrix(dist_matrix)
    mst = minimum_spanning_tree(sparse_dist)
    return {
        'wire': mst.sum(),
        'connections': mst.nnz,
        'circuits': 1
    }


def mst_k_connections(dist_matrix: np.ndarray, k: int) -> dict:
    """MST: use only k shortest edges from the MST."""
    sparse_dist = csr_matrix(dist_matrix)
    mst = minimum_spanning_tree(sparse_dist)

    # Extract and sort edges by distance (ascending)
    mst_coo = mst.tocoo()
    edges = sorted(zip(mst_coo.data, mst_coo.row, mst_coo.col))

    # Take k shortest MST edges
    edges_to_use = edges[:k]
    total_wire = sum(d for d, _, _ in edges_to_use)

    # Count resulting circuits
    n = dist_matrix.shape[0]
    uf = UnionFind(n)
    for d, i, j in edges_to_use:
        uf.union(i, j)

    return {
        'wire': total_wire,
        'connections': len(edges_to_use),
        'circuits': uf.num_components
    }


def mst_match_circuits(dist_matrix: np.ndarray, target_circuits: int) -> dict:
    """MST pruned to match a target number of circuits."""
    n = dist_matrix.shape[0]
    sparse_dist = csr_matrix(dist_matrix)
    mst = minimum_spanning_tree(sparse_dist)

    mst_coo = mst.tocoo()
    edges = sorted(zip(mst_coo.data, mst_coo.row, mst_coo.col), reverse=True)

    # Remove (target_circuits - 1) longest edges
    edges_to_keep = edges[target_circuits - 1:]

    total_wire = sum(d for d, _, _ in edges_to_keep)
    return {
        'wire': total_wire,
        'connections': len(edges_to_keep),
        'circuits': target_circuits
    }


# Load data
coords = load_junction_boxes('data.dat')
n = len(coords)
k = 1000

console = Console()
console.print(Padding(f"[cyan]Junction boxes:[/cyan] {n:,}", (1, 0, 0, 4)))
console.print(Padding(f"[cyan]Connection budget:[/cyan] {k:,}", (0, 0, 1, 4)))

# Compute distance matrix (shared cost)
distances = pdist(coords, metric='euclidean')
dist_matrix = squareform(distances)

# === Comparison 1: Same budget (k connections) ===
results1 = []

r = benchmark("Greedy k closest", lambda: greedy_k_connections(dist_matrix, k))
results1.append(r)

r = benchmark("MST k shortest", lambda: mst_k_connections(dist_matrix, k))
results1.append(r)

table1 = Table(title="Same Budget: k=1,000 connections")
table1.add_column("Approach", style="cyan")
table1.add_column("Total Wire", justify="right", style="green")
table1.add_column("Circuits", justify="right", style="magenta")
table1.add_column("Time (ms)", justify="right", style="yellow")
table1.add_column("Memory (KB)", justify="right", style="blue")

for r in results1:
    res = r['result']
    table1.add_row(
        r['name'],
        f"{res['wire']:,.0f}",
        f"{res['circuits']:,}",
        f"{r['time_ms']:.1f}",
        f"{r['memory_kb']:.0f}",
    )

console.print(Padding(table1, (0, 0, 1, 4)))

# === Comparison 2: Same goal (1 circuit) ===
results2 = []

r = benchmark("Greedy until connected", lambda: greedy_until_connected(dist_matrix))
results2.append(r)

r = benchmark("MST (optimal)", lambda: mst_full(dist_matrix))
results2.append(r)

table2 = Table(title="Same Goal: Connect ALL boxes (1 circuit)")
table2.add_column("Approach", style="cyan")
table2.add_column("Total Wire", justify="right", style="green")
table2.add_column("Connections", justify="right", style="yellow")
table2.add_column("Time (ms)", justify="right", style="magenta")
table2.add_column("Memory (KB)", justify="right", style="blue")

for r in results2:
    res = r['result']
    table2.add_row(
        r['name'],
        f"{res['wire']:,.0f}",
        f"{res['connections']:,}",
        f"{r['time_ms']:.1f}",
        f"{r['memory_kb']:.0f}",
    )

console.print(Padding(table2, (0, 0, 1, 4)))

# === Comparison 3: Match greedy's circuit count with optimal ===
greedy_result = greedy_k_connections(dist_matrix, k)
target_circuits = greedy_result['circuits']

results3 = []

r = benchmark(f"Greedy k={k}", lambda: greedy_k_connections(dist_matrix, k))
results3.append(r)

r = benchmark(f"MST → {target_circuits} circuits", lambda: mst_match_circuits(dist_matrix, target_circuits))
results3.append(r)

table3 = Table(title=f"Same Outcome: {target_circuits} circuits")
table3.add_column("Approach", style="cyan")
table3.add_column("Total Wire", justify="right", style="green")
table3.add_column("Connections", justify="right", style="yellow")
table3.add_column("Time (ms)", justify="right", style="magenta")

for r in results3:
    res = r['result']
    table3.add_row(
        r['name'],
        f"{res['wire']:,.0f}",
        f"{res['connections']:,}",
        f"{r['time_ms']:.1f}",
    )

console.print(Padding(table3, (0, 0, 1, 4)))

# Summary
greedy_wire = results3[0]['result']['wire']
optimal_wire = results3[1]['result']['wire']
savings = greedy_wire - optimal_wire
savings_pct = (savings / greedy_wire) * 100

console.print(Padding(
    f"[bold green]Wire savings with optimal planning:[/bold green] {savings:,.0f} ({savings_pct:.1f}%)",
    (0, 0, 1, 4)
))
