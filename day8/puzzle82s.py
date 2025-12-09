"""Junction box circuits - strategy comparison.

Comparing scipy hierarchical clustering vs manual Union-Find.
Spoiler: scipy's linkage is basically Union-Find with a PhD.
"""
from rich.console import Console

from strategies import (
    load_junction_boxes,
    hierarchical,
    union_find,
    sparse_graph,
    kdtree,
    pytorch_gpu,
)
from benchmark import run_benchmarks


coords = load_junction_boxes('data.dat')
k = 1000

console = Console()
run_benchmarks(
    console,
    coords,
    strategies={
        'Hierarchical (scipy)': hierarchical,
        'Union-Find (manual)': union_find,
        'Sparse Graph (csgraph)': sparse_graph,
        'KD-Tree + csgraph': kdtree,
    },
    k=k,
    pytorch_strategy=pytorch_gpu,
)
