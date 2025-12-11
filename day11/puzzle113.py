"""Day 11: Plutonian Pebbles - Matrix Approach

Santa's reactor wiring needs path analysis! The elves have two questions:
  Part 1: How many signal paths from 'you' to 'out'?
  Part 2: How many paths from 'svr' to 'out' pass through BOTH 'dac' AND 'fft'?

This solution uses linear algebra instead of recursive DFS:

PART 1 - NEUMANN SERIES:
    Build adjacency matrix A where A[i,j] = 1 if device i connects to j.
    Matrix multiplication counts paths: A²[i,j] = number of 2-hop paths from i to j.
    Why? The dot product sums over all intermediates k: "i→k exists" × "k→j exists"

    For DAGs, we can sum ALL path lengths with the Neumann series:
        A + A² + A³ + ... = (I - A)⁻¹ - I

    This works because DAG matrices are nilpotent (A^n = 0 for some n),
    so the infinite series actually terminates.

PART 2 - INCLUSION-EXCLUSION WITH INTEGER VECTOR ITERATION:
    We need paths through BOTH checkpoints. Naive decomposition overcounts
    (paths visiting checkpoints multiple times get counted multiple times).

    Set theory to the rescue! Let:
        |¬fft| = paths NOT through fft = paths when fft removed from graph
        |¬dac| = paths NOT through dac = paths when dac removed from graph

    Then: paths_through_both = total - |¬fft| - |¬dac| + |¬fft ∩ ¬dac|

    But floating point fails at 10¹⁷ scale! Instead of matrix inversion,
    we use integer vector iteration: start with 1 "particle" at source,
    propagate via v = v @ A, count arrivals at destination. Exact integers.

References:
    - Neumann series: https://en.wikipedia.org/wiki/Neumann_series
    - Adjacency matrix powers: https://en.wikipedia.org/wiki/Adjacency_matrix#Matrix_powers
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()


# =============================================================================
# PARSING: Untangle the elf's cable spaghetti
# =============================================================================

def untangle_cable_spaghetti(file_name: str) -> pd.DataFrame:
    """Parse the device wiring list that some elf scribbled on a napkin.

    Input format: "aaa: bbb ccc ddd" means device aaa outputs to bbb, ccc, ddd.
    Returns DataFrame with columns [src, dest] - one row per edge.
    """
    napkin_path = Path(__file__).parent / file_name
    df = pd.read_csv(napkin_path, sep=":", names=["src", "dest"], header=None)
    df["src"] = df["src"].str.strip()
    df["dest"] = df["dest"].str.split()
    df = df.explode("dest").reset_index(drop=True)
    return df


# =============================================================================
# MATRIX BUILDING: Transform the wiring diagram into linear algebra
# =============================================================================

def map_devices_to_indices(wiring: pd.DataFrame) -> dict[str, int]:
    """Assign each device a matrix index. Santa's got 631 devices to track!"""
    all_devices = pd.concat([wiring["src"], wiring["dest"]]).unique()
    device_to_idx = {device: i for i, device in enumerate(all_devices)}
    return device_to_idx


def build_adjacency_matrix(
    wiring: pd.DataFrame,
    device_to_idx: dict[str, int],
    dtype=np.float64
) -> np.ndarray:
    """Build NxN adjacency matrix A where A[i,j] = 1 if device i → device j.

    The dtype parameter matters:
        - float64 for Part 1 (matrix inversion needs floats)
        - int64 for Part 2 (exact integer arithmetic)
    """
    src_indices = wiring["src"].map(device_to_idx).values
    dest_indices = wiring["dest"].map(device_to_idx).values
    n_devices = len(device_to_idx)

    A = np.zeros((n_devices, n_devices), dtype=dtype)
    A[src_indices, dest_indices] = 1  # Vectorized edge insertion
    return A


def verify_no_feedback_loops(A: np.ndarray) -> None:
    """The elves swear there are no feedback loops. Let's verify mathematically.

    DAG adjacency matrices are nilpotent: A^n = 0 for some n.
    Nilpotent matrices have ALL eigenvalues = 0.
    If any eigenvalue has magnitude > 0, someone wired a loop!
    """
    eigenvalues = np.linalg.eigvals(A.astype(np.float64))
    max_eigenvalue = np.max(np.abs(eigenvalues))
    if not np.isclose(max_eigenvalue, 0):
        raise ValueError(
            f"Uh oh! Found a feedback loop! (max eigenvalue: {max_eigenvalue:.6f}). "
            "The reactor might explode. Please notify the head elf immediately."
        )


# =============================================================================
# PART 1: Neumann Series Path Counting
# =============================================================================

def compute_all_path_counts(A: np.ndarray) -> np.ndarray:
    """Compute total paths between ALL device pairs using the Neumann series.

    The magic formula: (I - A)⁻¹ - I = A + A² + A³ + ...

    Result[i,j] = total number of paths from device i to device j.

    Why does this work?
        - A[i,j] counts 1-hop paths (direct edges)
        - A²[i,j] counts 2-hop paths (via any intermediate)
        - A³[i,j] counts 3-hop paths
        - Sum them all = total paths of any length

    The closed form (I-A)⁻¹ computes this infinite sum because for DAGs,
    the matrix A is nilpotent (eventually becomes zero when raised to
    powers exceeding the longest path length).
    """
    I = np.eye(A.shape[0])
    path_matrix = np.linalg.inv(I - A) - I
    return path_matrix


def count_signal_paths(file_name: str) -> int:
    """Part 1: How many ways can a signal travel from 'you' to 'out'?

    Uses the Neumann series for elegant O(V³) computation of all paths.
    """
    wiring = untangle_cable_spaghetti(file_name)
    device_idx = map_devices_to_indices(wiring)
    A = build_adjacency_matrix(wiring, device_idx, dtype=np.float64)
    path_matrix = compute_all_path_counts(A)

    # Look up the specific answer we need
    you_idx = device_idx["you"]
    out_idx = device_idx["out"]
    return int(round(path_matrix[you_idx, out_idx]))


# =============================================================================
# PART 2: Integer Vector Iteration + Inclusion-Exclusion
# =============================================================================

def count_paths_exact(
    wiring: pd.DataFrame,
    start_device: str,
    end_device: str,
    verify_dag: bool = False
) -> int:
    """Count paths using integer vector iteration. No floating point errors!

    Instead of matrix inversion, we propagate "particles" through the graph:
        1. Place 1 particle at the start device
        2. Each iteration: v = v @ A (particles follow ALL outgoing edges)
        3. Count arrivals at destination each step
        4. Stop when all particles have exited (np.sum(v) == 0)

    Why integers matter:
        Part 2 has ~10¹⁷ total paths. Float64 has ~15 digits of precision.
        When we subtract similar large numbers, floating point loses accuracy.
        Integer arithmetic is exact regardless of magnitude.

    Args:
        wiring: Edge DataFrame with columns [src, dest]
        start_device: Where signals originate (e.g., "svr")
        end_device: Where signals terminate (e.g., "out")
        verify_dag: If True, check for feedback loops before computing
    """
    device_idx = map_devices_to_indices(wiring)

    # Handle edge case: device was removed from graph
    if start_device not in device_idx or end_device not in device_idx:
        return 0

    A = build_adjacency_matrix(wiring, device_idx, dtype=np.int64)

    if verify_dag:
        verify_no_feedback_loops(A)

    start_idx = device_idx[start_device]
    end_idx = device_idx[end_device]

    # Initialize: one particle at the start device
    particles = np.zeros(len(device_idx), dtype=np.int64)
    particles[start_idx] = 1

    # Propagate until all particles exit the graph
    total_arrivals = 0
    max_iterations = len(device_idx)  # Longest possible path in a DAG

    for _ in range(max_iterations):
        particles = particles @ A  # Particles follow all outgoing edges
        if np.sum(particles) == 0:
            break  # All particles have exited
        total_arrivals += particles[end_idx]

    return int(total_arrivals)


def remove_device(wiring: pd.DataFrame, device: str) -> pd.DataFrame:
    """Remove a device from the wiring diagram (for inclusion-exclusion)."""
    return wiring[(wiring["src"] != device) & (wiring["dest"] != device)]


def count_suspicious_paths(file_name: str) -> int:
    """Part 2: Count paths from 'svr' to 'out' through BOTH 'dac' AND 'fft'.

    The inclusion-exclusion principle from set theory:

        paths_through_both = total - |¬dac| - |¬fft| + |¬dac ∩ ¬fft|

    Where |¬X| means "paths NOT through X" = paths when X is removed.

    Why this works (tracking each path type through the formula):
        - Paths through both:  +1 -0 -0 +0 = +1  (counted!)
        - Paths through dac only: +1 -1 -0 +0 = 0   (cancelled)
        - Paths through fft only: +1 -0 -1 +0 = 0   (cancelled)
        - Paths through neither:  +1 -1 -1 +1 = 0   (cancelled)

    Only paths through BOTH checkpoints survive the arithmetic!
    """
    wiring = untangle_cable_spaghetti(file_name)

    # Build the four graphs needed for inclusion-exclusion
    wiring_no_fft = remove_device(wiring, "fft")
    wiring_no_dac = remove_device(wiring, "dac")
    wiring_no_both = remove_device(remove_device(wiring, "fft"), "dac")

    # Compute the four path counts (all exact integers)
    total = count_paths_exact(wiring, "svr", "out")
    without_fft = count_paths_exact(wiring_no_fft, "svr", "out")
    without_dac = count_paths_exact(wiring_no_dac, "svr", "out")
    without_both = count_paths_exact(wiring_no_both, "svr", "out")

    # Apply inclusion-exclusion
    return total - without_fft - without_dac + without_both


# =============================================================================
# MAIN: Run both parts with festive output
# =============================================================================

if __name__ == "__main__":
    # Verify against samples
    sample1 = count_signal_paths("sample.dat")
    print(f"Part 1 Sample: {sample1} (expected 5)")

    sample2 = count_suspicious_paths("sample2.dat")
    print(f"Part 2 Sample: {sample2} (expected 2)")

    # Solve the real puzzle
    answer1 = count_signal_paths("data.dat")
    console.print(Panel(
        f"[bold green]{answer1:,}[/]",
        title="[red]Day 11 Part 1 (Matrix)[/red]",
        border_style="green"
    ))

    answer2 = count_suspicious_paths("data.dat")
    console.print(Panel(
        f"[bold green]{answer2:,}[/]",
        title="[red]Day 11 Part 2 (Matrix)[/red]",
        border_style="green"
    ))
