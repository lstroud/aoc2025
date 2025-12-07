"""
Day 7: Tachyon Beam Splitting - Advent of Code 2025

Simulates tachyon beam propagation through a manifold.
Compares three approaches:
1. Matrix Propagation - Markov chain-like transition matrices (numpy)
2. Convolution - 1D convolution with splitter kernel [1, 0, 1] (scipy)
3. PyTorch Neural - Feedforward network with activation functions (torch)
"""

from pathlib import Path
import time
import tracemalloc
import numpy as np
from scipy.ndimage import convolve1d
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.padding import Padding


# =============================================================================
# Data Loading
# =============================================================================

def load_tachyon_manifold(file_path: str) -> np.ndarray:
    """Load tachyon manifold diagram from file as character array."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    with open(full_path) as f:
        lines = [list(line.strip()) for line in f]
    return np.array(lines, dtype='U1')


# =============================================================================
# Strategy 1: Matrix Propagation - Similar to a Neural Network with Numpy
# =============================================================================

def build_propagation_matrix(passthrough_mask: np.ndarray) -> np.ndarray:
    """
    Build a propagation matrix for tachyon beam movement through one row.

    Models beam behavior as a linear transformation:
    - Empty space (.): beam passes through unchanged (identity)
    - Splitter (^): beam stops, emits new beams left and right
    """
    # Beams in empty space continue straight down (identity diagonal)
    passthrough = np.diag(passthrough_mask)

    # Splitters stop the beam and emit left/right
    splitter_mask = 1 - passthrough_mask
    splitter_diag = np.diag(splitter_mask)

    # Shift splitter contributions left and right
    # Zero out wrapped edges (beams exit manifold, don't wrap)
    emit_left = np.roll(splitter_diag, -1, axis=1)
    emit_left[:, -1] = 0
    emit_right = np.roll(splitter_diag, 1, axis=1)
    emit_right[:, 0] = 0

    return passthrough + emit_left + emit_right


def matrix_propagation(manifold: np.ndarray) -> int:
    """
    Solve using matrix multiplication for beam propagation.

    Each row's splitter configuration defines a transition matrix.
    Beam state propagates via: beam_positions @ propagation_matrix
    """
    splitter_locations = (manifold == '^')
    beam_positions = np.where(manifold[0, :] == 'S', 1, 0)

    total_splits = 0
    for row_splitters in splitter_locations:
        row_splitters_int = row_splitters.astype(int)

        # Count splits: each beam at a splitter position causes one split
        total_splits += np.sum(beam_positions * row_splitters_int)

        # Build propagation matrix and advance beam state
        passthrough_mask = np.logical_not(row_splitters).astype(int)
        propagation = build_propagation_matrix(passthrough_mask)

        # Propagate beams, clip to binary (beams merge, don't accumulate)
        beam_positions = np.clip(beam_positions @ propagation, 0, 1)

    return total_splits


# =============================================================================
# Strategy 2: Convolution
# =============================================================================

def convolution(manifold: np.ndarray) -> int:
    """
    Solve using 1D convolution for beam propagation.

    Splitter acts as kernel [1, 0, 1] - spreading signal left and right.
    Pass-through is identity (no change).

    TODO: Implement this strategy
    - Use scipy.ndimage.convolve1d or manual convolution
    - Splitter kernel: [1, 0, 1]
    - Handle boundaries (beams exit, don't wrap)
    - Clip to binary after each row (beams merge)
    """
    splitter_locations = (manifold == '^')
    beam_positions = np.where(manifold[0, :] == 'S', 1, 0).astype(float)

    # Convolution kernels
    splitter_kernel = np.array([1, 0, 1])  # Emit left and right
    identity_kernel = np.array([0, 1, 0])  # Pass through

    total_splits = 0
    for row_splitters in splitter_locations:
        row_splitters_int = row_splitters.astype(int)

        # Count splits before propagation
        total_splits += int(np.sum(beam_positions * row_splitters_int))

        # Beams that hit the splitters
        beams_at_splitters = beam_positions * row_splitters_int

        # Beams that pass through
        passthrough_mask = (~row_splitters).astype(int)
        beams_passing = beam_positions * passthrough_mask
        
        # Apply convolution based on splitter positions
        beam_spread = convolve1d(beams_at_splitters, splitter_kernel, mode='constant', cval=0)
        # Hint: For each column, apply either splitter_kernel or identity_kernel
        new_positions = beam_spread + (beam_positions * passthrough_mask)
        beam_positions = np.clip(new_positions, 0, 1)

    return total_splits


# =============================================================================
# Strategy 3: PyTorch Neural Network Approximation
# =============================================================================
def build_propagation_matrix_torch(passthrough_mask: torch.Tensor) -> torch.Tensor:
    # passthrough_mask is a 1D tensor of 0s and 1s
    # Return a 2D tensor (W x W)

    passthrough = torch.diag(passthrough_mask)
    splitter_mask = 1 - passthrough_mask
    splitter_diag = torch.diag(splitter_mask)

    emit_left = torch.roll(splitter_diag, -1, dims=1)
    emit_left[:, -1] = 0
    emit_right = torch.roll(splitter_diag, 1, dims=1)
    emit_right[:, 0] = 0

    return passthrough + emit_left + emit_right

def pytorch_neural(manifold: np.ndarray, device: str = "cpu") -> int:
    """
    Solve using PyTorch tensors and activation functions.

    Frames the problem as a feedforward neural network:
    - Each row is a "layer" with fixed weights (propagation matrix)
    - Activation function: hard sigmoid (clamp to [0,1]) for beam merging
    - Forward pass = beam propagation through manifold

    Args:
        manifold: Character array of the tachyon manifold
        device: PyTorch device ("cpu", "cuda", or "mps")
    """
    torch_device = torch.device(device)
    splitter_locations = (manifold == '^')

    # Convert to torch tensors on specified device
    beam_positions = torch.tensor(
        np.where(manifold[0, :] == 'S', 1.0, 0.0),
        dtype=torch.float32,
        device=torch_device
    )

    total_splits = 0
    for row_splitters in splitter_locations:
        row_splitters_tensor = torch.tensor(
            row_splitters.astype(np.float32),
            device=torch_device
        )

        # Count splits before propagation
        total_splits += int(torch.sum(beam_positions * row_splitters_tensor).item())

        # Build propagation matrix as torch tensor
        passthrough_mask = torch.tensor(
            (~row_splitters).astype(np.float32),
            device=torch_device
        )
        propagation = build_propagation_matrix_torch(passthrough_mask)

        # Forward pass: beam_positions = beam_positions @ propagation
        beam_positions = beam_positions @ propagation
        # Conceptually an activation function
        beam_positions = torch.clamp(beam_positions, 0, 1)

    return total_splits


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark(name: str, func, manifold: np.ndarray) -> dict:
    """Run a strategy and measure time and memory (CPU and GPU)."""
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    # Reset GPU memory stats
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tracemalloc.start()
    start = time.perf_counter()

    result = func(manifold)

    # Ensure GPU operations complete before measuring
    if cuda_available:
        torch.cuda.synchronize()
    elif mps_available:
        torch.mps.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get GPU memory if available
    gpu_memory_kb = 0
    if cuda_available:
        gpu_memory_kb = torch.cuda.max_memory_allocated() / 1024
    elif mps_available:
        # driver_allocated_memory shows total GPU memory allocated by MPS
        gpu_memory_kb = torch.mps.driver_allocated_memory() / 1024

    return {
        'name': name,
        'result': result,
        'time_ms': elapsed_ms,
        'memory_kb': peak_memory / 1024,
        'gpu_memory_kb': gpu_memory_kb
    }


# =============================================================================
# Main
# =============================================================================

def get_gpu_info() -> tuple[str | None, str | None]:
    """Return (gpu_type, gpu_name) or (None, None) if no GPU."""
    if torch.cuda.is_available():
        return "CUDA", torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        return "MPS", "Apple Silicon"
    return None, None


def display_results(console: Console, results: list[dict], manifold_shape: tuple, gpu_type: str | None):
    """Display benchmark results in a table."""
    height, width = manifold_shape
    table = Table(title=f"Strategy Comparison ({height} x {width} manifold)")
    table.add_column("Strategy", style="cyan")
    table.add_column("Total Splits", justify="right", style="green")
    table.add_column("Time (ms)", justify="right", style="magenta")
    table.add_column("Memory (KB)", justify="right", style="blue")
    if gpu_type:
        table.add_column(f"{gpu_type} (KB)", justify="right", style="yellow")

    for r in results:
        row = [r['name'], f"{r['result']:,}", f"{r['time_ms']:.3f}", f"{r['memory_kb']:.1f}"]
        if gpu_type:
            row.append(f"{r['gpu_memory_kb']:.1f}")
        table.add_row(*row)

    console.print(Padding(table, (1, 0, 1, 4)))


if __name__ == "__main__":
    console = Console()
    console.print(Padding("\n[bold]Day 7: Tachyon Beam Splitting[/bold]", (0, 0, 1, 4)))

    manifold = load_tachyon_manifold('data.dat')
    console.print(Padding(f"[cyan]Manifold size:[/cyan] {manifold.shape[0]} rows x {manifold.shape[1]} columns", (0, 0, 1, 4)))

    gpu_type, gpu_name = get_gpu_info()
    if gpu_type:
        console.print(Padding(f"[green]GPU available:[/green] {gpu_name} ({gpu_type})", (0, 0, 1, 4)))
    else:
        console.print(Padding("[yellow]GPU:[/yellow] Not available (using CPU)", (0, 0, 1, 4)))

    # Capture GPU memory baseline before benchmarks
    gpu_baseline_kb = 0
    if gpu_type == "MPS":
        gpu_baseline_kb = torch.mps.driver_allocated_memory() / 1024
    elif gpu_type == "CUDA":
        gpu_baseline_kb = torch.cuda.memory_allocated() / 1024

    # Run benchmarks
    results = [
        benchmark("Matrix Propagation", matrix_propagation, manifold),
        benchmark("Convolution", convolution, manifold),
        benchmark("PyTorch (CPU)", lambda m: pytorch_neural(m, "cpu"), manifold),
    ]
    if gpu_type == "MPS":
        results.append(benchmark("PyTorch (MPS)", lambda m: pytorch_neural(m, "mps"), manifold))
    elif gpu_type == "CUDA":
        results.append(benchmark("PyTorch (CUDA)", lambda m: pytorch_neural(m, "cuda"), manifold))

    # Subtract baseline from GPU memory
    for r in results:
        r['gpu_memory_kb'] = max(0, r['gpu_memory_kb'] - gpu_baseline_kb)

    display_results(console, results, manifold.shape, gpu_type)
