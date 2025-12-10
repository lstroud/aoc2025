from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re
import numpy as np
from rich.console import Console
from rich.panel import Panel

class MachineSpec:
    MACHINE_SPEC_REGEX = re.compile(r'^\[([^\]]*)\]\s*([^{]+)(?:\{([^}]*)\})?$')
    @classmethod
    def from_line(cls, line) -> MachineSpec:
        m = cls.MACHINE_SPEC_REGEX.match(line)
        if not m:
            raise ValueError(f"Invalid machine spec line: {line!r}")
        lp = m.group(1)
        bs = m.group(2)
        jr = m.group(3)
        return cls(lp, bs, jr)

    def __init__(self, target_str: str, button_spec_str: str, joltage_spec_str:str):
        self.target_light_pattern = self._parse_target_pattern(target_str)
        self.button_specs = self._parse_button_specs(button_spec_str, len(self.target_light_pattern))
        self.joltage_requirements = self._parse_joltage_requirements(joltage_spec_str)

    def _parse_target_pattern(self, target_str: str) -> np.ndarray:
        # .##.
        char_arr = np.array(list(target_str))
        mask = (char_arr == '#')
        return np.where(mask, 1, 0)
    
    def _parse_button_specs(self, spec_str: str, number_of_lights: int) -> np.ndarray:
        return np.array([self._button_spec_to_mask(spec, number_of_lights) for spec in spec_str.split()])
        
    
    def _button_spec_to_mask(self, spec_str: str, number_of_lights: int) -> np.ndarray:
        # (0,2,3,4)
        button_mask = np.zeros(number_of_lights, dtype=int)
        indices = np.fromstring(spec_str.strip("()"), sep=",", dtype=int)
        button_mask[indices] = True
        return button_mask.astype(int)
    
    def _parse_joltage_requirements(self, joltage_spec: str) -> np.ndarray:
        return np.fromstring(joltage_spec.strip("()"), sep=",", dtype=int)
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "\n".join([
            f"\tLight Pattern: {self.target_light_pattern}\n",
            f"\tButton Specs: \n{"".join(f"\t{i}: {np.array2string(row)}\n" for i, row in enumerate(self.button_specs))}\n",
            f"\tJoltage Reqs: {self.joltage_requirements}\n"
        ])
    
def load_panel_specification(file_name: str) -> list[MachineSpec]:
    current_dir = Path(__file__).parent
    full_path = current_dir / file_name
    specs = []
    with open(full_path) as f:
        specs = [MachineSpec.from_line(line.strip()) for line in f]
    return specs


def swap_rows(M, row_index_1, row_index_2):
    """
    Swap rows in the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to perform row swaps on.
    - row_index_1 (int): Index of the first row to be swapped.
    - row_index_2 (int): Index of the second row to be swapped.
    """

    # Copy matrix M so the changes do not affect the original matrix. 
    M = M.copy()
    # Swap indexes
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M

def get_index_first_non_zero_value_from_column(M, column, starting_row):
    """
    Retrieve the index of the first non-zero value in a specified column of the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to search for non-zero values.
    - column (int): The index of the column to search.
    - starting_row (int): The starting row index for the search.

    Returns:
    int: The index of the first non-zero value in the specified column, starting from the given row.
                Returns -1 if no non-zero value is found.
    """
    # Get the column array starting from the specified row
    column_array = M[starting_row:,column]
    for i, val in enumerate(column_array):
        # Iterate over every value in the column array. 
        # To check for non-zero values, you must always use np.isclose instead of doing "val == 0".
        if not np.isclose(val, 0, atol = 1e-5):
            # If one non zero value is found, then adjust the index to match the correct index in the matrix and return it.
            index = i + starting_row
            return index
    # If no non-zero value is found below it, return -1.
    return -1

def get_index_first_non_zero_value_from_row(M, row, augmented = False):
    """
    Find the index of the first non-zero value in the specified row of the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to search for non-zero values.
    - row (int): The index of the row to search.
    - augmented (bool): Pass this True if you are dealing with an augmented matrix, 
                        so it will ignore the constant values (the last column in the augmented matrix).

    Returns:
    int: The index of the first non-zero value in the specified row.
                Returns -1 if no non-zero value is found.
    """

    # Create a copy to avoid modifying the original matrix
    M = M.copy()


    # If it is an augmented matrix, then ignore the constant values
    if augmented == True:
        # Isolating the coefficient matrix (removing the constant terms)
        M = M[:,:-1]
        
    # Get the desired row
    row_array = M[row]
    for i, val in enumerate(row_array):
        # If finds a non zero value, returns the index. Otherwise returns -1.
        if not np.isclose(val, 0, atol = 1e-5):
            return i
    return -1

def augmented_matrix(A, B):
    """
    Create an augmented matrix by horizontally stacking two matrices A and B.

    Parameters:
    - A (numpy.array): First matrix.
    - B (numpy.array): Second matrix.

    Returns:
    - numpy.array: Augmented matrix obtained by horizontally stacking A and B.
    """
    if B.ndim == 1:
      B = B.reshape(-1, 1)
    augmented_M = np.hstack((A,B))
    return augmented_M

def row_echelon_form(A, B, mod=None):
    """
    Utilizes elementary row operations to transform a given set of matrices, 
    which represent the coefficients and constant terms of a linear system, into row echelon form.

    Parameters:
    - A (numpy.array): The input square matrix of coefficients.
    - B (numpy.array): The input column matrix of constant terms

    Returns:
    numpy.array: A new augmented matrix in row echelon form with pivots as 1.
    """
    
    # Make copies of the input matrices to avoid modifying the originals
    A = A.copy()
    B = B.copy()


    # Convert matrices to float to prevent integer division
    A = A.astype('float64')
    B = B.astype('float64')

    # Transform matrices A and B into the augmented matrix M
    M = augmented_matrix(A, B)
    
    # Number of rows, cols in the coefficient matrix
    num_rows = len(A)
    num_cols = A.shape[1]

    # Iterate over the rows
    row = 0
    col = 0
    while row < num_rows and col < num_cols:
        pivot_candidate = M[row, col]

        # If pivot_candidate is zero, look for a non-zero element below
        if np.isclose(pivot_candidate, 0):
            pivot_idx = get_index_first_non_zero_value_from_column(M, col, row)
            if pivot_idx == -1:
                col += 1
                continue
            M = swap_rows(M, row, pivot_idx)
            pivot = M[row, col]
        else:
            pivot = pivot_candidate

        # Scale row to make pivot = 1
        M[row] = M[row] / pivot
        if mod:
            M[row] = M[row] % mod

        # Eliminate below
        for j in range(row + 1, num_rows):
            factor = M[j, col]
            M[j] = M[j] - factor * M[row]
            if mod:
                M[j] = M[j] % mod

        row += 1
        col += 1

    return M

def back_substitution(M, mod=None):
    """
    Perform back substitution on an augmented matrix (with unique solution) in reduced row echelon form to find the solution to the linear system.

    Parameters:
    - M (numpy.array): The augmented matrix in row echelon form with unitary pivots (n x n+1).

    Returns:
    numpy.array: The solution vector of the linear system.
    """
    
    # Make a copy of the input matrix to avoid modifying the original
    M = M.copy()

    # Get the number of rows (and columns) in the matrix of coefficients
    num_rows = M.shape[0]
    num_cols = M.shape[1] - 1

    solution = np.zeros(num_cols)
    pivot_cols = []
    for r in range(num_rows):
        pc = get_index_first_non_zero_value_from_row(M, r, augmented=True)
        if pc != -1:
            pivot_cols.append((r, pc))
    
    # Find free columns (columns without pivots)
    pivot_col_set = {col for (row, col) in pivot_cols}
    free_cols = [c for c in range(num_cols) if c not in pivot_col_set]

    # Try all combinations of free variable assignments
    best_solution = None
    best_count = float('inf')

    for free_vals in product([0, 1], repeat=len(free_cols)):
        # Start with this assignment of free variables
        solution = np.zeros(num_cols)
        for i, col in enumerate(free_cols):
            solution[col] = free_vals[i]

        # Back-substitute to get pivot variable values
        for row, col in reversed(pivot_cols):
            val = M[row, -1]
            for other_col in range(col + 1, num_cols):
                val = val - M[row, other_col] * solution[other_col]
            if mod:
                val = val % mod
            solution[col] = val

        # Track minimum
        count = int(solution.sum())
        if count < best_count:
            best_count = count
            best_solution = solution.copy()

    solution = best_solution
    return solution

def gaussian_elimination(A, B, mod=None):
    """
    Solve a linear system represented by an augmented matrix using the Gaussian elimination method.

    Parameters:
    - A (numpy.array): Square matrix of size n x n representing the coefficients of the linear system
    - B (numpy.array): Column matrix of size 1 x n representing the constant terms.

    Returns:
    numpy.array: The solution vector.
    """

    row_echelon_M = row_echelon_form(A, B, mod=mod)
    solution = back_substitution(row_echelon_M, mod=mod)
    return solution


def solve(file_name: str) -> int:
    """Solve the puzzle for the given input file."""
    machine_specs = load_panel_specification(file_name)
    total_presses = 0

    for machine_spec in machine_specs:
        result = gaussian_elimination(machine_spec.button_specs.T, machine_spec.target_light_pattern, mod=2)
        presses = int(result.sum())
        total_presses += presses

    return total_presses


if __name__ == "__main__":
    console = Console()

    # Test on sample
    sample_result = solve('sample.dat')
    console.print(f"Sample: {sample_result} (expected 7)")

    # Run on real data
    result = solve('data.dat')
    console.print(Panel(
        f"[bold green]{result}[/bold green]",
        title="[red]Day 10 Part 1[/red]",
        border_style="red"
    ))