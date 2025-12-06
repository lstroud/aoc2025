from dataclasses import dataclass
from pathlib import Path
import time
import tracemalloc
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.padding import Padding


@dataclass
class Boundary:
    start: int
    end: int
    op: str

def parse_file(file_path: str) -> pd.DataFrame:
    """Load numbers from CSV file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    with open(full_path) as f:
        df =  pd.DataFrame([line.rstrip('\n') for line in f])
    return df

def pivot_column(col_chargrid: np.ndarray) -> list[int]:
    _, cols = col_chargrid.shape
    numbers = []

    # right to left
    for col_idx in range(cols - 1, -1, -1):
        col_chars = col_chargrid[:, col_idx]
        digits = [c for c in col_chars if c != ' ']
        if digits:
            number = int(''.join(digits))
            numbers.append(number)
    return numbers


def find_column_boundaries(op_line: str) -> list[Boundary]:
    op_chars = np.array(list(op_line))
    # find op positions
    op_mask = np.isin(op_chars, ['*', '+'])
    op_positions = np.where(op_mask)[0]
    operators = op_chars[op_mask]

    # column boundaries
    starts = op_positions
    ends = np.append(op_positions[1:], len(op_line))
    boundaries = [Boundary(s, e, o) for s, e, o in zip(starts, ends, operators)]
    return boundaries

def convert_column(boundary: Boundary, char_grid: np.ndarray) -> list[int]:
    col_chars = char_grid[:, boundary.start:boundary.end]
    numbers = pivot_column(col_chars)
    return numbers

# get rows
df = parse_file('data.dat')

# get ops row
op_line = df.iloc[-1, 0]

# get char grid
data_rows = df.iloc[:-1, 0]
char_grid = np.array(data_rows.apply(list).to_list())

# find the columns
boundaries = find_column_boundaries(op_line)

# pivot and aggregate
total = 0
for boundary in boundaries:
    numbers = convert_column(boundary, char_grid)
    if boundary.op == '*':
        total += np.prod(numbers)
    elif boundary.op == '+':
        total += np.sum(numbers)
    else:
        raise ValueError("Unknown Operator: ", boundary.op)

print(total)

