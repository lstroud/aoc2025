

from pathlib import Path
import numpy as np


def load_tachyon_matrix(file_path: str) -> np.ndarray:
    """Load warehouse grid from file as character array."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    with open(full_path) as f:
        lines = [list(line.strip()) for line in f]
    grid = np.array(lines, dtype='U1')  # 'U1' = single Unicode char
    return np.array(grid)

def build_transition_matrix(row_mask: np.ndarray):
    """
    row_mask: 1D boolean array - True where NOT splitter 
    returns: W×W transition matrix
    """
    W = len(row_mask)
    # build the pass through matrix
    pass_through = np.diag(row_mask)
    
    splitter_mask = 1 - row_mask
    split_diag = np.diag(splitter_mask)
    split_left = np.roll(split_diag, -1, axis=1)
    split_left[:, -1] = 0 
    split_right = np.roll(split_diag, 1, axis=1)
    split_left[:, 0] = 0 

    return pass_through + split_left + split_right
    

grid = load_tachyon_matrix('data.dat')

splitter_mask = (grid == '^')
state = np.where(grid[0, :] == 'S', 1, 0)

split_count = 0
for idx, row in enumerate(splitter_mask):
    # print("Idx:", idx)
    splitter_row = row.astype(int)
    # invert to make splitters 0
    row_mask = np.logical_not(splitter_row)
    # multiply the state by the splitter to get split counts, then sum
    splits_this_row = np.sum(state * splitter_row)
    split_count += splits_this_row
    # build the tranisition matrix to convert the state to the next state, based on splits
    T = build_transition_matrix(row_mask)
    # compute the next state clipped to 1 since beams don't accumulate
    state = np.clip(state @ T, 0, 1)
    # print(f"Idx: {idx}, State: {state}")

print("Total Splits: ", split_count)
    



# print(grid)
# print(splitter_mask)
# print(pass_through)


