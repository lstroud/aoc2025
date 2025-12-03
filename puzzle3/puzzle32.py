from pathlib import Path
import time
import numpy as np
import pandas as pd
from rich.console import Console

def parse_file(file_path: str) -> pd.DataFrame:
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)

df = parse_file('data.dat')
df = pd.DataFrame(df.values.flatten(), columns=['value']).dropna().reset_index(drop=True)

def get_max_n_digit(s: str, length: int) -> int:
    arr = np.array(list(s), dtype=np.int8)
    result = np.empty(length, dtype=np.int8)
    start = 0

    for i in range(length):
        remaining = length - i
        end = len(arr) - remaining + 1
        best_idx = start + np.argmax(arr[start:end])
        result[i] = arr[best_idx]
        start = best_idx + 1
    return int(''.join(result.astype(str)))

def get_max_n_digit_stack(s: str, length: int) -> int:
    drop = len(s) - length
    stack = []
    for d in s:
        while drop and stack and d > stack[-1]:
            stack.pop()
            drop -= 1
        stack.append(d)
    return int(''.join(stack[:length]))

start = time.perf_counter()
df['largest_two'] = df['value'].apply(lambda x: get_max_n_digit(str(x), 12))
end = time.perf_counter()
print(f"Execution time argmax: {end - start:.4f} seconds")

start = time.perf_counter()
df['largest_two_stack'] = df['value'].apply(lambda x: get_max_n_digit_stack(str(x), 12))
end = time.perf_counter()
print(f"Execution time stack: {end - start:.4f} seconds")

total_lt = df['largest_two'].sum()

# Display Results
console = Console()
console.print("\n[bold]Puzzle Summary[/bold]")
console.print(f"  [red]Max Joltage [/red] : {int(total_lt)}")
