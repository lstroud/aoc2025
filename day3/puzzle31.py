from itertools import combinations
from pathlib import Path
import re
import pandas as pd
from rich.console import Console

def parse_file(file_path: str) -> pd.DataFrame:
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)

df = parse_file('data.dat')
df = pd.DataFrame(df.values.flatten(), columns=['value']).dropna().reset_index(drop=True)

def get_max_two_digit(s:str):
    return max(int(s[i] + s[j]) for i, j in combinations(range(len(s)), 2))

def get_max_two_digit_p(s: str) -> int:
    d = pd.Series(list(s)).astype(int)
    max_after = d[::-1].cummax()[::-1].shift(-1)
    return int((d * 10 + max_after).max())


df['largest_two'] = df['value'].apply(lambda x: get_max_two_digit_p(str(x)))
total_lt = df['largest_two'].sum()

# Display Results
console = Console()
console.print("\n[bold]Puzzle Summary[/bold]")
console.print(f"  [red]Max Joltage [/red] : {int(total_lt)}")
