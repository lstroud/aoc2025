"""Largest two-digit selection using cummax from the right.

Select two digits (in order) to form the largest possible number.
Example: "9512" → can form 95, 91, 92, 51, 52, 12 → max is 95

The cummax trick: reverse the series, cummax gives running max,
reverse back → now max_after[i] = max of all digits after position i.
This turns O(n²) brute force into O(n). Pandas makes it almost trivial.
"""
from pathlib import Path
import pandas as pd
from rich.console import Console


def parse_file(file_path: str) -> pd.DataFrame:
    """Load numbers from CSV file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)


def get_max_two_digit(s: str) -> int:
    """
    Find largest two-digit number from ordered digit pairs.

    Uses cummax trick: for each digit, find the max digit after it,
    then compute digit*10 + max_after for all positions.
    """
    d = pd.Series(list(s)).astype(int)
    # max_after[i] = max of all digits after position i
    max_after = d[::-1].cummax()[::-1].shift(-1)
    return int((d * 10 + max_after).max())


def display_results(console: Console, total: int):
    """Display puzzle results."""
    console.print("\n[bold]Puzzle Summary[/bold]")
    console.print(f"  [bold green]Total[/bold green] : {int(total):,}\n")


if __name__ == "__main__":
    df = parse_file('data.dat')
    df = pd.DataFrame(df.values.flatten(), columns=['value']).dropna().reset_index(drop=True)

    df['largest_two'] = df['value'].apply(lambda x: get_max_two_digit(str(x)))
    total = df['largest_two'].sum()

    console = Console()
    display_results(console, total)
