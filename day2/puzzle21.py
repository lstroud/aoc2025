"""
Advent of Code 2025 - Day 2, Part 1: Repeating Half Detection

Problem: Given ranges of IDs, find IDs where the first half of digits
equals the second half (only for even-length IDs).

Examples of invalid IDs: 11 (1==1), 1212 (12==12), 123123 (123==123)
"""
from pathlib import Path
import pandas as pd
from rich.console import Console


def parse_file(file_path: str) -> pd.DataFrame:
    """Load ID ranges from CSV file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)


def expand_ranges(ranges_df: pd.DataFrame) -> pd.DataFrame:
    """Expand range strings like '10-15' into individual IDs."""
    flat_df = pd.DataFrame(
        ranges_df.values.flatten(),
        columns=['range_str']
    ).dropna().reset_index(drop=True)

    # Parse ranges and expand to individual IDs
    flat_df['id_list'] = flat_df['range_str'].apply(
        lambda x: list(range(int(x.split('-')[0]), int(x.split('-')[1]) + 1))
    )

    # Explode into separate rows
    expanded_df = (flat_df
        .explode('id_list')
        .rename(columns={'id_list': 'id_number'})
        .astype({'id_number': 'int'})
        .reset_index(drop=True)
    )
    return expanded_df


def find_repeating_ids(expanded_df: pd.DataFrame) -> pd.DataFrame:
    """Find IDs where first half equals second half (even-length only)."""
    df = expanded_df.copy()
    df['id_str'] = df['id_number'].astype(str)
    df['length'] = df['id_str'].str.len()

    # Filter to even-length IDs only
    even_df = df[df['length'] % 2 == 0].copy()
    even_df['mid'] = even_df['length'] // 2

    # Split into halves and compare
    even_df['first_half'] = even_df.apply(lambda row: row['id_str'][:row['mid']], axis=1)
    even_df['second_half'] = even_df.apply(lambda row: row['id_str'][row['mid']:], axis=1)

    # Return rows where halves match
    return even_df.loc[
        even_df['first_half'] == even_df['second_half'],
        ['range_str', 'id_number', 'id_str']
    ].copy()


def display_results(console: Console, total_ids: int, invalid_df: pd.DataFrame):
    """Display puzzle results."""
    invalid_count = len(invalid_df)
    invalid_sum = invalid_df['id_number'].sum()

    console.print("\n[bold]Puzzle Summary[/bold]")
    console.print(f"  [cyan]Total IDs[/cyan]          : {total_ids:,}")
    console.print(f"  [yellow]Repeating IDs[/yellow]      : {invalid_count:,}")
    console.print(f"  [bold green]Sum of Repeating[/bold green]   : {int(invalid_sum):,}\n")


if __name__ == "__main__":
    ranges_df = parse_file('data.dat')
    expanded_df = expand_ranges(ranges_df)
    invalid_df = find_repeating_ids(expanded_df)

    console = Console()
    display_results(console, len(expanded_df), invalid_df)
