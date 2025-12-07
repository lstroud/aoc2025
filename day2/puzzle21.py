"""Repeating half detection using string slicing.

Find IDs where first half equals second half (even-length only).
The elves apparently have a pattern problem: 1212, 123123, etc.
"""
from pathlib import Path
import pandas as pd
from rich.console import Console


def parse_file(file_path: str) -> pd.DataFrame:
    """
    Load ID ranges from CSV file.

    Args:
        file_path: Path to CSV (relative to this module)

    Returns:
        DataFrame with range strings like "10-15"
    """
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)


def expand_ranges(ranges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand range strings into individual IDs using pandas explode.

    Args:
        ranges_df: DataFrame with range strings

    Returns:
        DataFrame with one row per ID
    """
    flat_df = pd.DataFrame(
        ranges_df.values.flatten(),
        columns=['range_str']
    ).dropna().reset_index(drop=True)

    # Parse "10-15" into [10, 11, 12, 13, 14, 15]
    flat_df['id_list'] = flat_df['range_str'].apply(
        lambda x: list(range(int(x.split('-')[0]), int(x.split('-')[1]) + 1))
    )

    # Explode turns one row with a list into many rows
    return (flat_df
        .explode('id_list')
        .rename(columns={'id_list': 'id_number'})
        .astype({'id_number': 'int'})
        .reset_index(drop=True)
    )


def find_repeating_ids(expanded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find IDs where first half equals second half using string slicing.

    Only checks even-length IDs since odd ones can't split evenly.

    Args:
        expanded_df: DataFrame with id_number column

    Returns:
        DataFrame of matching IDs
    """
    df = expanded_df.copy()
    df['id_str'] = df['id_number'].astype(str)
    df['length'] = df['id_str'].str.len()

    even_df = df[df['length'] % 2 == 0].copy()
    even_df['mid'] = even_df['length'] // 2

    # Split at midpoint and compare - O(n) string slicing
    even_df['first_half'] = even_df.apply(lambda row: row['id_str'][:row['mid']], axis=1)
    even_df['second_half'] = even_df.apply(lambda row: row['id_str'][row['mid']:], axis=1)

    return even_df.loc[
        even_df['first_half'] == even_df['second_half'],
        ['range_str', 'id_number', 'id_str']
    ].copy()


def display_results(console: Console, total_ids: int, invalid_df: pd.DataFrame):
    """
    Print the final results.

    Args:
        console: Rich console for output
        total_ids: How many IDs we checked
        invalid_df: DataFrame of repeating pattern matches
    """
    console.print("\n[bold]Puzzle Summary[/bold]")
    console.print(f"  [cyan]Total IDs[/cyan]          : {total_ids:,}")
    console.print(f"  [yellow]Repeating IDs[/yellow]      : {len(invalid_df):,}")
    console.print(f"  [bold green]Sum of Repeating[/bold green]   : {int(invalid_df['id_number'].sum()):,}\n")


if __name__ == "__main__":
    ranges_df = parse_file('data.dat')
    expanded_df = expand_ranges(ranges_df)
    invalid_df = find_repeating_ids(expanded_df)

    console = Console()
    display_results(console, len(expanded_df), invalid_df)
