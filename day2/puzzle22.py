"""Repeating pattern detection using regex backreferences.

Find IDs made entirely of a repeating pattern (any length).
Examples: 1111 ("1"×4), 123123 ("123"×2), 121212 ("12"×3)

The regex ^(.+)\\1+$ is elegant: capture any prefix, require it to
repeat at least once to fill the string. Backreferences do the work.
"""
from pathlib import Path
import pandas as pd
from rich.console import Console

REPEATING_PATTERN = r'^(.+)\1+$'


def parse_file(file_path: str) -> pd.DataFrame:
    """
    Load ID ranges from CSV file.

    Args:
        file_path: Path to CSV (relative to this module)

    Returns:
        DataFrame with range strings
    """
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)


def expand_ranges(ranges_df: pd.DataFrame) -> pd.DataFrame:
    """Expand range strings like '10-15' into individual IDs."""
    flat_df = pd.DataFrame(
        ranges_df.values.flatten(),
        columns=['range_str']
    ).dropna().reset_index(drop=True)

    flat_df['id_list'] = flat_df['range_str'].apply(
        lambda x: list(range(int(x.split('-')[0]), int(x.split('-')[1]) + 1))
    )

    expanded_df = (flat_df
        .explode('id_list')
        .rename(columns={'id_list': 'id_number'})
        .astype({'id_number': 'int'})
        .reset_index(drop=True)
    )
    expanded_df['id_str'] = expanded_df['id_number'].astype(str)
    return expanded_df


def find_repeating_pattern_ids(expanded_df: pd.DataFrame) -> pd.DataFrame:
    """Find IDs that consist of a repeating pattern using regex."""
    return expanded_df.loc[
        expanded_df['id_str'].str.match(REPEATING_PATTERN),
        ['range_str', 'id_number', 'id_str']
    ].copy()


def display_results(console: Console, total_ids: int, invalid_df: pd.DataFrame):
    """Display puzzle results."""
    invalid_count = len(invalid_df)
    invalid_sum = invalid_df['id_number'].sum()

    console.print("\n[bold]Puzzle Summary[/bold]")
    console.print(f"  [cyan]Total IDs[/cyan]          : {total_ids:,}")
    console.print(f"  [yellow]Repeating Pattern[/yellow]  : {invalid_count:,}")
    console.print(f"  [bold green]Sum of Repeating[/bold green]   : {int(invalid_sum):,}\n")


if __name__ == "__main__":
    ranges_df = parse_file('data.dat')
    expanded_df = expand_ranges(ranges_df)
    invalid_df = find_repeating_pattern_ids(expanded_df)

    console = Console()
    display_results(console, len(expanded_df), invalid_df)
