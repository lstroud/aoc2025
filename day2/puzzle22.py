
from pathlib import Path
import pandas as pd
from rich.console import Console

def parse_file(file_path: str) -> pd.DataFrame:
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)

df = parse_file('data.dat')
df = pd.DataFrame(df.values.flatten(), columns=['value']).dropna().reset_index(drop=True)

# Expand ranges into individual product IDs
df['product-id-range'] = df['value'].apply(lambda x: list(range(int(x.split('-')[0]), int(x.split('-')[1])+1)))

# Explode the ranges into separate rows
dfe = (df
    .explode('product-id-range')
    .rename(columns={'product-id-range': 'product-id-number'})
    .astype({'product-id-number': 'int'})
    .reset_index(drop=True)
)
# Create string representation of product IDs
dfe['product-id'] = dfe['product-id-number'].astype(str)
dfe['length'] = dfe['product-id'].str.len()

# Using regex to find invalid product ids
# ^(.+) - captures any sequence of 1+ characters at the start
# \1+ - backreference requiring that pattern to repeat 1+ more times
regex_pattern = r'^(.+)\1+$'
dfr = dfe.loc[dfe['product-id'].str.match(r'^(.+)\1+$'), ['value', 'product-id-number', 'product-id']].copy()

# Calculate Outputs
product_id_count = len(dfe)
invalid_id_count = len(dfr)
total_invalid_product_ids = dfr['product-id-number'].sum()

# Display Results
console = Console()
console.print("\n[bold]Puzzle Summary[/bold]")
console.print(f"  [cyan]Id Count [/cyan]            : {int(product_id_count)}")
console.print(f"  [yellow]Invalid Ids [/yellow]         : {int(invalid_id_count)}")
console.print(f"  [bold green]Total of Invalid Ids[/bold green] : {int(total_invalid_product_ids)}\n")
