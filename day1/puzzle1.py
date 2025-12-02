import numpy as np
from pathlib import Path
from rich.console import Console
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout

class RotationList:
    def __init__(self, rotations:list[str], start_position: int, circular_range: int):
        self.start_position = start_position
        self.circular_range = circular_range
        self.signed_rotations = self._convert_to_signed_array(rotations)


    @staticmethod
    def parse_file(file_path: str) -> list[str]:
        current_dir = Path(__file__).parent
        full_path = current_dir / file_path
        with open(full_path, 'r') as file:
            return [line.strip() for line in file.readlines()]
        
    def get_positions(self) -> np.ndarray:
        """Returns all positions as integers"""
        positions = np.mod(self.start_position + np.cumsum(self.signed_rotations), self.circular_range)
        return positions.astype(int)
    
    def count_zero_crossings(self) -> int:
        """Count how many times the dial points at 0 (landing or passing through)"""
        cumsum = self.start_position + np.cumsum(self.signed_rotations)
        cumsum_with_start = np.concatenate([[self.start_position], cumsum])

        A = cumsum_with_start[:-1]
        B = cumsum_with_start[1:]
        r = self.circular_range

        # // in python floors towards negative infinity
        # Rightward: floor(B/r) - floor(A/r)
        # Leftward: ceil(A/r) - ceil(B/r) = (-B)//r - (-A)//r
        crossings = np.where(
            self.signed_rotations >= 0,
            B // r - A // r,
            (-B) // r - (-A) // r
        )

        return int(np.sum(crossings))
   

    def display_results(self, positions: np.ndarray, console: Console):
        left_rotations = np.sum(self.signed_rotations < 0)
        right_rotations = np.sum(self.signed_rotations > 0)
        total_distance = np.sum(np.abs(self.signed_rotations))
        final_pos = int(positions[-1])
        zero_count = self.count_zero_crossings()
        
        console.print("\n[bold]Puzzle Summary[/bold]")
        console.print(f"  [cyan]Moves[/cyan]        : {len(self.signed_rotations)}")
        console.print(f"  [green]Right[/green]        : {int(right_rotations)}")
        console.print(f"  [red]Left[/red]         : {int(left_rotations)}")
        console.print(f"  [yellow]Distance[/yellow]     : {int(total_distance)}")
        console.print(f"  [magenta]Zero Crossings[/magenta] : {zero_count}")
        console.print(f"\n  [bold green]Answer: {final_pos}[/bold green]\n")

    def display_journey(self, positions: np.ndarray, console: Console):
         # Create table
        table = Table(title="Journey")
        table.add_column("Step", style="cyan")
        table.add_column("Position", style="green")
        
        for i in range(0, len(positions)):
            table.add_row(str(i), str(int(positions[i])))
        
        console.print(table)

    def _convert_to_signed_array(self, rotations:list[str]) -> np.ndarray:
        instructions = np.array(rotations)
        directions = np.array([x[0] for x in instructions])
        values = np.array([int(x[1:]) for x in instructions])
        
        # Vectorized sign conversion
        signs = np.where(directions == 'L', -1, 1)
        result = signs * values
        return result
    
console = Console()
instructions = RotationList.parse_file('./data.dat')
if len(instructions) > 0:
    rotationList = RotationList(instructions, start_position=50, circular_range=100)
    positions = rotationList.get_positions()
    # print(positions)
    rotationList.display_results(positions, console=console)
    # rotationList.display_journey(positions, console)
