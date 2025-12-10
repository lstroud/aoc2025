from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, OptimizeResult
import pandas as pd
from rich.console import Console
from rich.panel import Panel

class MachineSpec:
    MACHINE_SPEC_REGEX = re.compile(r'^\[([^\]]*)\]\s*([^{]+)(?:\{([^}]*)\})?$')
    @classmethod
    def from_line(cls, line) -> MachineSpec:
        m = cls.MACHINE_SPEC_REGEX.match(line)
        if not m:
            raise ValueError(f"Invalid machine spec line: {line!r}")
        lp = m.group(1)
        bs = m.group(2)
        jr = m.group(3)
        return cls(lp, bs, jr)

    def __init__(self, target_str: str, button_spec_str: str, joltage_spec_str:str):
        self.target_light_pattern = self._parse_target_pattern(target_str)
        self.button_specs = self._parse_button_specs(button_spec_str, len(self.target_light_pattern))
        self.joltage_requirements = self._parse_joltage_requirements(joltage_spec_str)

    def _parse_target_pattern(self, target_str: str) -> np.ndarray:
        # .##.
        char_arr = np.array(list(target_str))
        mask = (char_arr == '#')
        return np.where(mask, 1, 0)
    
    def _parse_button_specs(self, spec_str: str, number_of_lights: int) -> np.ndarray:
        return np.array([self._button_spec_to_mask(spec, number_of_lights) for spec in spec_str.split()])
        
    
    def _button_spec_to_mask(self, spec_str: str, number_of_lights: int) -> np.ndarray:
        # (0,2,3,4)
        button_mask = np.zeros(number_of_lights, dtype=int)
        indices = np.fromstring(spec_str.strip("()"), sep=",", dtype=int)
        button_mask[indices] = True
        return button_mask.astype(int)
    
    def _parse_joltage_requirements(self, joltage_spec: str) -> np.ndarray:
        return np.fromstring(joltage_spec.strip("()"), sep=",", dtype=int)
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "\n".join([
            f"\tLight Pattern: {self.target_light_pattern}\n",
            f"\tButton Specs: \n{"".join(f"\t{i}: {np.array2string(row)}\n" for i, row in enumerate(self.button_specs))}\n",
            f"\tJoltage Reqs: {self.joltage_requirements}\n"
        ])
    
def load_panel_specification(file_name: str) -> list[MachineSpec]:
    current_dir = Path(__file__).parent
    full_path = current_dir / file_name
    specs = []
    with open(full_path) as f:
        specs = [MachineSpec.from_line(line.strip()) for line in f]
    return specs

def solve(file_name: str) -> int:
    """Solve the puzzle for the given input file."""
    machine_specs = load_panel_specification(file_name)
    total_presses = 0

    for machine_spec in machine_specs:
        n_buttons = machine_spec.button_specs.shape[0]
        result: OptimizeResult = milp(np.ones(n_buttons),
                      constraints=LinearConstraint(A=machine_spec.button_specs.T, 
                                                   lb=machine_spec.joltage_requirements, 
                                                   ub=machine_spec.joltage_requirements),
                        integrality = np.ones(n_buttons),
                        bounds = Bounds(lb=0, ub=np.inf) 
        )
        print(result)
        if result.success:
            presses = result.fun
        else:
            print(f"No solution found: {result.message}")
        total_presses += presses

    return total_presses


if __name__ == "__main__":
    console = Console()

    # Test on sample
    sample_result = solve('sample.dat')
    console.print(f"Sample: {sample_result} (expected 33)")

    # Run on real data
    result = solve('data.dat')
    console.print(Panel(
        f"[bold green]{result}[/bold green]",
        title="[red]Day 10 Part 2[/red]",
        border_style="red"
    ))