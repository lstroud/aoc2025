"""Day 10 Part 2: Factory Joltage Configuration

The elves need to configure factory machines by pressing buttons to hit
exact joltage targets. Each button increments specific counters.
Minimize total button presses across all machines.

This is Integer Linear Programming (ILP):
- Minimize: sum of button presses
- Subject to: button_effects @ presses = joltage_targets
- Constraints: presses >= 0, integers only

scipy.optimize.milp handles the heavy lifting. The elves' fingers thank us.
"""
from pathlib import Path
import re
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from rich.console import Console
from rich.panel import Panel


class FactoryMachine:
    """A factory machine with buttons that affect joltage counters."""

    MANUAL_REGEX = re.compile(r'^\[([^\]]*)\]\s*([^{]+)(?:\{([^}]*)\})?$')

    @classmethod
    def from_manual_page(cls, line: str) -> 'FactoryMachine':
        """Parse a line from the half-eaten manual."""
        match = cls.MANUAL_REGEX.match(line)
        if not match:
            raise ValueError(f"The Shiba got this page too: {line!r}")
        lights, buttons, joltage = match.groups()
        return cls(lights, buttons, joltage)

    def __init__(self, light_pattern: str, button_wiring: str, joltage_spec: str):
        self.indicator_lights = self._parse_lights(light_pattern)
        self.button_effects = self._parse_buttons(button_wiring, len(self.indicator_lights))
        self.joltage_targets = self._parse_joltage(joltage_spec)

    def _parse_lights(self, pattern: str) -> np.ndarray:
        """Parse indicator light pattern: . = off, # = on"""
        return np.array([1 if c == '#' else 0 for c in pattern])

    def _parse_buttons(self, wiring: str, n_counters: int) -> np.ndarray:
        """Parse button wiring schematics into effect matrix."""
        buttons = []
        for spec in wiring.split():
            effect = np.zeros(n_counters, dtype=int)
            indices = np.fromstring(spec.strip("()"), sep=",", dtype=int)
            effect[indices] = 1
            buttons.append(effect)
        return np.array(buttons)

    def _parse_joltage(self, spec: str) -> np.ndarray:
        """Parse joltage requirements."""
        return np.fromstring(spec.strip("{}"), sep=",", dtype=int)

    @property
    def n_buttons(self) -> int:
        return self.button_effects.shape[0]

    @property
    def n_counters(self) -> int:
        return len(self.joltage_targets)


def load_factory_floor(file_name: str) -> list[FactoryMachine]:
    """Load machine specs from what remains of the manual."""
    manual_path = Path(__file__).parent / file_name
    with open(manual_path) as f:
        return [FactoryMachine.from_manual_page(line.strip()) for line in f]


def calibrate_machine(machine: FactoryMachine) -> int:
    """
    Find minimum button presses to hit joltage targets.

    ILP formulation:
    - Variables: x[i] = times to press button i
    - Objective: minimize sum(x)
    - Constraint: button_effects.T @ x = joltage_targets
    - Bounds: x >= 0, integers
    """
    # Objective: minimize total presses (coefficient 1 for each button)
    finger_fatigue = np.ones(machine.n_buttons)

    # Constraint: effects must sum to targets exactly
    wiring_matrix = machine.button_effects.T  # (counters x buttons)
    targets = machine.joltage_targets
    joltage_constraint = LinearConstraint(wiring_matrix, targets, targets)

    # Bounds: can't press negative times, no upper limit
    press_limits = Bounds(lb=0, ub=np.inf)

    # All variables must be integers (no half-presses allowed)
    whole_presses_only = np.ones(machine.n_buttons)

    result = milp(
        finger_fatigue,
        constraints=joltage_constraint,
        bounds=press_limits,
        integrality=whole_presses_only
    )

    if not result.success:
        raise ValueError(f"Machine uncalibratable: {result.message}")

    return {
        'presses': int(result.fun),
        'nodes': result.mip_node_count,
        'buttons': machine.n_buttons,
        'counters': machine.n_counters,
    }


def solve(file_name: str) -> dict:
    """Configure all machines, return stats about the solve."""
    machines = load_factory_floor(file_name)
    results = [calibrate_machine(m) for m in machines]

    press_counts = [r['presses'] for r in results]
    total_nodes = sum(r['nodes'] for r in results)
    total_buttons = sum(r['buttons'] for r in results)
    total_counters = sum(r['counters'] for r in results)

    return {
        'total': sum(press_counts),
        'machines': len(machines),
        'buttons': total_buttons,
        'counters': total_counters,
        'nodes': total_nodes,
    }


MACHINE_ART = """\
            [green]◉ ◉ ◉ ◉ ◉[/]     🔴🟢🟢🔴🟢   [bold cyan]3 7 4 2 5[/]
            [dim]└──btns──┘  └──lights───┘ └joltage─┘[/]"""


if __name__ == "__main__":
    console = Console()

    stats = solve('data.dat')

    summary = (
        f"[bold green]{stats['total']:,}[/bold green] button presses\n\n"
        f"{MACHINE_ART}\n\n"
        f"[dim]Machines:[/dim] {stats['machines']:,}  "
        f"[dim]Buttons:[/dim] {stats['buttons']:,}  "
        f"[dim]Counters:[/dim] {stats['counters']:,}  "
        f"[dim]ILP nodes:[/dim] {stats['nodes']:,}"
    )

    console.print(Panel(
        summary,
        title="🎄 Day 10 Part 2 🎅",
        border_style="green",
        expand=False,
        padding=(1, 2)
    ))
