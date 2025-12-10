"""Shapely polygon containment strategy.

Approach:
- Build Polygon from red tile coordinates (they're in traversal order)
- For each pair of red tiles, create a box
- Check if polygon.contains(box)
- Track maximum valid area

Tools: shapely.Polygon, shapely.box, itertools.combinations
"""
from itertools import combinations
from shapely import Polygon, box
from .base import tile_area


def solve(coords: list[tuple[int, int]]) -> int:
    """Find largest carpet area using Shapely containment.

    Args:
        coords: Red tile coordinates in polygon traversal order

    Returns:
        Maximum tile count for valid rectangle
    """
    # The elves' renovation zone - only red and green tiles allowed
    renovation_zone = Polygon(coords)

    if not renovation_zone.is_valid:
        raise ValueError("The elves drew an impossible floor plan. Again.")

    # Find the biggest carpet that fits in the renovation zone
    # with red tiles at opposite corners (elf union rules, don't ask)
    biggest_carpet = 0

    for corner1, corner2 in combinations(coords, 2):
        carpet_candidate = box(corner1[0], corner1[1], corner2[0], corner2[1])

        if renovation_zone.contains(carpet_candidate):
            carpet_size = tile_area(corner1[0], corner1[1], corner2[0], corner2[1])
            if carpet_size > biggest_carpet:
                biggest_carpet = carpet_size

    return biggest_carpet
