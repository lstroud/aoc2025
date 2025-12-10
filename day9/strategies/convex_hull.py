"""Convex Hull + Vectorized Ray Casting strategy.

Learning exercise demonstrating:
- Numpy broadcasting for vectorized point-in-polygon testing
- Convex hull computation with scipy.spatial.ConvexHull
- Ray casting algorithm for polygon containment

LIMITATIONS:
This approach has fundamental limitations for concave/winding polygons:
1. Convex hull optimization only works for convex polygons - for concave shapes,
   optimal rectangle corners may not be on the hull
2. Point sampling (even with grids) cannot detect when rectangle EDGES cross
   polygon boundaries - a rectangle can have all sampled points inside while
   its edges cut through a winding polygon path

For production use on arbitrary polygons, use Shapely's polygon.contains(box)
which properly handles edge-edge intersection testing.
"""
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
from .base import tile_area


def point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Test if points are inside polygon using vectorized ray casting.

    Casts a ray from each point to the right (+x direction) and counts
    how many polygon edges it crosses. Odd crossings = inside.

    Uses numpy broadcasting to test all points against all edges in one
    operation, creating an (n_points, n_edges) matrix of crossing tests.

    Args:
        points: (n, 2) array of points to test
        polygon: (m, 2) array of polygon vertices in traversal order

    Returns:
        (n,) boolean array - True if point is inside polygon
    """
    # Extract point coordinates as column vectors for broadcasting
    px = points[:, 0]  # (n,)
    py = points[:, 1]  # (n,)

    # Extract edge start and end points
    edge_start = polygon  # (m, 2)
    edge_end = np.roll(polygon, -1, axis=0)  # shift to get next vertex
    x1, y1 = edge_start[:, 0], edge_start[:, 1]  # (m,) each
    x2, y2 = edge_end[:, 0], edge_end[:, 1]  # (m,) each

    # Reshape for broadcasting: points (n,1) vs edges (1,m) -> (n,m)
    y1_arr, y2_arr = y1[None, :], y2[None, :]
    x1_arr, x2_arr = x1[None, :], x2[None, :]
    py_arr, px_arr = py[:, None], px[:, None]

    # Step 1: Which edges span the point's y-coordinate?
    # Edge crosses horizontal line at py if min(y1,y2) <= py < max(y1,y2)
    spans_y = (np.minimum(y1_arr, y2_arr) <= py_arr) & \
              (py_arr < np.maximum(y1_arr, y2_arr))

    # Step 2: Where does edge cross the horizontal line y = py?
    # Linear interpolation: x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
    # Note: produces inf/nan for horizontal edges, but spans_y is False for those
    x_intersect = x1_arr + (py_arr - y1_arr) * (x2_arr - x1_arr) / (y2_arr - y1_arr)

    # Step 3: Is the crossing to the right of our point?
    crossing_right = x_intersect >= px_arr

    # Count crossings: edge must span y AND crossing must be to the right
    crossings = spans_y & crossing_right
    crossing_count = crossings.sum(axis=1)

    # Odd number of crossings = inside polygon
    return crossing_count % 2 == 1


def rectangle_in_polygon(corners: list, polygon: np.ndarray) -> bool:
    """Test if rectangle corners are inside polygon.

    WARNING: This only checks corners, not edges! A rectangle can have all
    4 corners inside while its edges cross polygon boundaries. This gives
    incorrect results for winding/spiral polygons.

    Args:
        corners: List of two opposite corner points [p1, p2]
        polygon: (m, 2) array of polygon vertices

    Returns:
        True if all 4 corners are inside polygon (necessary but not sufficient)
    """
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    # Test all 4 corners, nudged inward to avoid boundary issues
    eps = 0.0001
    four_corners = np.array([
        [min_x + eps, min_y + eps],
        [min_x + eps, max_y - eps],
        [max_x - eps, min_y + eps],
        [max_x - eps, max_y - eps],
    ])

    return point_in_polygon(four_corners, polygon).all()


def solve(coords: list[tuple[int, int]]) -> int:
    """Find largest carpet area using convex hull pruning.

    Only checks pairs of convex hull vertices, reducing O(n²) to O(h²)
    where h is the number of hull vertices (typically much smaller than n).

    NOTE: This optimization only gives correct results for convex polygons.
    For concave polygons, optimal corners may not be on the hull.

    Args:
        coords: Red tile coordinates in polygon traversal order

    Returns:
        Maximum tile count for valid rectangle (approximate for concave polygons)
    """
    points = np.array(coords)

    # Compute convex hull - only hull vertices are candidates
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Find largest valid rectangle from hull vertex pairs
    biggest_carpet = 0
    for p1, p2 in combinations(hull_points, 2):
        if rectangle_in_polygon([p1, p2], points):
            area = tile_area(p1[0], p1[1], p2[0], p2[1])
            biggest_carpet = max(area, biggest_carpet)

    return biggest_carpet