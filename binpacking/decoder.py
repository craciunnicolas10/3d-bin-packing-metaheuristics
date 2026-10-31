"""Extreme-point decoder: turns a solution (order + orientations) into concrete box placements."""
from typing import Iterable, List, Optional, Set, Tuple

from .models import Box, Container, Dimensions, Placement, Solution

Point = Tuple[float, float, float]


def overlaps(p1: Placement, p2: Placement) -> bool:
    return not (
        p1.x + p1.w <= p2.x or p2.x + p2.w <= p1.x or
        p1.y + p1.h <= p2.y or p2.y + p2.h <= p1.y or
        p1.z + p1.d <= p2.z or p2.z + p2.d <= p1.z
    )


def fits_in_container(p: Placement, c: Container) -> bool:
    return (
        p.x >= 0 and p.y >= 0 and p.z >= 0 and
        p.x + p.w <= c.W and
        p.y + p.h <= c.H and
        p.z + p.d <= c.D
    )


def blb_key(point: Point) -> Tuple[float, float, float]:
    """Bottom-left-back priority: lowest z first, then y, then x."""
    return point[2], point[1], point[0]


Bounds = Tuple[float, float, float, float, float, float]


def bounds(p: Placement) -> Bounds:
    return p.x, p.y, p.z, p.x + p.w, p.y + p.h, p.z + p.d


def first_fit(
    box_id: int,
    dims: Dimensions,
    points: Iterable[Point],
    occupied: List[Bounds],
    container: Container,
) -> Optional[Placement]:
    """Place the box at the first extreme point (in BLB order) where it fits without overlapping.

    `occupied` holds the bounds of the boxes already placed. This is the hot loop of every
    optimizer, so the overlap test is inlined instead of calling `overlaps()`.
    """
    w, h, d = dims
    for x, y, z in sorted(points, key=blb_key):
        x2, y2, z2 = x + w, y + h, z + d
        if x2 > container.W or y2 > container.H or z2 > container.D:
            continue
        for ax, ay, az, bx, by, bz in occupied:
            if x < bx and ax < x2 and y < by and ay < y2 and z < bz and az < z2:
                break
        else:
            return Placement(box_id=box_id, x=x, y=y, z=z, w=w, h=h, d=d)
    return None


def add_placement(points: Set[Point], placement: Placement, container: Container) -> Set[Point]:
    """Add the three new extreme points of a placed box and drop points that can no longer host a box.

    A point that lies inside a placed box (or outside the container) would make any box placed there
    overlap or stick out, so removing it never changes the result, it only makes decoding faster.
    """
    p = placement
    points = points | {(p.x + p.w, p.y, p.z), (p.x, p.y + p.h, p.z), (p.x, p.y, p.z + p.d)}
    return {
        (x, y, z) for x, y, z in points
        if x < container.W and y < container.H and z < container.D
        and not (p.x <= x < p.x + p.w and p.y <= y < p.y + p.h and p.z <= z < p.z + p.d)
    }


def decode(solution: Solution, boxes: List[Box], container: Container) -> Tuple[List[Placement], float]:
    """Pack the boxes in the solution's order; boxes that don't fit anywhere are skipped.

    Returns the placements and the fitness (packed volume / container volume).
    """
    placements: List[Placement] = []
    occupied: List[Bounds] = []
    points: Set[Point] = {(0.0, 0.0, 0.0)}

    for box_idx in solution.order:
        orientations = boxes[box_idx].get_orientations()
        dims = orientations[solution.orientations[box_idx] % len(orientations)]
        placement = first_fit(box_idx, dims, points, occupied, container)
        if placement is not None:
            placements.append(placement)
            occupied.append(bounds(placement))
            points = add_placement(points, placement, container)

    fitness = sum(p.volume for p in placements) / container.volume
    return placements, fitness


def evaluate(solution: Solution, boxes: List[Box], container: Container) -> float:
    """Decode the solution, store its placements and fitness, and return the fitness."""
    solution.placements, solution.fitness = decode(solution, boxes, container)
    return solution.fitness
