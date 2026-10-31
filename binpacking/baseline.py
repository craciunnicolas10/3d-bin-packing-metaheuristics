"""Greedy constructive baseline: largest boxes first, each placed in its best orientation."""
from typing import List, Set

from .decoder import Bounds, Point, add_placement, blb_key, bounds, first_fit
from .models import Box, Container, Placement, Solution


def greedy_baseline(boxes: List[Box], container: Container) -> Solution:
    """Deterministic reference solution used to judge how much the metaheuristics actually improve.

    Boxes are sorted by decreasing volume. For each box every allowed orientation is tried and the
    one that lands on the lowest bottom-left-back extreme point is kept.
    """
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].volume, reverse=True)
    orientations = [0] * len(boxes)
    placements: List[Placement] = []
    occupied: List[Bounds] = []
    points: Set[Point] = {(0.0, 0.0, 0.0)}

    for box_idx in order:
        best = None
        for ori_idx, dims in enumerate(boxes[box_idx].get_orientations()):
            candidate = first_fit(box_idx, dims, points, occupied, container)
            if candidate and (best is None or blb_key((candidate.x, candidate.y, candidate.z)) < blb_key((best[1].x, best[1].y, best[1].z))):
                best = (ori_idx, candidate)
        if best is not None:
            orientations[box_idx], placement = best
            placements.append(placement)
            occupied.append(bounds(placement))
            points = add_placement(points, placement, container)

    fitness = sum(p.volume for p in placements) / container.volume
    return Solution(order=order, orientations=orientations, fitness=fitness, placements=placements)
