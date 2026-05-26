from typing import List, Tuple
from models import Box, Container, Placement, Solution


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


def decode_blb(
    solution: Solution,
    boxes: List[Box],
    container: Container
) -> Tuple[List[Placement], float]:
    placements: List[Placement] = []
    extreme_points = [(0.0, 0.0, 0.0)]
    placed_volume = 0.0

    for box_idx in solution.order:
        box = boxes[box_idx]
        orientations = box.get_orientations()
        ori_idx = solution.orientations[box_idx] % len(orientations)
        bw, bh, bd = orientations[ori_idx]

        sorted_eps = sorted(extreme_points, key=lambda p: (p[2], p[1], p[0]))

        for (ex, ey, ez) in sorted_eps:
            candidate = Placement(box_id=box_idx, x=ex, y=ey, z=ez, w=bw, h=bh, d=bd)

            if not fits_in_container(candidate, container):
                continue
            if any(overlaps(candidate, p) for p in placements):
                continue

            placements.append(candidate)
            placed_volume += bw * bh * bd
            extreme_points.append((ex + bw, ey, ez))
            extreme_points.append((ex, ey + bh, ez))
            extreme_points.append((ex, ey, ez + bd))
            break

    fitness = placed_volume / container.volume
    return placements, fitness


def evaluate(solution: Solution, boxes: List[Box], container: Container) -> float:
    placements, fitness = decode_blb(solution, boxes, container)
    solution.fitness = fitness
    solution.placements = placements
    return fitness