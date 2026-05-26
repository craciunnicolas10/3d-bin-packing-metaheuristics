from typing import Dict, List
from models import Box, Container, Placement
from decoder import fits_in_container, overlaps


def check_feasibility(
    placements: List[Placement],
    boxes: List[Box],
    container: Container
) -> Dict:
    errors = []

    for p in placements:
        if not fits_in_container(p, container):
            errors.append(
                f"Cutia {p.box_id} iese din container: "
                f"start=({p.x:.2f},{p.y:.2f},{p.z:.2f}), "
                f"dims=({p.w:.2f},{p.h:.2f},{p.d:.2f})"
            )

        box = boxes[p.box_id]
        valid_orientations = box.get_orientations()
        if (p.w, p.h, p.d) not in valid_orientations:
            errors.append(
                f"Cutia {p.box_id} are orientare nepermisă: ({p.w},{p.h},{p.d})"
            )

    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            if overlaps(placements[i], placements[j]):
                errors.append(
                    f"Suprapunere: cutia {placements[i].box_id} și {placements[j].box_id}"
                )

    n_total = len(boxes)
    n_placed = len(placements)

    return {
        "is_feasible": len(errors) == 0,
        "n_placed":    n_placed,
        "n_total":     n_total,
        "coverage":    n_placed / n_total if n_total > 0 else 0.0,
        "errors":      errors,
    }


def print_feasibility_report(result: Dict) -> None:
    status = "VALID ✓" if result["is_feasible"] else "INVALID ✗"
    print(f"  Fezabilitate: {status}")
    print(f"  Cutii plasate: {result['n_placed']}/{result['n_total']} "
          f"({result['coverage']*100:.1f}%)")
    if result["errors"]:
        for err in result["errors"][:5]:
            print(f"    - {err}")
        if len(result["errors"]) > 5:
            print(f"    ... și {len(result['errors']) - 5} mai multe")