"""Independent feasibility checker: validates any decoded solution against the problem constraints."""
from dataclasses import dataclass, field
from typing import List

from .decoder import fits_in_container, overlaps
from .models import Box, Container, Placement


@dataclass
class FeasibilityReport:
    is_feasible: bool
    n_placed: int
    n_total: int
    errors: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of the boxes that were placed in the container."""
        return self.n_placed / self.n_total if self.n_total else 0.0


def check_feasibility(placements: List[Placement], boxes: List[Box], container: Container) -> FeasibilityReport:
    """Check that every box is inside the container, uses an allowed orientation and overlaps no other box."""
    errors = []
    box_ids = [p.box_id for p in placements]
    if len(box_ids) != len(set(box_ids)):
        errors.append("A box is placed more than once")

    for p in placements:
        if not fits_in_container(p, container):
            errors.append(
                f"Box {p.box_id} is outside the container: "
                f"corner=({p.x:.2f}, {p.y:.2f}, {p.z:.2f}), size=({p.w:.2f}, {p.h:.2f}, {p.d:.2f})"
            )
        if (p.w, p.h, p.d) not in boxes[p.box_id].get_orientations():
            errors.append(f"Box {p.box_id} has a forbidden orientation: ({p.w}, {p.h}, {p.d})")

    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            if overlaps(placements[i], placements[j]):
                errors.append(f"Boxes {placements[i].box_id} and {placements[j].box_id} overlap")

    return FeasibilityReport(
        is_feasible=not errors,
        n_placed=len(placements),
        n_total=len(boxes),
        errors=errors,
    )


def print_feasibility_report(report: FeasibilityReport) -> None:
    status = "VALID ✓" if report.is_feasible else "INVALID ✗"
    print(f"    Feasibility: {status} | boxes placed: {report.n_placed}/{report.n_total} ({report.coverage * 100:.1f}%)")
    for error in report.errors[:5]:
        print(f"      - {error}")
    if len(report.errors) > 5:
        print(f"      ... and {len(report.errors) - 5} more")
