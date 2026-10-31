"""Core data structures: boxes, the container, placements and candidate solutions."""
import copy
from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Optional, Tuple

Dimensions = Tuple[float, float, float]


@dataclass
class Box:
    """A rectangular box with dimensions (w, h, d) and an optional list of allowed orientations."""
    id: int
    w: float
    h: float
    d: float
    allowed_orientations: Optional[List[Dimensions]] = None

    @property
    def volume(self) -> float:
        return self.w * self.h * self.d

    def get_orientations(self) -> List[Dimensions]:
        """All distinct axis-aligned rotations of the box, filtered by the allowed orientations."""
        rotations = sorted(set(permutations((self.w, self.h, self.d))))
        if self.allowed_orientations is None:
            return rotations
        return [r for r in rotations if r in self.allowed_orientations]


@dataclass
class Container:
    W: float
    H: float
    D: float

    @property
    def volume(self) -> float:
        return self.W * self.H * self.D


@dataclass
class Placement:
    """A box placed in the container: (x, y, z) is its bottom-left-back corner, (w, h, d) its oriented size."""
    box_id: int
    x: float
    y: float
    z: float
    w: float
    h: float
    d: float

    @property
    def volume(self) -> float:
        return self.w * self.h * self.d


@dataclass
class Solution:
    """Chromosome: the order in which boxes are packed plus an orientation index for every box."""
    order: List[int]
    orientations: List[int]
    fitness: float = 0.0
    placements: List[Placement] = field(default_factory=list)

    def copy(self) -> "Solution":
        return Solution(
            order=self.order[:],
            orientations=self.orientations[:],
            fitness=self.fitness,
            placements=copy.deepcopy(self.placements),
        )
