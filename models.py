from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Optional, Tuple


@dataclass
class Box:
    id: int
    w: float
    h: float
    d: float
    allowed_orientations: Optional[List[Tuple]] = None

    def get_orientations(self) -> List[Tuple[float, float, float]]:
        dims = (self.w, self.h, self.d)
        all_perms = list(set(permutations(dims)))
        if self.allowed_orientations is None:
            return all_perms
        return [p for p in all_perms if p in self.allowed_orientations]


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
    box_id: int
    x: float
    y: float
    z: float
    w: float
    h: float
    d: float


@dataclass
class Solution:
    order: List[int]
    orientations: List[int]
    fitness: float = 0.0
    placements: List[Placement] = field(default_factory=list)

    def copy(self) -> "Solution":
        import copy
        return Solution(
            order=self.order[:],
            orientations=self.orientations[:],
            fitness=self.fitness,
            placements=copy.deepcopy(self.placements)
        )