"""Benchmark instances: randomly generated datasets plus a hand-made one."""
import random
from typing import Dict, List, Optional

from .models import Box, Container


def generate_dataset(
    container: Container,
    n_boxes: int,
    min_frac: float = 0.05,
    max_frac: float = 0.40,
    seed: Optional[int] = None,
    fixed_orientations: bool = False,
) -> List[Box]:
    """Generate boxes whose sides are a random fraction [min_frac, max_frac] of the container's sides."""
    rng = random.Random(seed)
    boxes = []
    for i in range(n_boxes):
        w = round(rng.uniform(min_frac * container.W, max_frac * container.W), 2)
        h = round(rng.uniform(min_frac * container.H, max_frac * container.H), 2)
        d = round(rng.uniform(min_frac * container.D, max_frac * container.D), 2)
        allowed = [(w, h, d)] if fixed_orientations else None
        boxes.append(Box(id=i, w=w, h=h, d=d, allowed_orientations=allowed))
    return boxes


def dataset_summary(name: str, boxes: List[Box], container: Container) -> str:
    total_volume = sum(b.volume for b in boxes)
    upper_bound = min(100.0, 100.0 * total_volume / container.volume)
    return f"[{name}] {len(boxes)} boxes | total volume: {total_volume:,.0f} | utilization upper bound: {upper_bound:.1f}%"


CONTAINER = Container(W=200, H=50, D=50)

DATASETS: Dict[str, List[Box]] = {
    "small":  generate_dataset(CONTAINER, n_boxes=10, min_frac=0.15, max_frac=0.45, seed=1),
    "medium": generate_dataset(CONTAINER, n_boxes=40, min_frac=0.15, max_frac=0.45, seed=2),
    "large":  generate_dataset(CONTAINER, n_boxes=60, min_frac=0.15, max_frac=0.45, seed=3),
    "custom": [
        Box(id=0,  w=60, h=30, d=25),
        Box(id=1,  w=50, h=40, d=30),
        Box(id=2,  w=70, h=25, d=20),
        Box(id=3,  w=40, h=40, d=35),
        Box(id=4,  w=55, h=30, d=25),
        Box(id=5,  w=45, h=35, d=30),
        Box(id=6,  w=65, h=20, d=25),
        Box(id=7,  w=35, h=45, d=30),
        Box(id=8,  w=50, h=30, d=35),
        Box(id=9,  w=60, h=25, d=30),
        Box(id=10, w=40, h=35, d=25),
        Box(id=11, w=55, h=40, d=20),
        Box(id=12, w=45, h=30, d=35),
        Box(id=13, w=70, h=20, d=30),
        Box(id=14, w=50, h=35, d=25),
    ],
}


if __name__ == "__main__":
    for name, boxes in DATASETS.items():
        print(dataset_summary(name, boxes, CONTAINER))
