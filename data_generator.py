import random
from typing import List
from models import Box, Container


def generate_dataset(
    container: Container,
    n_boxes: int,
    min_frac: float = 0.05,
    max_frac: float = 0.40,
    seed: int = None,
    fixed_orientations: bool = False
) -> List[Box]:
    if seed is not None:
        random.seed(seed)

    boxes = []
    for i in range(n_boxes):
        w = round(random.uniform(min_frac * container.W, max_frac * container.W), 2)
        h = round(random.uniform(min_frac * container.H, max_frac * container.H), 2)
        d = round(random.uniform(min_frac * container.D, max_frac * container.D), 2)
        allowed = [(w, h, d)] if fixed_orientations else None
        boxes.append(Box(id=i, w=w, h=h, d=d, allowed_orientations=allowed))

    return boxes


def print_dataset_summary(name: str, boxes: List[Box], container: Container) -> None:
    total_vol = sum(b.w * b.h * b.d for b in boxes)
    max_util = 100.0 * total_vol / container.volume
    print(f"  [{name}] {len(boxes)} cutii | Volum total: {total_vol:.1f} | Max util: {max_util:.1f}%")


CONTAINER = Container(W=200, H=50, D=50)

DATASETS = {
    "small":  generate_dataset(CONTAINER, n_boxes=10, min_frac=0.15, max_frac=0.45, seed=1),
    "medium": generate_dataset(CONTAINER, n_boxes=40, min_frac=0.15, max_frac=0.45, seed=2),
    "large":  generate_dataset(CONTAINER, n_boxes=60, min_frac=0.15, max_frac=0.45, seed=3),
    "custom_manual": [
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
        print_dataset_summary(name, boxes, CONTAINER)