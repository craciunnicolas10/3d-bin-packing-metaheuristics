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


CONTAINER = Container(W=100, H=100, D=100)

DATASETS = {
    "small":        generate_dataset(CONTAINER, n_boxes=10, seed=1),
    "medium":       generate_dataset(CONTAINER, n_boxes=20, seed=2),
    "large":        generate_dataset(CONTAINER, n_boxes=35, seed=3),
    "fixed_orient": generate_dataset(CONTAINER, n_boxes=15, seed=4, fixed_orientations=True),
}


if __name__ == "__main__":
    for name, boxes in DATASETS.items():
        print_dataset_summary(name, boxes, CONTAINER)