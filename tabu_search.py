import random
from collections import deque
from typing import List, Tuple

from models import Box, Container, Solution
from decoder import evaluate


def generate_neighbors(
    sol: Solution,
    boxes: List[Box],
    n_neighbors: int = 20
) -> List[Tuple[Solution, tuple]]:
    neighbors = []
    n = len(sol.order)

    for _ in range(n_neighbors):
        neighbor = sol.copy()

        if random.random() < 0.7:
            i, j = sorted(random.sample(range(n), 2))
            neighbor.order[i], neighbor.order[j] = neighbor.order[j], neighbor.order[i]
            move = ("swap", i, j)
        else:
            i = random.randint(0, n - 1)
            box_idx = neighbor.order[i]
            n_oris = len(boxes[box_idx].get_orientations())
            old_ori = neighbor.orientations[box_idx]
            new_ori = (old_ori + random.randint(1, n_oris - 1)) % n_oris
            neighbor.orientations[box_idx] = new_ori
            move = ("ori", box_idx, new_ori)

        neighbors.append((neighbor, move))

    return neighbors


def tabu_search(
    boxes: List[Box],
    container: Container,
    n_iterations: int = 300,
    tabu_tenure: int = 15,
    n_neighbors: int = 25,
    verbose: bool = True
) -> Tuple[Solution, List[float]]:
    current = _random_solution(boxes)
    evaluate(current, boxes, container)
    best = current.copy()
    history = [best.fitness]

    tabu_list = deque(maxlen=tabu_tenure)

    for iteration in range(n_iterations):
        neighbors = generate_neighbors(current, boxes, n_neighbors)

        best_neighbor = None
        best_neighbor_move = None

        for neighbor, move in neighbors:
            evaluate(neighbor, boxes, container)
            is_tabu = move in tabu_list
            aspiration = neighbor.fitness > best.fitness

            if not is_tabu or aspiration:
                if best_neighbor is None or neighbor.fitness > best_neighbor.fitness:
                    best_neighbor = neighbor
                    best_neighbor_move = move

        if best_neighbor is None:
            best_neighbor, best_neighbor_move = max(
                neighbors, key=lambda x: x[0].fitness
            )

        current = best_neighbor
        tabu_list.append(best_neighbor_move)

        if current.fitness > best.fitness:
            best = current.copy()

        history.append(best.fitness)

        if verbose and (iteration + 1) % 50 == 0:
            print(f"    Iter {iteration + 1:4d} | Best: {best.fitness:.4f} "
                  f"| Current: {current.fitness:.4f}")

    return best, history


def _random_solution(boxes: List[Box]) -> Solution:
    n = len(boxes)
    order = list(range(n))
    random.shuffle(order)
    orientations = [
        random.randint(0, len(boxes[i].get_orientations()) - 1)
        for i in range(n)
    ]
    return Solution(order=order, orientations=orientations)