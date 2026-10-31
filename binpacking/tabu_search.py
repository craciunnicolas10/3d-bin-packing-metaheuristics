"""Tabu search over the same (order, orientations) representation used by the genetic algorithm."""
import random
from collections import deque
from typing import List, Optional, Tuple

from .decoder import evaluate
from .genetic_algorithm import History, random_solution
from .models import Box, Container, Solution

Move = tuple


def generate_neighbors(
    solution: Solution,
    boxes: List[Box],
    n_neighbors: int,
    rng: random.Random,
) -> List[Tuple[Solution, Move]]:
    """Neighbors are made by swapping two boxes in the order (70%) or changing one box's orientation (30%)."""
    neighbors = []
    n = len(solution.order)

    for _ in range(n_neighbors):
        neighbor = solution.copy()

        if rng.random() < 0.7:
            i, j = sorted(rng.sample(range(n), 2))
            neighbor.order[i], neighbor.order[j] = neighbor.order[j], neighbor.order[i]
            move = ("swap", i, j)
        else:
            box_idx = neighbor.order[rng.randrange(n)]
            n_orientations = len(boxes[box_idx].get_orientations())
            if n_orientations == 1:
                continue
            new_orientation = (neighbor.orientations[box_idx] + rng.randint(1, n_orientations - 1)) % n_orientations
            neighbor.orientations[box_idx] = new_orientation
            move = ("orientation", box_idx, new_orientation)

        neighbors.append((neighbor, move))

    return neighbors


def tabu_search(
    boxes: List[Box],
    container: Container,
    n_iterations: int = 470,
    tabu_tenure: int = 15,
    n_neighbors: int = 25,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Solution, History]:
    """Move to the best non-tabu neighbor at every step; a tabu move is allowed if it beats the best solution.

    The default 470 iterations x 25 neighbors gives the same evaluation budget as the default GA (~11.8k).
    """
    rng = random.Random(seed)
    current = random_solution(boxes, rng)
    evaluate(current, boxes, container)
    evaluations = 1
    best = current.copy()
    history: History = [(evaluations, best.fitness)]
    tabu_list = deque(maxlen=tabu_tenure)

    for iteration in range(1, n_iterations + 1):
        neighbors = generate_neighbors(current, boxes, n_neighbors, rng)
        if not neighbors:
            break

        best_neighbor, best_move = None, None
        for neighbor, move in neighbors:
            evaluate(neighbor, boxes, container)
            evaluations += 1
            is_tabu = move in tabu_list
            aspiration = neighbor.fitness > best.fitness
            if (not is_tabu or aspiration) and (best_neighbor is None or neighbor.fitness > best_neighbor.fitness):
                best_neighbor, best_move = neighbor, move

        if best_neighbor is None:  # every neighbor is tabu: take the best one anyway
            best_neighbor, best_move = max(neighbors, key=lambda nm: nm[0].fitness)

        current = best_neighbor
        tabu_list.append(best_move)
        if current.fitness > best.fitness:
            best = current.copy()
        history.append((evaluations, best.fitness))

        if verbose and iteration % 50 == 0:
            print(f"    Iter {iteration:4d} | best: {best.fitness:.4f} | current: {current.fitness:.4f}")

    return best, history
