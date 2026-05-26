import random
from typing import List, Tuple
import numpy as np

from models import Box, Container, Solution
from decoder import evaluate


def random_solution(boxes: List[Box]) -> Solution:
    n = len(boxes)
    order = list(range(n))
    random.shuffle(order)
    orientations = [
        random.randint(0, len(boxes[i].get_orientations()) - 1)
        for i in range(n)
    ]
    return Solution(order=order, orientations=orientations)


def tournament_select(population: List[Solution], k: int = 3) -> Solution:
    contestants = random.sample(population, k)
    return max(contestants, key=lambda s: s.fitness)


def ox_crossover(p1: Solution, p2: Solution) -> Tuple[Solution, Solution]:
    n = len(p1.order)
    a, b = sorted(random.sample(range(n), 2))

    def _ox(parent_a: Solution, parent_b: Solution) -> Solution:
        child_order = [-1] * n
        child_order[a:b + 1] = parent_a.order[a:b + 1]
        segment = set(child_order[a:b + 1])
        remaining = [x for x in parent_b.order if x not in segment]
        ptr = 0
        for i in list(range(b + 1, n)) + list(range(0, a)):
            child_order[i] = remaining[ptr]
            ptr += 1
        child_oris = [
            parent_a.orientations[i] if random.random() < 0.5 else parent_b.orientations[i]
            for i in range(n)
        ]
        return Solution(order=child_order, orientations=child_oris)

    return _ox(p1, p2), _ox(p2, p1)


def mutate(sol: Solution, boxes: List[Box], p_swap: float = 0.2, p_ori: float = 0.1) -> Solution:
    child = sol.copy()
    n = len(child.order)

    if random.random() < p_swap:
        i, j = sorted(random.sample(range(n), 2))
        child.order[i:j + 1] = child.order[i:j + 1][::-1]

    for i in range(n):
        if random.random() < p_ori:
            box_idx = child.order[i]
            n_oris = len(boxes[box_idx].get_orientations())
            child.orientations[box_idx] = random.randint(0, n_oris - 1)

    return child


def genetic_algorithm(
    boxes: List[Box],
    container: Container,
    pop_size: int = 80,
    n_generations: int = 150,
    k_tournament: int = 3,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.15,
    elitism: int = 2,
    verbose: bool = True
) -> Tuple[Solution, List[float]]:
    population = [random_solution(boxes) for _ in range(pop_size)]
    for sol in population:
        evaluate(sol, boxes, container)

    best = max(population, key=lambda s: s.fitness).copy()
    history = [best.fitness]

    for gen in range(n_generations):
        new_pop = []
        sorted_pop = sorted(population, key=lambda s: s.fitness, reverse=True)
        new_pop.extend([s.copy() for s in sorted_pop[:elitism]])

        while len(new_pop) < pop_size:
            p1 = tournament_select(population, k_tournament)
            p2 = tournament_select(population, k_tournament)

            if random.random() < crossover_rate:
                c1, c2 = ox_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if random.random() < mutation_rate:
                c1 = mutate(c1, boxes)
            if random.random() < mutation_rate:
                c2 = mutate(c2, boxes)

            evaluate(c1, boxes, container)
            evaluate(c2, boxes, container)
            new_pop.extend([c1, c2])

        population = new_pop[:pop_size]
        gen_best = max(population, key=lambda s: s.fitness)
        if gen_best.fitness > best.fitness:
            best = gen_best.copy()

        history.append(best.fitness)

        if verbose and (gen + 1) % 25 == 0:
            avg = np.mean([s.fitness for s in population])
            print(f"    Gen {gen + 1:4d} | Best: {best.fitness:.4f} | Avg: {avg:.4f}")

    return best, history