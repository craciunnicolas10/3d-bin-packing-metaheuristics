"""Genetic algorithm with tournament selection, order crossover (OX) and inversion mutation."""
import random
from typing import List, Optional, Tuple

from .decoder import evaluate
from .models import Box, Container, Solution

# (number of fitness evaluations so far, best fitness so far)
History = List[Tuple[int, float]]


def random_solution(boxes: List[Box], rng: random.Random) -> Solution:
    order = list(range(len(boxes)))
    rng.shuffle(order)
    orientations = [rng.randrange(len(box.get_orientations())) for box in boxes]
    return Solution(order=order, orientations=orientations)


def tournament_select(population: List[Solution], k: int, rng: random.Random) -> Solution:
    return max(rng.sample(population, k), key=lambda s: s.fitness)


def ox_crossover(p1: Solution, p2: Solution, rng: random.Random) -> Tuple[Solution, Solution]:
    """Order crossover on the packing order, uniform crossover on the orientations."""
    n = len(p1.order)
    a, b = sorted(rng.sample(range(n), 2))

    def make_child(parent_a: Solution, parent_b: Solution) -> Solution:
        child_order = [-1] * n
        child_order[a:b + 1] = parent_a.order[a:b + 1]
        segment = set(child_order[a:b + 1])
        remaining = iter(x for x in parent_b.order if x not in segment)
        for i in list(range(b + 1, n)) + list(range(0, a)):
            child_order[i] = next(remaining)
        child_orientations = [
            parent_a.orientations[i] if rng.random() < 0.5 else parent_b.orientations[i]
            for i in range(n)
        ]
        return Solution(order=child_order, orientations=child_orientations)

    return make_child(p1, p2), make_child(p2, p1)


def mutate(
    solution: Solution,
    boxes: List[Box],
    rng: random.Random,
    p_inversion: float = 0.2,
    p_orientation: float = 0.1,
) -> Solution:
    """Reverse a random segment of the order and/or re-roll the orientation of a few boxes."""
    child = solution.copy()
    n = len(child.order)

    if rng.random() < p_inversion:
        i, j = sorted(rng.sample(range(n), 2))
        child.order[i:j + 1] = child.order[i:j + 1][::-1]

    for box_idx in child.order:
        if rng.random() < p_orientation:
            child.orientations[box_idx] = rng.randrange(len(boxes[box_idx].get_orientations()))

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
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Solution, History]:
    """Evolve a population of packing orders/orientations and return the best solution found."""
    rng = random.Random(seed)
    population = [random_solution(boxes, rng) for _ in range(pop_size)]
    for solution in population:
        evaluate(solution, boxes, container)
    evaluations = pop_size

    best = max(population, key=lambda s: s.fitness).copy()
    history: History = [(evaluations, best.fitness)]

    for generation in range(1, n_generations + 1):
        ranked = sorted(population, key=lambda s: s.fitness, reverse=True)
        new_population = [s.copy() for s in ranked[:elitism]]

        while len(new_population) < pop_size:
            p1 = tournament_select(population, k_tournament, rng)
            p2 = tournament_select(population, k_tournament, rng)

            if rng.random() < crossover_rate:
                c1, c2 = ox_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if rng.random() < mutation_rate:
                c1 = mutate(c1, boxes, rng)
            if rng.random() < mutation_rate:
                c2 = mutate(c2, boxes, rng)

            evaluate(c1, boxes, container)
            evaluate(c2, boxes, container)
            evaluations += 2
            new_population.extend([c1, c2])

        population = new_population[:pop_size]
        generation_best = max(population, key=lambda s: s.fitness)
        if generation_best.fitness > best.fitness:
            best = generation_best.copy()
        history.append((evaluations, best.fitness))

        if verbose and generation % 25 == 0:
            average = sum(s.fitness for s in population) / len(population)
            print(f"    Gen {generation:4d} | best: {best.fitness:.4f} | average: {average:.4f}")

    return best, history
