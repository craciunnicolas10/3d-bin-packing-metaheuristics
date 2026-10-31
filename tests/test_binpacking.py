import random

import pytest

from binpacking.baseline import greedy_baseline
from binpacking.datasets import CONTAINER, DATASETS, generate_dataset
from binpacking.decoder import decode, evaluate, overlaps
from binpacking.feasibility import check_feasibility
from binpacking.genetic_algorithm import genetic_algorithm, mutate, ox_crossover, random_solution
from binpacking.models import Box, Container, Placement, Solution
from binpacking.tabu_search import tabu_search

SMALL = DATASETS["custom"]


# ---------- Geometry and feasibility ----------

def test_touching_boxes_do_not_overlap():
    a = Placement(0, 0, 0, 0, 10, 10, 10)
    b = Placement(1, 10, 0, 0, 10, 10, 10)
    assert not overlaps(a, b)
    assert overlaps(a, Placement(2, 9.9, 0, 0, 10, 10, 10))


def test_feasibility_detects_box_outside_container():
    boxes = [Box(0, 10, 10, 10)]
    report = check_feasibility([Placement(0, 195, 0, 0, 10, 10, 10)], boxes, CONTAINER)
    assert not report.is_feasible
    assert "outside the container" in report.errors[0]


def test_feasibility_detects_overlap():
    boxes = [Box(0, 10, 10, 10), Box(1, 10, 10, 10)]
    placements = [Placement(0, 0, 0, 0, 10, 10, 10), Placement(1, 5, 5, 5, 10, 10, 10)]
    report = check_feasibility(placements, boxes, CONTAINER)
    assert not report.is_feasible
    assert any("overlap" in e for e in report.errors)


def test_feasibility_detects_forbidden_orientation():
    boxes = [Box(0, 10, 20, 30, allowed_orientations=[(10, 20, 30)])]
    report = check_feasibility([Placement(0, 0, 0, 0, 30, 20, 10)], boxes, CONTAINER)
    assert not report.is_feasible
    assert "forbidden orientation" in report.errors[0]


def test_feasibility_detects_duplicate_box():
    boxes = [Box(0, 10, 10, 10)]
    placements = [Placement(0, 0, 0, 0, 10, 10, 10), Placement(0, 20, 0, 0, 10, 10, 10)]
    assert not check_feasibility(placements, boxes, CONTAINER).is_feasible


def test_orientations_respect_allowed_list():
    assert len(Box(0, 1, 2, 3).get_orientations()) == 6
    assert len(Box(0, 2, 2, 3).get_orientations()) == 3
    assert Box(0, 1, 2, 3, allowed_orientations=[(1, 2, 3)]).get_orientations() == [(1, 2, 3)]


# ---------- Decoder ----------

@pytest.mark.parametrize("seed", range(20))
def test_decoded_solutions_are_always_feasible(seed):
    rng = random.Random(seed)
    boxes = DATASETS["medium"]
    placements, fitness = decode(random_solution(boxes, rng), boxes, CONTAINER)
    assert check_feasibility(placements, boxes, CONTAINER).is_feasible
    assert fitness == pytest.approx(sum(p.volume for p in placements) / CONTAINER.volume)


def test_decoder_packs_boxes_bottom_left_back():
    container = Container(20, 10, 10)
    boxes = [Box(0, 10, 10, 10), Box(1, 10, 10, 10)]
    placements, fitness = decode(Solution(order=[0, 1], orientations=[0, 0]), boxes, container)
    assert [(p.x, p.y, p.z) for p in placements] == [(0, 0, 0), (10, 0, 0)]
    assert fitness == 1.0


def test_box_that_does_not_fit_is_skipped():
    container = Container(10, 10, 10)
    boxes = [Box(0, 10, 10, 10), Box(1, 5, 5, 5)]
    placements, fitness = decode(Solution(order=[0, 1], orientations=[0, 0]), boxes, container)
    assert [p.box_id for p in placements] == [0]
    assert fitness == 1.0


# ---------- Genetic operators ----------

def test_crossover_and_mutation_keep_valid_permutations():
    rng = random.Random(0)
    p1, p2 = random_solution(SMALL, rng), random_solution(SMALL, rng)
    for _ in range(50):
        c1, c2 = ox_crossover(p1, p2, rng)
        for child in (c1, c2, mutate(c1, SMALL, rng, p_inversion=1.0, p_orientation=0.5)):
            assert sorted(child.order) == list(range(len(SMALL)))
            assert all(0 <= o < len(b.get_orientations()) for o, b in zip(child.orientations, SMALL))


# ---------- Optimizers ----------

@pytest.mark.parametrize("optimizer, kwargs", [
    (genetic_algorithm, {"pop_size": 20, "n_generations": 15}),
    (tabu_search, {"n_iterations": 40, "n_neighbors": 10}),
])
def test_optimizers_are_feasible_reproducible_and_beat_random(optimizer, kwargs):
    best, history = optimizer(SMALL, CONTAINER, seed=7, **kwargs)
    again, _ = optimizer(SMALL, CONTAINER, seed=7, **kwargs)

    assert check_feasibility(best.placements, SMALL, CONTAINER).is_feasible
    assert best.fitness == again.fitness
    assert [f for _, f in history] == sorted(f for _, f in history)  # best-so-far never decreases
    assert best.fitness >= history[0][1]
    assert evaluate(best.copy(), SMALL, CONTAINER) == pytest.approx(best.fitness)


def test_tabu_search_handles_boxes_with_a_single_orientation():
    boxes = generate_dataset(CONTAINER, n_boxes=8, seed=1, fixed_orientations=True)
    best, _ = tabu_search(boxes, CONTAINER, n_iterations=20, n_neighbors=10, seed=0)
    assert check_feasibility(best.placements, boxes, CONTAINER).is_feasible


def test_greedy_baseline_is_feasible():
    for boxes in DATASETS.values():
        solution = greedy_baseline(boxes, CONTAINER)
        assert check_feasibility(solution.placements, boxes, CONTAINER).is_feasible
        assert 0 < solution.fitness <= 1
