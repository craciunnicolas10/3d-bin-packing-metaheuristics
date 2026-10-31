"""Run the GA vs. tabu search experiment on every dataset and save the results to results/."""
import argparse
import statistics
import time
from pathlib import Path

from binpacking.baseline import greedy_baseline
from binpacking.datasets import CONTAINER, DATASETS, dataset_summary
from binpacking.feasibility import check_feasibility, print_feasibility_report
from binpacking.genetic_algorithm import genetic_algorithm
from binpacking.tabu_search import tabu_search
from binpacking.visualizer import plot_comparison, plot_convergence, plot_packings

ALGORITHMS = {"ga": genetic_algorithm, "ts": tabu_search}
RESULTS_DIR = Path("results")


def run_method(method: str, boxes, seed: int, verbose: bool) -> dict:
    start = time.perf_counter()
    solution, history = ALGORITHMS[method](boxes, CONTAINER, seed=seed, verbose=verbose)
    elapsed = time.perf_counter() - start

    report = check_feasibility(solution.placements, boxes, CONTAINER)
    if not report.is_feasible:
        print_feasibility_report(report)
        raise RuntimeError(f"{method} produced an infeasible solution (seed {seed})")

    return {
        "utilization": 100 * solution.fitness,
        "placed": report.n_placed,
        "time": elapsed,
        "history": history,
        "placements": solution.placements,
    }


def run_dataset(name: str, n_runs: int, base_seed: int, verbose: bool) -> dict:
    boxes = DATASETS[name]
    print(f"\n{'=' * 70}\n{dataset_summary(name, boxes, CONTAINER)}\n{'=' * 70}")

    start = time.perf_counter()
    baseline = greedy_baseline(boxes, CONTAINER)
    baseline_time = time.perf_counter() - start
    report = check_feasibility(baseline.placements, boxes, CONTAINER)
    print(f"  Greedy baseline: {100 * baseline.fitness:.2f}%")
    print_feasibility_report(report)

    result = {
        "n_boxes": len(boxes),
        "baseline": {"utilization": 100 * baseline.fitness, "placed": report.n_placed,
                     "time": baseline_time, "placements": baseline.placements},
    }
    for method in ALGORITHMS:
        runs = []
        for run in range(n_runs):
            r = run_method(method, boxes, seed=base_seed + run, verbose=verbose)
            print(f"  {method.upper()} run {run + 1}/{n_runs}: {r['utilization']:.2f}% "
                  f"({r['placed']}/{len(boxes)} boxes, {r['time']:.1f}s) - feasible ✓")
            runs.append(r)
        result[method] = {"runs": runs}
    return result


def summary_table(results: dict) -> str:
    lines = [
        "| Dataset | Boxes | Greedy baseline | GA mean ± std | GA best | GA time | TS mean ± std | TS best | TS time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        row = [name, str(res["n_boxes"]), f"{res['baseline']['utilization']:.1f}%"]
        for method in ALGORITHMS:
            utils = [r["utilization"] for r in res[method]["runs"]]
            std = statistics.pstdev(utils)
            row += [f"{statistics.mean(utils):.1f}% ± {std:.1f}", f"{max(utils):.1f}%",
                    f"{statistics.mean(r['time'] for r in res[method]['runs']):.1f}s"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D bin packing: genetic algorithm vs. tabu search")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                        help="datasets to run (default: all)")
    parser.add_argument("--runs", type=int, default=5, help="independent runs per method (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first run (default: 0)")
    parser.add_argument("--show", action="store_true", help="open the plots in a window as well")
    parser.add_argument("--verbose", action="store_true", help="print progress during each run")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    results = {name: run_dataset(name, args.runs, args.seed, args.verbose) for name in args.datasets}

    table = summary_table(results)
    print(f"\n{table}")
    (RESULTS_DIR / "summary.md").write_text(
        f"Results over {args.runs} runs per method (seeds {args.seed}..{args.seed + args.runs - 1}).\n\n{table}\n",
        encoding="utf-8",
    )

    for name, res in results.items():
        best = {"baseline": res["baseline"]["placements"]}
        for method in ALGORITHMS:
            best[method] = max(res[method]["runs"], key=lambda r: r["utilization"])["placements"]
        plot_packings(name, best, res["n_boxes"], CONTAINER, str(RESULTS_DIR / f"packing_{name}.png"), args.show)
    plot_convergence(results, str(RESULTS_DIR / "convergence.png"), args.show)
    plot_comparison(results, str(RESULTS_DIR / "comparison.png"), args.show)
    print(f"\nPlots and summary saved to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
