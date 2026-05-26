import time
from data_generator import CONTAINER, DATASETS
from generic_algorithm import genetic_algorithm
from tabu_search import tabu_search
from feasibility import check_feasibility, print_feasibility_report
from visualizer import plot_solution_3d, plot_convergence, plot_comparison_bars

def main():
    dataset_name = "small"
    boxes = DATASETS[dataset_name]
    
    print(f"=== Starting Evaluation for Dataset: {dataset_name} ===")

    print("\n[1] Running Genetic Algorithm (GA)...")
    ga_start = time.time()
    ga_solution, ga_history = genetic_algorithm(
        boxes=boxes, 
        container=CONTAINER, 
        pop_size=40,          
        n_generations=50,
        verbose=True
    )
    ga_duration = time.time() - ga_start
    
    ga_report = check_feasibility(ga_solution.placements, boxes, CONTAINER)
    print_feasibility_report(ga_report)

    print("\n[2] Running Tabu Search (TS)...")
    ts_start = time.time()
    ts_solution, ts_history = tabu_search(
        boxes=boxes, 
        container=CONTAINER, 
        n_iterations=80,      
        verbose=True
    )
    ts_duration = time.time() - ts_start
    
    ts_report = check_feasibility(ts_solution.placements, boxes, CONTAINER)
    print_feasibility_report(ts_report)

    print("\n=> Displaying 3D Container Plot...")
    plot_solution_3d(ga_solution.placements, CONTAINER, title=f"GA Solution (Dataset: {dataset_name})")

    performance_results = {
        dataset_name: {
            "n_boxes": len(boxes),
            "ga": {
                "history": ga_history,
                "space_util": ga_solution.fitness * 100,
                "time": round(ga_duration, 3)
            },
            "ts": {
                "history": ts_history,
                "space_util": ts_solution.fitness * 100,
                "time": round(ts_duration, 3)
            }
        }
    }
    
    print("\n=> Displaying Convergence Plot...")
    plot_convergence(performance_results)
    
    print("\n=> Displaying Metric Comparison Bars...")
    plot_comparison_bars(performance_results)


if __name__ == "__main__":
    main()