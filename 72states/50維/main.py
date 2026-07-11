import os
import time
import multiprocessing

from experiment.runner import run_experiment_task
from experiment.report import write_report
from experiment.plots import plot_convergence, plot_performance
from experiment.csv_export import export_all

if __name__ == "__main__":
    t_start = time.time()
    
    # Configuration
    dim = 50
    max_nfe = 1000
    seeds = [10, 20, 30, 42, 50]
    functions = ['Ellipsoid', 'Rosenbrock', 'Ackley', 'Griewank', 'SRR', 'RHC1', 'RHC2']
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chart_dir = os.path.join(base_dir,"charts")
    os.makedirs(os.path.join(base_dir, 'charts'), exist_ok=True)
    
    print("=====================================================================")
    print("Evolutionary Sampling Agent (ESA) Parallel 7 Run (Normalized RBF)")
    print("=====================================================================")
    print(f"Dimension: {dim}D | NFE Budget: {max_nfe} | Seeds: {seeds}")
    print("Running experiments in parallel...")
    
    task_args = []
    for f in functions:
        for s in seeds:
            task_args.append((f, s, dim, max_nfe))
            
    # Execute in parallel
    pool = multiprocessing.Pool(processes=min(len(task_args), os.cpu_count() or 4))
    results_list = pool.map(run_experiment_task, task_args)
    pool.close()
    pool.join()
    
    results_dict = {f: {} for f in functions}
    for res in results_list:
        results_dict[res['func_name']][res['seed']] = {
            'best_y': res['best_y'],
            'history_nfe': res['history_nfe'],
            'history_fitness': res['history_fitness'],
            'rbf_records': res['rbf_records'],
            'eval_records': res['eval_records'],
            'q_table': res['q_table'],
            'action_counts': res['action_counts']
        }
        # ================================
    # Generate report
    # ================================
    output_dir = base_dir

    print("Generating statistical report...")

    report_path = os.path.join(
        output_dir,
        "esa_experiments_output.txt"
    )

    summary_stats = write_report(results_dict, functions, seeds, output_dir)

    # Generate plots
    print("Generating convergence plots...")
    plot_convergence(results_dict, functions, seeds, chart_dir)

    print("Generating statistical summary bar chart...")
    plot_performance(summary_stats, functions, chart_dir)

    # Export CSV
    print("Generating analysis CSV files...")
    export_all(results_dict, functions, seeds, output_dir)

    elapsed_time = time.time() - t_start
    print("=====================================================================")
    print(f"All experiments completed successfully in {elapsed_time:.2f} seconds.")
    print("=====================================================================")
    