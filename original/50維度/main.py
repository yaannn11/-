if __name__ == "__main__":
    t_start = time.time()
    
    # Configuration
    dim = 50
    max_nfe = 1000
    seeds = [10, 20, 30, 42, 50]
    functions = ['Ellipsoid', 'Rosenbrock', 'Ackley', 'Griewank', 'SRR', 'RHC1', 'RHC2']
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
        }if __name__ == "__main__":
    t_start = time.time()
    
    # Configuration
    dim = 50
    max_nfe = 1000
    seeds = [10, 20, 30, 42, 50]
    functions = ['Ellipsoid', 'Rosenbrock', 'Ackley', 'Griewank', 'SRR', 'RHC1', 'RHC2']
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
    t_elapsed = time.time() - t_start
    print("=====================================================================")
    print(f"All experiments completed successfully in {t_elapsed:.2f} seconds.")
    print("=====================================================================")
