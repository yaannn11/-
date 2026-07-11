
import time
import multiprocessing
import numpy as np
import config
from benchmark_fn import get_benchmark_function
from esa_agent import EvolutionarySamplingAgent
from visualizer import ESAVisualizer
from data_manager import ESADataManager  

def run_experiment_task(args):
    #執行一個測試函數在特定隨機種子下的實驗
    func_name, seed, dim, max_nfe = args
    func = get_benchmark_function(func_name, dim)
    np.random.seed(seed)
    
    esa = EvolutionarySamplingAgent(func, dim, func.lb, func.ub, max_nfe=max_nfe)
    

    result_dict = esa.run()
    
    if 'q_table' in result_dict and hasattr(result_dict['q_table'], 'tolist'):
        result_dict['q_table'] = result_dict['q_table'].tolist()
        
    result_dict.pop('best_x', None) 
    
    result_dict.update({
        'func_name': func_name,
        'seed': seed
    })
    
    return result_dict



if __name__ == "__main__":
    t_start = time.time()
    
    print(f"Dimension: {config.dim}D | NFE Budget: {config.max_nfe} | Seeds: {config.seeds}")
    print("Running experiments in parallel...")
    
    task_args = []
    for f in config.functions:
        for s in config.seeds:
            task_args.append((f, s, config.dim, config.max_nfe))
            
    # Execute in parallel
    pool = multiprocessing.Pool(processes=config.MAX_WORKERS)
    results_list = pool.map(run_experiment_task, task_args)
    pool.close()
    pool.join()
    
    results_dict = {f: {} for f in config.functions}

    for res in results_list:
        data = res.copy()
        func_name = data.pop('func_name')
        seed = data.pop('seed')
        
        results_dict[func_name][seed] = data

    summary_stats = {}
    for f in config.functions:
        vals = [results_dict[f][s]['best_y'] for s in config.seeds]
        summary_stats[f] = {
            'mean': np.mean(vals), 'std': np.std(vals), 
            'best': np.min(vals), 'worst': np.max(vals)
        }

    #輸出詳細結果資料
    dm = ESADataManager()
    dm.save_text_report(results_dict, summary_stats)
    dm.save_analysis_csv(results_dict)
    dm.generate_action_analysis()

    # 繪圖：繪製收斂曲線、性能對比圖、以及安全邊界圖
    viz = ESAVisualizer(config.base_dir)
    viz.plot_convergence_curves(config.functions, config.seeds, results_dict)
    viz.plot_performance_comparison(config.functions, summary_stats)
    viz.plot_safety_dual_bounds(config.functions, config.seeds, results_dict)
    
    print("=====================================================================")
    print(f"All experiments completed successfully in {time.time() - t_start:.2f} seconds.")
