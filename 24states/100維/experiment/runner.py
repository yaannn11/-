import numpy as np

from benchmark import get_benchmark_function
from models.esa import EvolutionarySamplingAgent

def run_experiment_task(args):
    func_name, seed, dim, max_nfe = args
    func = get_benchmark_function(func_name, dim)
    np.random.seed(seed)
    
    esa = EvolutionarySamplingAgent(func, dim, func.lb, func.ub, max_nfe=max_nfe)
    best_x, best_y, history_nfe, history_fitness, rbf_records, eval_records, q_table, action_counts = esa.run()
    
    return {
        'func_name': func_name,
        'seed': seed,
        'best_y': best_y,
        'history_nfe': history_nfe,
        'history_fitness': history_fitness,
        'rbf_records': rbf_records,
        'eval_records': eval_records,
        'q_table': q_table.tolist() if hasattr(q_table, 'tolist') else q_table,
        'action_counts': action_counts
    }
