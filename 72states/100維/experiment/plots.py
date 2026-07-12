import os
import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(results_dict, functions, seeds, base_dir):
    print("Generating convergence plots...")
    chart_dir = os.path.join(base_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    for f in functions:
        plt.figure(figsize=(8, 5))
        for s in seeds:
            hist_nfe = results_dict[f][s]['history_nfe']
            hist_fitness = results_dict[f][s]['history_fitness']
            plt.plot(hist_nfe, hist_fitness, label=f"Seed {s}", alpha=0.8)
            
        plt.yscale('log')
        plt.xlabel('Number of Function Evaluations (NFE)')
        plt.ylabel('Best Fitness (Log Scale)')
        plt.title(f"Convergence Curves: {f} (Normalized RBF)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, f"convergence_{f.lower()}.png"), dpi=200)
        plt.close()
        
def plot_performance(summary_stats, functions, base_dir):        
    # 5. Generate Bar Chart Performance Comparison
    print("Generating statistical summary bar chart...")
    chart_dir = os.path.join(base_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)    
    means = [summary_stats[f]['mean'] for f in functions]
    stds = [summary_stats[f]['std'] for f in functions]
    bests = [summary_stats[f]['best'] for f in functions]
    
    x_indices = np.arange(len(functions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x_indices - width/2, means, width, yerr=stds, label='Mean Fitness', color='#3498db', capsize=5)
    rects2 = ax.bar(x_indices + width/2, bests, width, label='Best Fitness', color='#2ecc71')
    
    ax.set_yscale('log')
    ax.set_ylabel('Fitness Value (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('ESA 7 Functions Optimization Performance (Normalized RBF)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(functions, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir,'performance_comparison.png'), dpi=300)
    plt.close()