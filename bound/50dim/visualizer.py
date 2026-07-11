# visualizer.py
import os
import numpy as np
import matplotlib.pyplot as plt
import config

class ESAVisualizer:
    def __init__(self, base_dir):
        self.charts_dir = os.path.join(base_dir, 'charts')
        os.makedirs(self.charts_dir, exist_ok=True)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False 

    def plot_convergence_curves(self, functions, seeds, results_dict):
        # 4. Generate Convergence Curves
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
            plt.savefig(os.path.join(config.base_dir, f"charts/convergence_{f.lower()}.png"), dpi=200)
            plt.close()

    def plot_performance_comparison(self, functions, summary_stats):
        # 5. Generate Bar Chart Performance Comparison
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
        plt.savefig(os.path.join(config.base_dir, 'charts/performance_comparison.png'), dpi=300)
        plt.close()

    def plot_safety_dual_bounds(self, functions, seeds, results_dict):

        print("Generating single-seed analysis plots (Each seed gets its own pure chart)...")
            
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False 

        charts_dir = os.path.join(config.base_dir, 'charts')
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

        for f in config.functions:
            if f not in results_dict or len(results_dict[f]) == 0:
                continue
                
            for s in seeds:
                if s not in results_dict[f]:
                    continue
                    
                round_nfe = results_dict[f][s]['history_round_nfe']
                
                first_imp_raw = np.array(results_dict[f][s]['history_first_improvement'])
                first_improvement_plot = -first_imp_raw
                
                pos_upper_ceiling_1 = np.array(results_dict[f][s]['history_upper_buffers'])
                neg_lower_floor_1 = -np.array(results_dict[f][s]['history_buffers'])
                
                final_imp_raw = np.array(results_dict[f][s]['history_final_improvement'])
                final_improvement_plot = -final_imp_raw
                
                pos_upper_ceiling_2 = np.array(results_dict[f][s]['history_final_upper_buffers'])
                neg_lower_floor_2 = -np.array(results_dict[f][s]['history_final_buffers'])
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
                
                ax1.plot(round_nfe, first_improvement_plot, color='#1f77b4', alpha=0.8, linewidth=1.5,
                        label='First Candidate Plot (-1 * (ymin - pred))')

                ax1.plot(round_nfe, pos_upper_ceiling_1, color='#ff7f0e', linestyle='-.', alpha=0.8, linewidth=1.2,
                        label='Pruning Upper Ceiling (+Multiplier * Std)')

                ax1.plot(round_nfe, neg_lower_floor_1, color='#d62728', linestyle='--', alpha=0.8, linewidth=1.2,
                        label='Safety Lower Floor (-Multiplier * Std)')
   
                ax1.fill_between(round_nfe, neg_lower_floor_1, pos_upper_ceiling_1, color='#1f77b4', alpha=0.08, label='Safety Buffer Zone')
                
                ax1.axhline(0, color='black', linestyle=':', alpha=0.5)
                ax1.set_ylabel('Inverted Scale (-Improvement)', fontsize=10)
                ax1.set_title(f'original', fontsize=11, fontweight='bold')
                ax1.grid(True, ls="--", alpha=0.3)
                ax1.legend(loc='best', fontsize=9)
                

                ax2.plot(round_nfe, final_improvement_plot, color='#2ca02c', alpha=0.8, linewidth=1.5,
                        label='Final Approved Plot (-1 * (ymin - pred))')

                ax2.plot(round_nfe, pos_upper_ceiling_2, color='#ff7f0e', linestyle='-.', alpha=0.8, linewidth=1.2,
                        label='Pruning Upper Ceiling')

                ax2.plot(round_nfe, neg_lower_floor_2, color='#d62728', linestyle='--', alpha=0.8, linewidth=1.2,
                        label='Safety Lower Floor')

                ax2.fill_between(round_nfe, neg_lower_floor_2, pos_upper_ceiling_2, color='#2ca02c', alpha=0.08, label='Safety Buffer Zone')
                
                ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
                ax2.set_xlabel('Current NFE ', fontsize=11)
                ax2.set_ylabel('Inverted Scale (-Improvement)', fontsize=10)
                ax2.set_title(f'final', fontsize=11, fontweight='bold')
                ax2.grid(True, ls="--", alpha=0.3)
                ax2.legend(loc='lower right', fontsize=9)

                plt.suptitle(f'Symmetric Dual-Bound Analysis - {f} (Seed {s})', fontsize=14, fontweight='bold', y=0.96)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                
                chart_filename = os.path.join(charts_dir, f'{f}_seed{s}_safety_dual_bound.png')
                plt.savefig(chart_filename, dpi=200, bbox_inches='tight')
                plt.close(fig)
        
