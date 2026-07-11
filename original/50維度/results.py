# Write statistical report
    output_path = os.path.join(base_dir, 'esa_experiments_output.txt')
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("========================================================\n")
        out_f.write("ESA 7 Functions Run Report (Normalized RBF & Bounds Corrected)\n")
        out_f.write("========================================================\n\n")
        
        summary_stats = {}
        
        for f in functions:
            out_f.write(f"--- Function: {f} ---\n")
            seed_vals = []
            for s in seeds:
                val = results_dict[f][s]['best_y']
                seed_vals.append(val)
                out_f.write(f"  Seed {s}: {val:.6e}\n")
                
            seed_vals = np.array(seed_vals)
            mean_val = np.mean(seed_vals)
            std_val = np.std(seed_vals)
            best_val = np.min(seed_vals)
            worst_val = np.max(seed_vals)
            
            summary_stats[f] = {
                'mean': mean_val,
                'std': std_val,
                'best': best_val,
                'worst': worst_val
            }
            
            out_f.write("  - Summary Stats -\n")
            out_f.write(f"    Mean   : {mean_val:.6e}\n")
            out_f.write(f"    Std Dev: {std_val:.6e}\n")
            out_f.write(f"    Best   : {best_val:.6e}\n")
            out_f.write(f"    Worst  : {worst_val:.6e}\n\n")
            
        # Write final markdown table
        out_f.write("========================================================\n")
        out_f.write("SUMMARY STATISTICS MATRIX\n")
        out_f.write("========================================================\n")
        out_f.write("| Function | Best | Worst | Mean | Std Dev |\n")
        out_f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for f in functions:
            stats = summary_stats[f]
            out_f.write(f"| {f} | {stats['best']:.6e} | {stats['worst']:.6e} | {stats['mean']:.6e} | {stats['std']:.6e} |\n")
        out_f.write("========================================================\n")
        
    print(f"Statistical report saved to {output_path}")
    
    # 4. Generate Convergence Curves
    print("Generating convergence plots...")
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
        plt.savefig(os.path.join(base_dir, f"charts/convergence_{f.lower()}.png"), dpi=200)
        plt.close()
        
    # 5. Generate Bar Chart Performance Comparison
    print("Generating statistical summary bar chart...")
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
    plt.savefig(os.path.join(base_dir, 'charts/performance_comparison.png'), dpi=300)
    plt.close()
    
    # 6. Save Analysis CSV Files
    print("Generating analysis CSV files...")
    
    # 6.1 Save Prediction Errors (Trajectory Diagnostic Log)
    rmse_path = os.path.join(base_dir, 'prediction_errors.csv')
    with open(rmse_path, 'w', encoding='utf-8') as f_rmse:
        f_rmse.write("Function,Seed,NFE,Action,State,Reward,Predicted_Y,Actual_Y,Error,Relative_Error,Min_Dist_DB\n")
        for f in functions:
            for s in seeds:
                records = results_dict[f][s]['eval_records']
                for r in records:
                    f_rmse.write(f"{f},{s},{r['nfe']},{r['action']},{r['state']},{r['reward']:.1f},"
                                 f"{r['predicted_y']:.6e},{r['actual_y']:.6e},{r['error']:.6e},"
                                 f"{r['relative_error']:.6e},{r['min_dist_db']:.6e}\n")
    print(f"Prediction errors file saved to {rmse_path}")
    
    # 6.2 Save Condition Numbers
    cond_path = os.path.join(base_dir, 'condition_numbers.csv')
    with open(cond_path, 'w', encoding='utf-8') as f_cond:
        f_cond.write("Function,Seed,NFE,Action,NumPoints,CondPhi,CondPhiReg\n")
        for f in functions:
            for s in seeds:
                records = results_dict[f][s]['rbf_records']
                for r in records:
                    f_cond.write(f"{f},{s},{r['nfe']},{r['action']},{r['num_points']},{r['cond_phi']:.6e},{r['cond_phi_reg']:.6e}\n")
    print(f"Condition numbers file saved to {cond_path}")
    
    # 6.3 Save Q-tables
    q_path = os.path.join(base_dir, 'q_tables.csv')
    with open(q_path, 'w', encoding='utf-8') as f_q:
        f_q.write("Function,Seed,State,Q_Action_0,Q_Action_1,Q_Action_2,Q_Action_3\n")
        for f in functions:
            for s in seeds:
                q_table = results_dict[f][s]['q_table']
                for state_idx, q_vals in enumerate(q_table):
                    f_q.write(f"{f},{s},{state_idx},{q_vals[0]:.6e},{q_vals[1]:.6e},{q_vals[2]:.6e},{q_vals[3]:.6e}\n")
    print(f"Q-tables file saved to {q_path}")
    
    # 6.4 Save Strategy Usage Counts
    usage_path = os.path.join(base_dir, 'strategy_usage.csv')
    with open(usage_path, 'w', encoding='utf-8') as f_use:
        f_use.write("Function,Seed,Action_0_Count,Action_1_Count,Action_2_Count,Action_3_Count,Action_0_Ratio,Action_1_Ratio,Action_2_Ratio,Action_3_Ratio\n")
        for f in functions:
            for s in seeds:
                counts = results_dict[f][s]['action_counts']
                total = sum(counts.values())
                if total > 0:
                    ratios = {act: counts[act]/total for act in range(4)}
                else:
                    ratios = {act: 0.0 for act in range(4)}
                f_use.write(f"{f},{s},"
                            f"{counts[0]},{counts[1]},{counts[2]},{counts[3]},"
                            f"{ratios[0]:.4f},{ratios[1]:.4f},{ratios[2]:.4f},{ratios[3]:.4f}\n")
    print(f"Strategy usage counts file saved to {usage_path}")
