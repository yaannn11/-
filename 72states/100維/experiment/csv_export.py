import os  
# 6. Save Analysis CSV Files
def export_all(results_dict, functions, seeds, base_dir):
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
    