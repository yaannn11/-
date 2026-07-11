# data_manager.py
import os
import numpy as np
import config
import pandas as pd

class ESADataManager:
    def __init__(self):
        # 確保所有輸出目錄都存在
        os.makedirs(config.CHARTS_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    def save_text_report(self, results_dict, summary_stats):
        """將統計學結果輸出為結構化的文字報告與 Markdown 表格"""
        print(f"Saving text report to: {config.OUTPUT_TXT_PATH}")
        with open(config.OUTPUT_TXT_PATH, 'w', encoding='utf-8') as out_f:
            out_f.write("========================================================\n")
            out_f.write("ESA 7 Functions Run Report\n")
            out_f.write("========================================================\n\n")
            
            for f in config.functions:
                out_f.write(f"--- Function: {f} ---\n")
                seed_vals = []
                for s in config.seeds:
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
            for f in config.functions:
                stats = summary_stats[f]
                out_f.write(f"| {f} | {stats['best']:.6e} | {stats['worst']:.6e} | {stats['mean']:.6e} | {stats['std']:.6e} |\n")
            out_f.write("========================================================\n")

    def save_analysis_csv(self, results_dict):
        """將詳細的評估紀錄與安全防禦歷史儲存為 CSV 檔案"""
        
        # 1. 儲存預測誤差與歷史軌跡
       
        with open(config.ERRORS_CSV_PATH, 'w', encoding='utf-8') as f_rmse:
            f_rmse.write("Function,Seed,NFE,Action,State,Reward,Predicted_Y,Actual_Y,Error,Relative_Error,Min_Dist_DB\n")
            for f in config.functions:
                for s in config.seeds:
                    records = results_dict[f][s]['eval_records']
                    for r in records:
                        f_rmse.write(f"{f},{s},{r['nfe']},{r['action']},{r['state']},{r['reward']:.1f},"
                                    f"{r['predicted_y']:.6e},{r['actual_y']:.6e},{r['error']:.6e},"
                                    f"{r['relative_error']:.6e},{r['min_dist_db']:.6e}\n")
        print(f"Prediction errors file saved to {config.ERRORS_CSV_PATH}")
        
        
        # 6.2 Save Q-tables
        with open(config.Q_tables_path, 'w', encoding='utf-8') as f_q:
            f_q.write("Function,Seed,State,Q_Action_0,Q_Action_1,Q_Action_2,Q_Action_3\n")
            for f in config.functions:
                for s in config.seeds:
                    q_table = results_dict[f][s]['q_table']
                    for state_idx, q_vals in enumerate(q_table):
                        f_q.write(f"{f},{s},{state_idx},{q_vals[0]:.6e},{q_vals[1]:.6e},{q_vals[2]:.6e},{q_vals[3]:.6e}\n")
        print(f"Q-tables file saved to {config.Q_tables_path}")
        
        # 6.4 Save Strategy Usage Counts
        with open(config.usage_path, 'w', encoding='utf-8') as f_use:
            f_use.write("Function,Seed,Action_0_Count,Action_1_Count,Action_2_Count,Action_3_Count,Action_0_Ratio,Action_1_Ratio,Action_2_Ratio,Action_3_Ratio\n")
            for f in config.functions:
                for s in config.seeds:
                    counts = results_dict[f][s]['action_counts']
                    total = sum(counts.values())
                    if total > 0:
                        ratios = {act: counts[act]/total for act in range(4)}
                    else:
                        ratios = {act: 0.0 for act in range(4)}
                    f_use.write(f"{f},{s},"
                                f"{counts[0]},{counts[1]},{counts[2]},{counts[3]},"
                                f"{ratios[0]:.4f},{ratios[1]:.4f},{ratios[2]:.4f},{ratios[3]:.4f}\n")
        print(f"Strategy usage counts file saved to {config.usage_path}")
        
        # =====================================================================
        # 💡 6.5 儲存安全防禦歷史紀錄 (重試次數、安全係數、預測誤差)
        # =====================================================================
        
        with open(config.SAFETY_CSV_PATH, 'w', encoding='utf-8') as f_safety:
            f_safety.write("Function,Seed,NFE,Retry_Count,Safety_Multiplier,Lower_Buffer,Upper_Buffer,First_Improvement,Final_Lower_Buffer,Final_Upper_Buffer,Final_Improvement,All_Predicted_Improvements\n")
            for f in config.functions:
                for s in config.seeds:
                    retries = results_dict[f][s]['history_retries']
                    multipliers = results_dict[f][s]['history_multipliers']
                    round_nfe = results_dict[f][s]['history_round_nfe'] 
                    errors = results_dict[f][s]['history_errors']# 這裡面現在存的是串好的字串
                    first_imp= results_dict[f][s]['history_first_improvement']
                    final_imp= results_dict[f][s]['history_final_improvement']
                    history_buffers= results_dict[f][s]['history_buffers']
                    history_upper_buffers= results_dict[f][s]['history_upper_buffers']
                    history_final_buffers= results_dict[f][s]['history_final_buffers']
                    history_final_upper_buffers= results_dict[f][s]['history_final_upper_buffers']
                    for idx in range(len(retries)):
                        f_safety.write(f"{f},{s},{round_nfe[idx]+1},{retries[idx]},{multipliers[idx]},"
                                    f"{history_buffers[idx]},{history_upper_buffers[idx]},{first_imp[idx]},"
                                    f"{history_final_buffers[idx]},{history_final_upper_buffers[idx]},{final_imp[idx]},{errors[idx]}\n")
    
    def generate_action_analysis(self):
        print(f"Running automated data analysis on: {config.ERRORS_CSV_PATH}")
        try:
            df = pd.read_csv(config.ERRORS_CSV_PATH)
            target_columns = ['Reward', 'Error', 'Relative_Error', 'Min_Dist_DB']
            
            target_columns = [col for col in target_columns if col in df.columns]
            
            summary_df = df.groupby(['Function', 'Action'])[target_columns].agg(['mean', 'std']).reset_index()
            
            summary_df.columns = [
                f"{col[0]}_{col[1].capitalize()}" if col[1] else col[0] 
                for col in summary_df.columns
            ]
            

            summary_df.to_csv(config.ACTION_SUMMARY_CSV_PATH, index=False)
            print(f"Action performance summary successfully saved to: {config.ACTION_SUMMARY_CSV_PATH}")
            
        except FileNotFoundError:
            print(f"[Error] Cannot find {config.ERRORS_CSV_PATH}. Automated analysis skipped.")
        