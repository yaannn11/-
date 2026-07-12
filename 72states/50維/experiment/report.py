import os
import numpy as np

def write_report(results_dict, functions, seeds, base_dir):
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
    return summary_stats
    