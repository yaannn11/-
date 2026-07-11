import multiprocessing
import time

from config import DIM, MAX_NFE, SEEDS, FUNCTIONS
from experiment_utils import (
    run_parallel_experiments,
    save_all_results,
    generate_all_charts,
)


def main():
    t_start = time.time()

    print("=====================================================================")
    print("Evolutionary Sampling Agent (ESA) Parallel 7 Run (Normalized RBF)")
    print("=====================================================================")
    print(
        f"Dimension: {DIM}D | "
        f"NFE Budget: {MAX_NFE} | "
        f"Seeds: {SEEDS}"
    )
    print("Running experiments in parallel...")

    # 1. 執行七個函數、五組種子的平行實驗
    results_dict = run_parallel_experiments(
        functions=FUNCTIONS,
        seeds=SEEDS,
        dim=DIM,
        max_nfe=MAX_NFE,
    )

    # 2. 輸出統計報告與 CSV
    save_all_results(
        results=results_dict,
        functions=FUNCTIONS,
        seeds=SEEDS,
    )

    # 3. 產生收斂曲線與效能比較圖
    generate_all_charts(
        results=results_dict,
        functions=FUNCTIONS,
        seeds=SEEDS,
    )

    # 4. 顯示總執行時間
    t_elapsed = time.time() - t_start

    print("=====================================================================")
    print(
        f"All experiments completed successfully "
        f"in {t_elapsed:.2f} seconds."
    )
    print("=====================================================================")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
