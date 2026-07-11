# experiment_utils.py

import multiprocessing
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark_functions import get_benchmark_function
from esa_agent import EvolutionarySamplingAgent


# =============================================================================
# 1. Output Directory Settings
# =============================================================================

# 專案根目錄，也就是 experiment_utils.py 所在的位置
PROJECT_DIR = Path(__file__).resolve().parent

# 實驗結果輸出資料夾
RESULTS_DIR = PROJECT_DIR / "results"

# 圖片輸出資料夾
CHARTS_DIR = PROJECT_DIR / "charts"


def create_output_directories():
    """
    建立輸出資料夾。

    results/
        儲存 TXT 與 CSV 實驗結果。

    charts/
        儲存收斂曲線與效能比較圖。
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. Single Experiment Task
# =============================================================================

def run_experiment_task(args):
    """
    執行單一 benchmark function 與單一 random seed 的 ESA 實驗。

    Parameters
    ----------
    args : tuple
        格式為：
        (
            function_name,
            seed,
            dimension,
            maximum_nfe
        )

    Returns
    -------
    dict
        回傳單次實驗的最佳值、收斂紀錄、RBF 紀錄、
        預測誤差、Q-table 與各策略使用次數。
    """
    func_name, seed, dim, max_nfe = args

    # 固定隨機種子，確保實驗可以重現
    np.random.seed(seed)

    # 取得 benchmark function
    benchmark_function = get_benchmark_function(
        name=func_name,
        dim=dim,
    )

    # 建立 ESA
    esa = EvolutionarySamplingAgent(
        obj_func=benchmark_function,
        dim=dim,
        lb=benchmark_function.lb,
        ub=benchmark_function.ub,
        max_nfe=max_nfe,
    )

    # 執行 ESA
    (
        best_x,best_y,history_nfe,history_fitness,rbf_records,eval_records,q_table,action_counts,
    ) = esa.run()

    # multiprocessing 回傳的資料盡量轉為一般 Python 型態
    if hasattr(q_table, "tolist"):
        q_table = q_table.tolist()

    return {
        "func_name": func_name,
        "seed": seed,
        "best_x": (
            best_x.tolist()
            if hasattr(best_x, "tolist")
            else best_x
        ),
        "best_y": float(best_y),
        "history_nfe": history_nfe,
        "history_fitness": history_fitness,
        "rbf_records": rbf_records,
        "eval_records": eval_records,
        "q_table": q_table,
        "action_counts": action_counts,
    }


# =============================================================================
# 3. Parallel Experiment Runner
# =============================================================================

def run_parallel_experiments(
    functions,
    seeds,
    dim,
    max_nfe,
):
    """
    平行執行所有 benchmark functions 與 random seeds。

    Parameters
    ----------
    functions : list[str]
        Benchmark function 名稱。

    seeds : list[int]
        實驗使用的 random seeds。

    dim : int
        問題維度。

    max_nfe : int
        最大真實函數評估次數。

    Returns
    -------
    dict
        格式如下：

        results_dict[function_name][seed] = {
            "best_y": ...,
            "history_nfe": ...,
            "history_fitness": ...,
            ...
        }
    """
    create_output_directories()

    print("=====================================================================")
    print("Evolutionary Sampling Agent Parallel Experiment")
    print("=====================================================================")
    print(f"Dimension : {dim}D")
    print(f"NFE Budget: {max_nfe}")
    print(f"Seeds     : {seeds}")
    print(f"Functions : {functions}")
    print("Running experiments in parallel...")

    # 建立所有實驗組合
    task_args = []

    for func_name in functions:
        for seed in seeds:
            task_args.append(
                (
                    func_name,
                    seed,
                    dim,
                    max_nfe,
                )
            )

    # 不超過任務數量及電腦 CPU 核心數
    cpu_count = os.cpu_count() or 4
    process_count = min(
        len(task_args),
        cpu_count,
    )

    print(f"Number of tasks    : {len(task_args)}")
    print(f"Number of processes: {process_count}")

    # 平行執行
    with multiprocessing.Pool(
        processes=process_count
    ) as pool:
        results_list = pool.map(
            run_experiment_task,
            task_args,
        )

    # 將 list 整理成 function → seed 的巢狀 dictionary
    results_dict = {
        func_name: {}
        for func_name in functions
    }

    for result in results_list:
        func_name = result["func_name"]
        seed = result["seed"]

        results_dict[func_name][seed] = {
            "best_x": result["best_x"],
            "best_y": result["best_y"],
            "history_nfe": result["history_nfe"],
            "history_fitness": result["history_fitness"],
            "rbf_records": result["rbf_records"],
            "eval_records": result["eval_records"],
            "q_table": result["q_table"],
            "action_counts": result["action_counts"],
        }

    print("All parallel experiments completed.")

    return results_dict


# =============================================================================
# 4. Summary Statistics
# =============================================================================

def calculate_summary_stats(
    results,
    functions,
    seeds,
):
    """
    計算每個 benchmark function 的統計結果。

    包含：
    - Mean
    - Standard deviation
    - Best
    - Worst
    """
    summary_stats = {}

    for func_name in functions:
        seed_values = []

        for seed in seeds:
            best_y = results[func_name][seed]["best_y"]
            seed_values.append(best_y)

        seed_values = np.asarray(
            seed_values,
            dtype=float,
        )

        summary_stats[func_name] = {
            "mean": float(np.mean(seed_values)),
            "std": float(np.std(seed_values)),
            "best": float(np.min(seed_values)),
            "worst": float(np.max(seed_values)),
        }

    return summary_stats


# =============================================================================
# 5. Save Text Summary Report
# =============================================================================

def save_summary_report(
    results,
    functions,
    seeds,
):
    """
    儲存 ESA 實驗統計報告。

    輸出：
        results/esa_experiments_output.txt
    """
    create_output_directories()

    output_path = (
        RESULTS_DIR
        / "esa_experiments_output.txt"
    )

    summary_stats = calculate_summary_stats(
        results=results,
        functions=functions,
        seeds=seeds,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:

        output_file.write(
            "========================================================\n"
        )
        output_file.write(
            "ESA 7 Functions Run Report\n"
        )
        output_file.write(
            "Normalized Cubic RBF and 30-Dimensional Experiments\n"
        )
        output_file.write(
            "========================================================\n\n"
        )

        for func_name in functions:
            output_file.write(
                f"--- Function: {func_name} ---\n"
            )

            for seed in seeds:
                best_y = results[
                    func_name
                ][seed]["best_y"]

                output_file.write(
                    f"  Seed {seed}: "
                    f"{best_y:.6e}\n"
                )

            stats = summary_stats[func_name]

            output_file.write(
                "  - Summary Statistics -\n"
            )
            output_file.write(
                f"    Mean   : "
                f"{stats['mean']:.6e}\n"
            )
            output_file.write(
                f"    Std Dev: "
                f"{stats['std']:.6e}\n"
            )
            output_file.write(
                f"    Best   : "
                f"{stats['best']:.6e}\n"
            )
            output_file.write(
                f"    Worst  : "
                f"{stats['worst']:.6e}\n\n"
            )

        # Markdown 格式摘要表格
        output_file.write(
            "========================================================\n"
        )
        output_file.write(
            "SUMMARY STATISTICS MATRIX\n"
        )
        output_file.write(
            "========================================================\n"
        )

        output_file.write(
            "| Function | Best | Worst | Mean | Std Dev |\n"
        )
        output_file.write(
            "| :--- | :---: | :---: | :---: | :---: |\n"
        )

        for func_name in functions:
            stats = summary_stats[func_name]

            output_file.write(
                f"| {func_name} "
                f"| {stats['best']:.6e} "
                f"| {stats['worst']:.6e} "
                f"| {stats['mean']:.6e} "
                f"| {stats['std']:.6e} |\n"
            )

        output_file.write(
            "========================================================\n"
        )

    print(
        f"Statistical report saved to:\n"
        f"{output_path}"
    )

    return summary_stats


# =============================================================================
# 6. Save Prediction Errors
# =============================================================================

def save_prediction_errors(
    results,
    functions,
    seeds,
):
    """
    儲存每次候選點的 RBF 預測值與真實值誤差。

    輸出：
        results/prediction_errors.csv
    """
    create_output_directories()

    output_path = (
        RESULTS_DIR
        / "prediction_errors.csv"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:

        output_file.write(
