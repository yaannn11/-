# config.py
import os

# 演算法與問題維度設定
dim = 30
max_nfe = 200

# 實驗範疇設定
seeds = [10, 20, 30, 42, 50]
functions = ['Ellipsoid', 'Rosenbrock', 'Ackley', 'Griewank', 'SRR', 'RHC1', 'RHC2']

#  路徑設定
base_dir = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(base_dir, 'charts')
RESULTS_DIR = os.path.join(base_dir, 'results')
OUTPUT_TXT_PATH = os.path.join(base_dir, 'results\esa_experiments_output.txt')
ERRORS_CSV_PATH = os.path.join(base_dir, 'results\prediction_errors.csv') if 'base_dir' in locals() else os.path.join(base_dir, 'results/prediction_errors.csv')
SAFETY_CSV_PATH = os.path.join(base_dir, 'results\safety_defense_history.csv')
ACTION_SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, 'action_performance_summary.csv')
Q_tables_path = os.path.join(base_dir, 'results\Q-tables.csv')
usage_path=os.path.join(base_dir, 'results\strategy_usage.csv')


MAX_WORKERS = min(len(seeds) * len(functions), os.cpu_count() or 4)
