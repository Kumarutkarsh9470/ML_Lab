"""Train a subset of the MSE models over all 40 windows. Run two of these at once to use the GPU fully:

    python run_mse_worker.py A NN5 NN2 NN1
    python run_mse_worker.py B NN4 NN3 Linear

Finished model-window pairs are skipped, so this also resumes an interrupted notebook run.
"""
import sys
from pathlib import Path

from mlm.data import rolling_windows
from mlm.train import TrainConfig, run_rolling

name, models = sys.argv[1], sys.argv[2:]
run_rolling(models, Path("D:/ML_Markowitz_data/panel_cn"), rolling_windows("2014-01-01", "2023-12-31"),
            Path("D:/ML_Markowitz_data/preds/cn"), TrainConfig(), log_name=f"train_log_{name}.csv")
print(f"worker {name} done: {models}")
