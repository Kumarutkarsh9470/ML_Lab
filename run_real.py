"""Run the extensions and the E2E experiments on the China A-share panel, one stage after another.

    python run_real.py ext C A D B E      # stages, in the order given

Stages (notebook sections): ext = 01c (reduced networks, GMM / size experts, SVR); A, B, C, D, E = notebook 03.
A2, B2, C2, E2 are a reduced plan (about 6 h): E2E for Linear and NN3, costs of 20 bps, decision-aware PCA with
mu in {0.1, 1}, and mean-SD E2E at eta = 1 before costs.
Every runner skips model-window pairs already saved, so an interrupted chain resumes where it stopped.
Run one chain at a time: two processes holding a window each run out of memory.
"""
import sys
import time
from pathlib import Path

from mlm.data import rolling_windows
from mlm.e2e import E2EConfig, run_e2e
from mlm.mixture import run_mixture
from mlm.models import MODELS
from mlm.reduce import run_reduced
from mlm.train import TrainConfig, run_rolling

PANEL_DIR = Path("D:/ML_Markowitz_data/panel_cn")
PRED_DIR = Path("D:/ML_Markowitz_data/preds/cn")
windows = rolling_windows("2014-01-01", "2023-12-31")
ZETA_GRID = (0.05, 0.2, 1.0)  # compared on validation in the four 2014 windows, then fixed per model
ETAS = (0.1, 0.5, 1.0, 2.5, 10.0)  # five of the paper's nine risk aversions, spanning the same range


def e2e(models, cfg, tag, wins=windows):
    run_e2e(models, PANEL_DIR, wins, PRED_DIR, PRED_DIR, cfg, tag=tag)


def ext():
    run_reduced(["PCA20-Linear", "PCA20-NN3", "PLS5-NN3", "LDA9-NN3", "SFA20-NN3"], PANEL_DIR, windows, PRED_DIR, TrainConfig())
    run_mixture(PANEL_DIR, windows, PRED_DIR)
    svr = TrainConfig(loss="eps", eps=0.5, weight_decay=1e-4)
    run_rolling(["Linear", "RFF"], PANEL_DIR, windows, PRED_DIR, svr, tag="-SVR", log_name="svr_log.csv")


def A():
    e2e(MODELS, E2EConfig(zeta_rel=ZETA_GRID, zeta_tune_windows=4), "E2E")


def B():
    for bps in (10, 20, 30, 40):
        e2e(["NN3"], E2EConfig(gamma=bps / 1e4), f"E2E-tc{bps}")


def C():
    e2e(["Ridge", "PCA20-NN3", "GMM-Experts"], E2EConfig(zeta_rel=ZETA_GRID, zeta_tune_windows=4), "E2E")
    for mu in (0.1, 1.0, 10.0):
        e2e(["PCA20-Linear", "PCA20-NN3"], E2EConfig(da_mu=mu), f"DA{mu:g}")


def C2():  # the reduced plan: decision-aware PCA with mu = 0.1 and 1 only
    e2e(["Ridge", "PCA20-NN3", "GMM-Experts"], E2EConfig(zeta_rel=ZETA_GRID, zeta_tune_windows=4), "E2E")
    for mu in (0.1, 1.0):
        e2e(["PCA20-Linear", "PCA20-NN3"], E2EConfig(da_mu=mu), f"DA{mu:g}")


def A2():  # the reduced plan: the paper's two headline models
    e2e(["Linear", "NN3"], E2EConfig(zeta_rel=ZETA_GRID, zeta_tune_windows=4), "E2E")


def B2():  # the reduced plan: one cost level
    e2e(["NN3"], E2EConfig(gamma=0.002), "E2E-tc20")


def E2():  # the reduced plan: mean-SD E2E at eta = 1, before costs (PPP left out for time; stage E runs it)
    e2e(["NN3"], E2EConfig(objective="meansd", eta=1.0), "MSD-gross-eta1")


def D():
    e2e(["NN3"], E2EConfig(warm_start=False), "E2E-scratch", windows[:8])


def E():
    for gamma, g in ((0.0, "gross"), (0.002, "net")):
        for eta in ETAS:
            lr = 1e-4 if eta < 1 else 1e-3  # as in Appendix A.2.2
            e2e(["NN3"], E2EConfig(objective="meansd", eta=eta, gamma=gamma, lr=lr), f"MSD-{g}-eta{eta:g}")
            e2e(["NN3"], E2EConfig(objective="ppp", eta=eta, gamma=gamma, lr=lr), f"PPP-{g}-eta{eta:g}")


if __name__ == "__main__":
    if sys.platform == "win32":  # keep the machine awake while this runs: CUDA does not survive sleep
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    stages = {"ext": ext, "A": A, "B": B, "C": C, "D": D, "E": E, "A2": A2, "B2": B2, "C2": C2, "E2": E2}
    for s in sys.argv[1:]:
        t = time.time()
        print(f"=== stage {s} started {time.strftime('%Y-%m-%d %H:%M')}", flush=True)
        stages[s]()
        print(f"=== stage {s} done in {(time.time() - t) / 3600:.2f} h", flush=True)
