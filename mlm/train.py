"""Two-stage baseline, stage one: fit the prediction model by minimizing MSE over rolling windows."""
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import Q, load_panel, panel_features, split_window
from .metrics import oos_r2
from .models import make_model


@dataclass
class TrainConfig:
    lr: float = 1e-3            # Adam learning rate (Appendix A.2.2)
    batch_size: int = 10_000    # not reported in the paper; 10,000 follows Gu et al. (2020)
    max_epochs: int = 100       # not reported in the paper
    patience: int = 10          # early stopping on validation MSE (Appendix A.2.2)
    target_scale: float = 100.0  # train on percent returns for better conditioning; predictions are rescaled back
    winsor: tuple | None = (0.005, 0.995)  # per-day clipping of training/validation labels only
    embargo: int = 2            # trading days dropped before each split boundary
    data_on_device: bool = True  # keep the window's int8 features on the GPU (about 1 GB)
    use_cuda_graph: bool = False  # optional CUDA-graph steps (GraphStep); off: barely faster on this GPU, and
    #                               dropout draws differ inside a graph, so runs would not match earlier ones
    loss: str = "mse"           # "mse" (the paper), "eps" (epsilon-insensitive, SVR) or "huber"
    eps: float = 0.5            # SVR tube half-width / Huber threshold, in percent daily return
    weight_decay: float = 0.0   # L2 penalty on the weights (the ||w||^2 term of an SVR)
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def _dev(x, device):
    return x.to(device, non_blocking=True).float().mul_(1.0 / Q)


def to_tensors(df, features, cfg, labels=True):
    """int8 features stay on the CPU and are converted batch by batch on the device, which keeps memory low."""
    X = torch.from_numpy(np.ascontiguousarray(df[features].to_numpy(np.int8)))
    if not labels:
        return X
    y = df["target"].astype(np.float64)
    if cfg.winsor:
        q = df.groupby("date")["target"].quantile(list(cfg.winsor)).unstack()
        y = y.clip(df["date"].map(q[cfg.winsor[0]]), df["date"].map(q[cfg.winsor[1]]))
    return X, torch.from_numpy((y * cfg.target_scale).to_numpy(np.float32))


@torch.no_grad()
def predict(model, X, device, batch=200_000):
    model.eval()
    out = [model(_dev(X[s:s + batch], device)).squeeze(-1).cpu() for s in range(0, len(X), batch)]
    return torch.cat(out).numpy()


class GraphStep:
    """One full training step (forward, backward, Adam) captured as a CUDA graph and replayed per batch.

    The networks are so small that a step is dominated by launching ~100 tiny kernels; replaying a captured
    graph launches them as one unit. Batches, updates and the protocol are unchanged. Capture needs a few
    warm-up steps on throwaway data, so all parameters, buffers and optimizer state are restored afterwards.
    """

    def __init__(self, model, opt, batch_size, n_features, device):
        self.x = torch.zeros(batch_size, n_features, device=device)
        self.y = torch.zeros(batch_size, device=device)
        keep = [t.detach().clone() for t in list(model.parameters()) + list(model.buffers())]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                F.mse_loss(model(self.x).squeeze(-1), self.y).backward()
                opt.step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            F.mse_loss(model(self.x).squeeze(-1), self.y).backward()
            opt.step()
        with torch.no_grad():  # undo the warm-up
            for t, k in zip(list(model.parameters()) + list(model.buffers()), keep):
                t.copy_(k)
            for st in opt.state.values():
                for v in st.values():
                    if torch.is_tensor(v):
                        v.zero_()

    def __call__(self, xb, yb):
        self.x.copy_(xb)
        self.y.copy_(yb)
        self.graph.replay()


def loss_fn(cfg):
    if cfg.loss == "mse":
        return F.mse_loss
    if cfg.loss == "eps":  # epsilon-insensitive: errors inside the tube cost nothing
        return lambda p, y: torch.clamp((p - y).abs() - cfg.eps, min=0).mean()
    if cfg.loss == "huber":
        return lambda p, y: F.huber_loss(p, y, delta=cfg.eps)
    raise ValueError(cfg.loss)


def fit_mse(model, Xtr, ytr, Xva, yva, cfg):
    """Fit by the configured statistical loss (MSE by default); early stopping on validation MSE."""
    model.to(cfg.device)
    use_graph = cfg.use_cuda_graph and cfg.device != "cpu" and Xtr.is_cuda and cfg.loss == "mse"
    crit = loss_fn(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, capturable=use_graph, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    best, best_epoch, best_state, bad = np.inf, 0, None, 0
    step = None
    if use_graph:
        model.train()
        try:
            step = GraphStep(model, opt, cfg.batch_size, Xtr.shape[1], cfg.device)
        except RuntimeError:  # capture not supported here: fall back to ordinary steps
            step = None
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr), generator=gen).to(Xtr.device)  # drawn on the CPU: same batches either way
        for s in range(0, len(perm), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            if len(idx) < 2:  # BatchNorm needs at least two rows
                continue
            xb, yb = _dev(Xtr[idx], cfg.device), ytr[idx].to(cfg.device)
            if step is not None and len(idx) == cfg.batch_size:
                step(xb, yb)
                continue
            loss = crit(model(xb).squeeze(-1), yb)  # (the last, shorter batch always runs here)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = float(np.mean((predict(model, Xva, cfg.device) - yva.numpy()) ** 2))
        if val < best:
            best, best_epoch, bad = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, dict(epochs=epoch, best_epoch=best_epoch, val_mse=best)


def run_rolling(models, panel_dir, windows, out_dir, cfg, features=None, verbose=True, log_name="train_log.csv",
                tag=""):
    """Train every model on every window and save out-of-sample predictions.

    Writes out_dir/<model><tag>/<window>.parquet (date, permno, me, target, pred), <window>_val.parquet and
    <window>.pt (weights, reused to warm-start the E2E model as in Appendix A.2.2). `tag` names variants of
    the same architecture, e.g. tag="-SVR" with cfg.loss="eps". Finished model-window pairs are skipped,
    so an interrupted run can be resumed by calling this again. Parallel workers need their own log_name.
    """
    out_dir = Path(out_dir)
    features = features or panel_features(panel_dir)
    log_path = out_dir / log_name
    for w in windows:
        todo = [m for m in models if not (out_dir / f"{m}{tag}" / f"{w['name']}.parquet").exists()]
        if not todo:
            continue
        df = load_panel(panel_dir, w["train_start"], w["test_end"], features)
        tr, va, te = split_window(df, w, cfg.embargo)
        Xtr, ytr = to_tensors(tr, features, cfg)
        Xva, yva = to_tensors(va, features, cfg)
        Xte = to_tensors(te, features, cfg, labels=False)
        if cfg.data_on_device and cfg.device != "cpu":
            need = Xtr.nbytes + Xva.nbytes + Xte.nbytes + ytr.nbytes
            if torch.cuda.mem_get_info()[0] > need + 1.5e9:  # leave room for the model and batches
                Xtr, ytr, Xva, Xte = (t.to(cfg.device) for t in (Xtr, ytr, Xva, Xte))
        n_tr, n_va = len(tr), len(va)
        te = te[["date", "permno", "me", "target"]].reset_index(drop=True)
        va_meta = va[["date", "permno", "me", "target"]].reset_index(drop=True)
        del df, tr, va  # the features now live in the tensors
        for m in todo:
            t0 = time.time()
            torch.manual_seed(cfg.seed)
            model, info = fit_mse(make_model(m, len(features)), Xtr, ytr, Xva, yva, cfg)
            res = te.assign(pred=predict(model, Xte, cfg.device) / cfg.target_scale)
            d = out_dir / f"{m}{tag}"
            d.mkdir(parents=True, exist_ok=True)
            # validation predictions, for choices made on validation (e.g. the tuned MSE-Opt benchmark)
            va_meta.assign(pred=predict(model, Xva, cfg.device) / cfg.target_scale).to_parquet(
                d / f"{w['name']}_val.parquet", index=False)
            res.to_parquet(d / f"{w['name']}.parquet", index=False)
            torch.save(model.state_dict(), d / f"{w['name']}.pt")
            row = dict(window=w["name"], model=f"{m}{tag}", n_train=n_tr, n_val=n_va, n_test=len(te), **info,
                       test_r2=oos_r2(res["pred"], res["target"]), seconds=round(time.time() - t0, 1))
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            if verbose:
                print(f"{w['name']} {m:6s} epochs={info['epochs']:3d} val_mse={info['val_mse']:.4f} "
                      f"test_R2={row['test_r2']:+.4%} ({row['seconds']}s)")
    return pd.read_csv(log_path) if log_path.exists() else None


def load_predictions(out_dir, model, split="test"):
    """Out-of-sample (split="test") or validation (split="val") predictions of one model, all windows.

    A `window` column records the source window: validation periods of consecutive windows overlap, so
    validation predictions must be used window by window.
    """
    files = sorted((Path(out_dir) / model).glob("*.parquet"))  # E2E runs also hold <window>_daily portfolios
    files = [f for f in files if f.stem.endswith("_val") == (split == "val") and not f.stem.endswith("_daily")]
    parts = [pd.read_parquet(f).assign(window=f.stem.removesuffix("_val")) for f in files]
    return pd.concat(parts, ignore_index=True).sort_values(["date", "permno"])
