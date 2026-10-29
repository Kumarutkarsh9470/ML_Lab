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
    lr: float = 1e-3
    batch_size: int = 10_000
    max_epochs: int = 100
    patience: int = 10
    target_scale: float = 100.0
    winsor: tuple | None = (0.005, 0.995)
    embargo: int = 2
    data_on_device: bool = True
    loss: str = "mse"
    eps: float = 0.5
    weight_decay: float = 0.0
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


def loss_fn(cfg):
    if cfg.loss == "mse":
        return F.mse_loss
    if cfg.loss == "eps":
        return lambda p, y: torch.clamp((p - y).abs() - cfg.eps, min=0).mean()
    if cfg.loss == "huber":
        return lambda p, y: F.huber_loss(p, y, delta=cfg.eps)
    raise ValueError(cfg.loss)


def fit_mse(model, Xtr, ytr, Xva, yva, cfg):
    """Fit by the configured statistical loss (MSE by default); early stopping on validation MSE."""
    model.to(cfg.device)
    crit = loss_fn(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    best, best_epoch, best_state, bad = np.inf, 0, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr), generator=gen).to(Xtr.device)
        for s in range(0, len(perm), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            if len(idx) < 2:
                continue
            xb, yb = _dev(Xtr[idx], cfg.device), ytr[idx].to(cfg.device)
            loss = crit(model(xb).squeeze(-1), yb)
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
    """Train every model on every window and save out-of-sample predictions."""
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
            if torch.cuda.mem_get_info()[0] > need + 1.5e9:
                Xtr, ytr, Xva, Xte = (t.to(cfg.device) for t in (Xtr, ytr, Xva, Xte))
        n_tr, n_va = len(tr), len(va)
        te = te[["date", "permno", "me", "target"]].reset_index(drop=True)
        va_meta = va[["date", "permno", "me", "target"]].reset_index(drop=True)
        del df, tr, va
        for m in todo:
            t0 = time.time()
            torch.manual_seed(cfg.seed)
            model, info = fit_mse(make_model(m, len(features)), Xtr, ytr, Xva, yva, cfg)
            res = te.assign(pred=predict(model, Xte, cfg.device) / cfg.target_scale)
            d = out_dir / f"{m}{tag}"
            d.mkdir(parents=True, exist_ok=True)
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
    """Out-of-sample (split="test") or validation (split="val") predictions of one model, all windows."""
    files = sorted((Path(out_dir) / model).glob("*.parquet"))
    files = [f for f in files if f.stem.endswith("_val") == (split == "val") and not f.stem.endswith("_daily")]
    parts = [pd.read_parquet(f).assign(window=f.stem.removesuffix("_val")) for f in files]
    return pd.concat(parts, ignore_index=True).sort_values(["date", "permno"])
