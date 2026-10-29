"""Decision-level results: each E2E run against its own two-stage starting point, on the same test days.

E2E and two-stage models are compared as the equal-weight top decile of their forecasts (the paper's Table 2);
the cost and mean-SD experiments use the realized portfolios saved by mlm.e2e.run_e2e (<run>/<window>_daily.parquet).
Writes report/results_decision.json and report/tables/decision.tex.
Runs that have not finished all 40 windows are skipped (the from-scratch check uses its own 8 windows).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlm import metrics  # noqa: E402
from mlm.data import rolling_windows  # noqa: E402
from mlm.e2e import backtest_meansd, load_sigma  # noqa: E402
from mlm.metrics import nw_test, perf  # noqa: E402
from mlm.portfolio import top_portfolio, tuned_kappa  # noqa: E402
from mlm.train import load_predictions  # noqa: E402

PRED = Path("D:/ML_Markowitz_data/preds/cn")
UNRANKED = "D:/ML_Markowitz_data/panel_cn_unranked"
metrics.ANN = json.loads(Path("D:/ML_Markowitz_data/panel_cn/meta.json").read_text())["ann"]
START, END = "2014-01-01", "2023-12-31"
GAMMA = 0.002  # 20 bps
WINDOWS = rolling_windows(START, END)

PAIRS = [("E2E-Linear", "Linear"), ("E2E-NN3", "NN3"), ("E2E-Ridge", "Ridge"), ("E2E-GMM-Experts", "GMM-Experts"),
         ("E2E-PCA20-NN3", "PCA20-NN3"), ("DA0.1-PCA20-NN3", "PCA20-NN3"), ("DA1-PCA20-NN3", "PCA20-NN3"),
         ("DA0.1-PCA20-Linear", "PCA20-Linear"), ("DA1-PCA20-Linear", "PCA20-Linear")]


def daily(run, n_windows=40):
    files = sorted((PRED / run).glob("*_daily.parquet"))
    if len(files) < n_windows:
        return None
    return pd.concat([pd.read_parquet(f) for f in files]).set_index("date").sort_index()


def preds(model):
    p = load_predictions(PRED, model)
    return p[(p.date >= START) & (p.date <= END)]


def stats(r, to=None):
    s = perf(r)
    out = {"avg": s["Avg %"], "sd": s["SD %"], "sr": s["SR"]}
    if to is not None:
        out["to"] = 100 * float(to.mean())
    return out


def compare(a, b):
    """Newey-West test of the daily difference a - b, on the days both have."""
    d = (a - b).dropna()
    t = nw_test(d)
    return {"diff_avg": 100 * metrics.ANN * float(d.mean()), "t": t["t"], "p_one": t["p one-sided"], "p_two": t["p two-sided"]}


res = {"gamma": GAMMA, "pairs": {}, "costs": {}, "meansd": {}, "scratch": {}}
# E2E against its start, both as the equal-weight top decile of their own forecasts (the paper's Table 2)
top = {}
for run, base in PAIRS:
    if len(list((PRED / run).glob("20??Q?.pt"))) < 40:
        print(f"skip {run}: not finished")
        continue
    for m in (run, base):
        if m not in top:
            top[m] = top_portfolio(preds(m))
    e, b = top[run], top[base].loc[top[run].index]
    res["pairs"][run] = {"base": base, "e2e": stats(e["gross"], e["to_p"]), "two_stage": stats(b["gross"], b["to_p"]),
                         "test": compare(e["gross"], b["gross"])}
    r = res["pairs"][run]
    print(f"{run:20s} E2E {r['e2e']['avg']:5.1f}% SR {r['e2e']['sr']:.2f} | {base} {r['two_stage']['avg']:5.1f}% "
          f"SR {r['two_stage']['sr']:.2f} | t {r['test']['t']:+.2f}")

# Transaction costs (Table 3): NN3 trained with 20 bps costs, against MSE-Top, MSE-Opt and tuned MSE-Opt at 20 bps
tc = daily("E2E-tc20-NN3")
if tc is not None:
    nn3, nn3_val = preds("NN3"), load_predictions(PRED, "NN3", split="val")
    kappa, chosen = tuned_kappa(nn3, nn3_val, WINDOWS, GAMMA)
    bench = {"MSE-Top": top_portfolio(nn3, GAMMA), "MSE-Opt": top_portfolio(nn3, GAMMA, optimize=True),
             "MSE-Opt (tuned)": top_portfolio(nn3, GAMMA, optimize=True, kappa=kappa)}
    e_gross = daily("E2E-NN3")
    rows = {"E2E-tc20": stats(tc["net"], tc["to_p"])}
    if e_gross is not None:  # the gross-trained E2E model, charged the same costs
        rows["E2E (trained gross)"] = stats(e_gross["gross"] - GAMMA * e_gross["to_p"], e_gross["to_p"])
    for k, v in bench.items():
        rows[k] = stats(v["net"].loc[tc.index], v["to_p"].loc[tc.index])
    res["costs"] = {"rows": rows, "kappa_chosen": chosen.value_counts().to_dict(),
                    "tests": {k: compare(tc["net"], v["net"]) for k, v in bench.items()}}
    for k, v in rows.items():
        print(f"costs {k:20s} net {v['avg']:5.1f}% SR {v['sr']:.2f} TO {v['to']:.0f}%")

# Mean-SD (Section 4.4), eta = 1 before costs: E2E, PPP and MSE-Opt (eq. 21 solved with the MSE forecasts)
msd, ppp = daily("MSD-gross-eta1-NN3"), daily("PPP-gross-eta1-NN3")
if msd is not None:
    nn3 = preds("NN3")
    mse_opt = backtest_meansd(nn3, load_sigma(UNRANKED, nn3), eta=1.0)
    rows = {"E2E mean-SD": msd, "MSE-Opt": mse_opt.loc[msd.index]}
    if ppp is not None:
        rows["PPP"] = ppp
    res["meansd"] = {"eta": 1.0, "rows": {k: {**stats(v["gross"], v["to_p"]), "sd_exante": 100 * np.sqrt(metrics.ANN) * float(v["sd_exante"].mean())}
                                          for k, v in rows.items()},
                     "tests": {k: compare(msd["gross"], v["gross"]) for k, v in rows.items() if k != "E2E mean-SD"}}
    for k, v in res["meansd"]["rows"].items():
        print(f"mean-SD {k:12s} {v['avg']:5.1f}% SD {v['sd']:.1f}% SR {v['sr']:.2f}")

# Warm start (PAPER_ISSUES D1): NN3 trained end to end from scratch, on the first 8 windows
sc = daily("E2E-scratch-NN3", n_windows=8)
if sc is not None:
    warm = daily("E2E-NN3")
    rows = {"E2E from scratch": stats(sc["gross"])}
    if warm is not None:
        rows["E2E warm start"] = stats(warm["gross"].loc[sc.index])
    rows["NN3 two-stage"] = stats(top_portfolio(preds("NN3"))["gross"].loc[sc.index])
    res["scratch"] = {"days": len(sc), "rows": rows}
    for k, v in rows.items():
        print(f"scratch {k:18s} {v['avg']:5.1f}% SR {v['sr']:.2f}")

(ROOT / "report" / "results_decision.json").write_text(json.dumps(res, indent=1, default=float))

# LaTeX table
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r" & \multicolumn{2}{c}{E2E} & \multicolumn{2}{c}{Two-stage start} & \multicolumn{2}{c}{Difference} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"Model & Avg \% & SR & Avg \% & SR & pp & NW $t$ \\", r"\midrule"]
name = {"E2E-Linear": "Linear", "E2E-NN3": "NN3", "E2E-Ridge": "Ridge", "E2E-GMM-Experts": "GMM experts",
        "E2E-PCA20-NN3": "PCA20-NN3 (PCA fixed)", "DA0.1-PCA20-NN3": r"PCA20-NN3, decision-aware $\mu=0.1$",
        "DA1-PCA20-NN3": r"PCA20-NN3, decision-aware $\mu=1$", "DA0.1-PCA20-Linear": r"PCA20-Linear, decision-aware $\mu=0.1$",
        "DA1-PCA20-Linear": r"PCA20-Linear, decision-aware $\mu=1$"}
for run, r in res["pairs"].items():
    L.append(f"{name.get(run, run)} & {r['e2e']['avg']:.1f} & {r['e2e']['sr']:.2f} & {r['two_stage']['avg']:.1f} & "
             f"{r['two_stage']['sr']:.2f} & {r['test']['diff_avg']:+.1f} & {r['test']['t']:+.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "report" / "tables").mkdir(exist_ok=True)
(ROOT / "report" / "tables" / "decision.tex").write_text("\n".join(L) + "\n")
print("wrote report/results_decision.json and report/tables/decision.tex")
