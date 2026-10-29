"""LaTeX table of the two-stage models from report/results_summary.json (written by make_figures.py).

E2E runs are compared with their starting models in tables/decision.tex (make_decision.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S = json.loads((ROOT / "results_summary.json").read_text())
M = S["models"]
paper = [m for m in ["Linear", "NN1", "NN2", "NN3", "NN4", "NN5"] if m in M]
ours = [m for m in ["OLS", "Ridge", "Ridge-rank", "PCR", "PLS", "LDA", "LDA-top"] if m in M]
ext = [m for m in ["PCA20-Linear", "PCA20-NN3", "PLS5-NN3", "LDA9-NN3", "SFA20-NN3", "GMM-Experts", "Size-Experts",
                   "Linear-SVR", "RFF-SVR"] if m in M]
label = {"Linear": "Linear (paper)", "OLS": "OLS (closed form)"}
label.update({m: f"{m[2:].split('-', 1)[1]}, DA $\\mu={m[2:].split('-')[0]}$" for m in M if m.startswith("DA")})


def row(m):
    r = M[m]
    return (f"{label.get(m, m)} & {r['IC']:.3f} ({r['IC_t']:.1f}) & {r['H_avg']:.1f} & {r['H_sr']:.2f} & "
            f"{r['HL_sr']:.2f} & {r['TO']:.0f} & {r['H_ex30_avg']:.1f} & {r['H_ex30_sr']:.2f} \\\\")


lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r" & Rank IC & \multicolumn{2}{c}{Top decile} & H--L & Turnover & \multicolumn{2}{c}{Excl.\ bottom 30\%} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){7-8}",
         r"Model & (NW $t$) & Avg \% & SR & SR & \% & Avg \% & SR \\", r"\midrule"]
for group, title in [(paper, "Two-stage MSE, paper's architectures"),
                     (ours, "Two-stage, closed-form linear family (ours)"),
                     (ext, "Two-stage, reduced networks, mixtures and SVR (ours)")]:
    if group:
        lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}} \\")
        lines += [row(m) for m in group]
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "tables").mkdir(exist_ok=True)
(ROOT / "tables" / "results.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
