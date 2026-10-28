"""Render the slide equations with LaTeX into transparent, tightly cropped PNGs (report/eq/*.png)."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "eq"
OUT.mkdir(exist_ok=True)
INK = (31, 42, 68)  # 1F2A44, the deck's text colour

EQ = {
    "mse": r"\min_{\bm\theta}\;\frac{1}{NT}\sum_{i,t}\big(r_{i,t+1}-g(\bm x_{i,t};\bm\theta)\big)^2",
    "lower": r"\bm w_t^\star=\arg\min_{\substack{\bm 0\le\bm w\le u_t\bm 1\\ \bm 1^\top\bm w=1}}\;-\bm w^\top\hat{\bm r}_{t+1}+\zeta\lVert\bm w\rVert_2^2+\tfrac{\gamma}{2}\lVert\bm w-\bm w_{t-1}^\star\rVert_1",
    "upper": r"\min_{\bm\theta}\;\frac1T\sum_t\Big[-\bm w_t^{\star\top}\bm r_{t+1}+\tfrac{\gamma}{2}\lVert\bm w_t^\star-\bm w_{t-1}^\star\rVert_1\Big],\quad \hat{\bm r}_{t+1}=g(\bm x_t;\bm\theta)",
    "layer": r"w_i^\star=\Pi_{[0,u]}\big(S_{p_i}(\hat r_i-\tau)\big),\qquad \textstyle\sum_i w_i^\star(\tau)=1",
    "jacobian": r"\frac{\partial\bm w^\star}{\partial\hat{\bm r}}=\frac{1}{2\zeta}\Big(\bm D_F-\frac{\bm 1_F\bm 1_F^\top}{|F|}\Big)",
    "pcr": r"\bm\beta_{\mathrm{PCR}}=\bm V_K\big(\bm V_K^\top\bm C\bm V_K\big)^{-1}\bm V_K^\top\bm c",
    "lda": r"\bm S_b\,\bm v=\lambda\,\big[(1-\alpha)\bm S_w+\alpha\tfrac{\mathrm{tr}\,\bm S_w}{p}\bm I\big]\bm v",
    "dapca": r"\min_{\bm\theta,\bm W}\;\mathcal L_{\mathrm{dec}}(\bm\theta,\bm W)+\mu\lVert\bm W-\bm V_K\rVert_F^2",
    "gmm": r"g(\bm x)=\sum_k\pi_k(\bm x)\,\bm x^\top\bm\beta_k",
}

DOC = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,bm,lmodern}
\usepackage[paperwidth=30cm,paperheight=8cm,margin=1cm]{geometry}
\pagestyle{empty}
\begin{document}
\[ %s \]
\end{document}
"""


def render(name, tex, dpi=450):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "e.tex").write_text(DOC % tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "e.tex"], cwd=tmp,
                   check=True, capture_output=True)
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", "e.pdf", "e"], cwd=tmp, check=True)
    g = np.asarray(Image.open(tmp / "e.png").convert("L")).astype(np.float32)
    alpha = (255 - g).clip(0, 255).astype(np.uint8)
    ys, xs = np.nonzero(alpha > 8)
    pad = 12
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
    rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    rgba[..., :3] = INK
    rgba[..., 3] = alpha[y0:y1, x0:x1]
    Image.fromarray(rgba).save(OUT / f"{name}.png")
    shutil.rmtree(tmp)
    return rgba.shape[1] / dpi, rgba.shape[0] / dpi  # inches at the rendering size


sizes = {k: render(k, v) for k, v in EQ.items()}
for k, (w, h) in sizes.items():
    print(f"{k}: {w:.2f} x {h:.2f} in at 12pt")
(OUT / "sizes.txt").write_text("\n".join(f"{k} {w:.4f} {h:.4f}" for k, (w, h) in sizes.items()))
