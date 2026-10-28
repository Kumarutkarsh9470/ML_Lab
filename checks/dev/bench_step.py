"""Where does an MSE training epoch spend its time? Synthetic data of the real size, on the GPU."""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlm.models import make_model  # noqa: E402

dev = "cuda"
n, p = 1_500_000, 251
X = torch.randint(-127, 128, (n, p), dtype=torch.int8, device=dev)
y = torch.randn(n, device=dev)


def epoch(model, opt, bs, steps=None):
    perm = torch.randperm(n).to(dev)
    torch.cuda.synchronize()
    t = time.time()
    k = 0
    for s in range(0, n, bs):
        idx = perm[s:s + bs]
        xb = X[idx].float().mul_(1 / 127)
        loss = F.mse_loss(model(xb).squeeze(-1), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        k += 1
        if steps and k >= steps:
            break
    torch.cuda.synchronize()
    return time.time() - t, k


for name in ["Linear", "NN3", "NN5"]:
    for label, kw, bs in [("Adam, batch 10k", {}, 10_000), ("Adam fused, batch 10k", {"fused": True}, 10_000),
                          ("Adam fused, batch 50k", {"fused": True}, 50_000)]:
        torch.manual_seed(0)
        m = make_model(name, p).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, **kw)
        epoch(m, opt, bs, steps=5)  # warm-up
        secs, k = epoch(m, opt, bs)
        print(f"{name:6s} {label:24s} {secs:5.2f}s per epoch, {1000 * secs / k:5.2f} ms per step ({k} steps)")
