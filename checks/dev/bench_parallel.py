"""Does running several training processes at once raise total throughput on this GPU?"""
import subprocess
import sys
import time
from pathlib import Path

WORKER = r'''
import time, torch, torch.nn.functional as F, sys
sys.path.insert(0, r"{root}")
from mlm.models import make_model
dev = "cuda"; n, p = 1_000_000, 251
X = torch.randint(-127, 128, (n, p), dtype=torch.int8, device=dev); y = torch.randn(n, device=dev)
m = make_model("NN3", p).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
def ep():
    perm = torch.randperm(n).to(dev)
    for s in range(0, n, 10_000):
        i = perm[s:s + 10_000]; loss = F.mse_loss(m(X[i].float().mul_(1/127)).squeeze(-1), y[i])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
ep(); torch.cuda.synchronize(); t = time.time()
for _ in range(5): ep()
torch.cuda.synchronize(); print(f"{{(time.time() - t) / 5:.3f}}")
'''
root = str(Path(__file__).resolve().parents[2])
code = WORKER.format(root=root)
for k in (1, 2, 3):
    t = time.time()
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) for _ in range(k)]
    per = [float(p.communicate()[0].strip()) for p in procs]
    wall = time.time() - t
    print(f"{k} process(es): {sum(per) / k:.2f}s per epoch each -> throughput {k / (sum(per) / k):.2f} epochs/s")
