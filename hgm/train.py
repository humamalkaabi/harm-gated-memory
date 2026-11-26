from hgm import paths as HP
import argparse, os, time
import numpy as np, torch
from hgm.data import load
from hgm.models.forecaster import BenchMem, views
import copy

ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--H", type=int); ap.add_argument("--seed", type=int)
ap.add_argument("--kind", default="mem"); ap.add_argument("--mem", default="linear")
ap.add_argument("--L", type=int, default=336); ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--B", type=int, default=64); ap.add_argument("--assoc", default="auto"); ap.add_argument("--read_drop", type=float, default=0.0); ap.add_argument("--tag", default=""); ap.add_argument("--head", default="mlp"); ap.add_argument("--steps", type=int, default=192); ap.add_argument("--rex", type=float, default=0.0)
a = ap.parse_args()
torch.set_num_threads(1)
name = f"{a.data}_H{a.H}_{a.kind}{a.tag}_{a.mem}_s{a.seed}"
os.makedirs(HP.CKPT, exist_ok=True)
if os.path.exists(f"{HP.CKPT}/{name}.pt"):
    print("exists", name); raise SystemExit
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
z, tr, va, end = load(a.data)
pool = torch.tensor(z[:tr].T)
N, T = pool.shape
m = BenchMem(L=a.L, H=a.H, mem=a.mem, assoc=a.assoc, head=a.head)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
g = torch.full((a.B,), 1.0 if a.kind == "mem" else 0.0)
ch = rng.integers(N, size=a.B); pos = rng.integers(a.L, T - a.H - a.steps, size=a.B)
M, S = m.init_state(a.B)
t0 = time.time(); chunk = 64
zval = torch.tensor(z[:va].T)


def val_mse(model):
    with torch.no_grad():
        Mv, Sv = model.init_state(N); gv = torch.full((N,), 1.0 if a.kind == "mem" else 0.0); err = []
        for t in range(tr, va - a.H + 1):
            tok, xn, mu, sd = views(zval, t, a.L, model.P)
            yn, Mv, Sv, _ = model.step(tok, xn, None, Mv, Sv, gv)
            if t >= tr + a.H: err.append((((yn * sd + mu) - zval[:, t:t + a.H]) ** 2).mean().item())
    return float(np.mean(err))


best = (float("inf"), -1, None)
for it in range(a.iters):
    over = pos + a.steps + a.H > T
    if over.any():
        M0, S0 = m.init_state(a.B); ov = torch.tensor(over)
        M = [torch.where(ov[:, None, None], x0, x) for x, x0 in zip(M, M0)]
        S = [torch.where(ov[:, None, None], x0, x) for x, x0 in zip(S, S0)]
        ch = np.where(over, rng.integers(N, size=a.B), ch); pos = np.where(over, a.L + rng.integers(0, a.steps, size=a.B), pos)
    zt = torch.stack([pool[c, p - a.L:p + a.steps + a.H] for c, p in zip(ch, pos)])
    tot = 0.0; nc = 0
    for c0 in range(a.L, a.L + a.steps, chunk):
        loss = 0.0; lane = 0.0
        rho = (torch.rand(a.B) >= a.read_drop).float() if a.read_drop > 0 else None
        for t in range(c0, min(c0 + chunk, a.L + a.steps)):
            tok, xn, mu, sd = views(zt, t, a.L, m.P)
            yn, M, S, _ = m.step(tok, xn, None, M, S, g, rho=rho)
            lane = lane + (((yn * sd + mu) - zt[:, t:t + a.H]) ** 2).mean(1)
        lane = lane / chunk; loss = lane.mean()
        if a.rex > 0: loss = loss + a.rex * lane.var()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        M, S = [x.detach() for x in M], [x.detach() for x in S]; tot += loss.item(); nc += 1
    pos = pos + a.steps
    sched.step()
    if (it + 1) % 20 == 0 or it == a.iters - 1:
        v = val_mse(m)
        if v < best[0]: best = (v, it, copy.deepcopy(m.state_dict()))
        print(f"[{name}] it {it} loss {tot / nc:.4f} val {v:.4f} best {best[0]:.4f}@{best[1]} ({time.time() - t0:.0f}s)", flush=True)
torch.save(best[2], f"{HP.CKPT}/{name}.pt")
