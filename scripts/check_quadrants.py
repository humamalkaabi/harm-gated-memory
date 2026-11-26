import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
import matplotlib.pyplot as plt
from hgm.data import load

DATA, SEED, H = "ETTh2", 1, 24
d = 24
W = 7 * d
D = d

z, tr, va, end = load(DATA)
A = np.load(f"outputs/policies/{DATA}_H{H}_s{SEED}.npz")["A"]
n = A.shape[0]; nv = va - tr
target = np.stack([z[tr + o: tr + o + H].T for o in range(n)])
L = ((A - target) ** 2).mean(axis=2)

lval = L[H:nv - H].mean(axis=0)
ts = np.arange(nv + W + H, n - D - W - H, d)
R, dX = [], []
for t in ts:
    a = t + D
    R.append(np.log(L[a:a + W].mean(axis=0) / lval))
    past = z[tr + t - W: tr + t]; future = z[tr + a: tr + a + W]
    dX.append(np.abs(future.mean(0) - past.mean(0)) / np.maximum(past.std(0), 0.1))
R, dX = np.array(R), np.array(dX)

x, y = dX.ravel(), R.ravel()
xt, yt = np.quantile(x, 0.75), np.quantile(y, 0.90)
big_change, harm = x >= xt, y >= yt
share = {"Stable": (~big_change & ~harm).mean(), "Change, no harm": (big_change & ~harm).mean(),
         "Harm, no change": (~big_change & harm).mean(), "Harmful change": (big_change & harm).mean()}
print(f"{len(ts)} decision points x {z.shape[1]} variates = {x.size} points; thresholds: dX >= {xt:.3f}, R >= {yt:.3f}")
for k, v in share.items(): print(f"  {k:16} {100 * v:5.1f}%")

col = {"Stable": "#BDC3C7", "Change, no harm": "#7F8C8D", "Harm, no change": "#C0392B", "Harmful change": "#8E44AD"}
mask = {"Stable": ~big_change & ~harm, "Change, no harm": big_change & ~harm,
        "Harm, no change": ~big_change & harm, "Harmful change": big_change & harm}
fig, ax = plt.subplots(figsize=(4.8, 4.0))
for k in mask:
    ax.scatter(x[mask[k]], y[mask[k]], s=6, c=col[k], alpha=0.3 if k == "Stable" else 0.55, lw=0, label=f"{k} ({100 * share[k]:.0f}%)")
ax.axvline(xt, c="#555", lw=0.8, ls="--"); ax.axhline(yt, c="#555", lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_ylim(-6, 6)
ax.set_xlabel("Input change between past and coming week (dX, log scale)"); ax.set_ylabel("Future log-risk R")
ax.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=2)
fig.tight_layout(); os.makedirs("outputs/figures", exist_ok=True)
fig.savefig("outputs/figures/V2_quadrants_check.png", dpi=220); plt.show()
