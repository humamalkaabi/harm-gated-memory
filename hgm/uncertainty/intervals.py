from hgm import paths as HP
import sys, os, numpy as np
from scipy.stats import t as student
from hgm.data import load
torch_free = True
A_ALPHA = 0.1


def main(data="ETTh1", H=24, seed=1, path=None):
    z, tr, va, end = load(data)
    G = dict(np.load(path or f"{HP.POLICY}/{data}_H{H}_s{seed}.npz")); Yh = G["A"]; n, B, _ = Yh.shape
    if os.environ.get("HGM_UQ_SUBSET") and B > 64:
        idx = np.sort(np.random.default_rng(0).choice(B, 64, replace=False)); Yh = Yh[:, idx]; z = z[:, idx]; B = 64
    tgt = np.stack([z[t:t + H].T for t in range(tr, tr + n)]); R = tgt - Yh; aR = np.abs(R)
    nv = va - tr
    lam = 0.5 ** (1 / 168)
    L = (R ** 2).mean(2); sig = np.zeros((n, B)); s2 = L[:max(1, nv - H)].mean(0)
    for i in range(n):
        o = i - H
        if o >= 0: s2 = lam * s2 + (1 - lam) * L[o]
        sig[i] = np.sqrt(s2)
    cal = slice(H, nv - H)
    q_split = np.quantile(aR[cal], 1 - A_ALPHA, axis=0)
    q_risk = np.quantile(aR[cal] / sig[cal][..., None], 1 - A_ALPHA, axis=0)
    W = {"split": np.broadcast_to(q_split, aR.shape).copy(), "risk": sig[..., None] * q_risk[None]}
    for base in ("split", "risk"):
        sc_cal = aR[cal] / (W[base][cal] + 1e-12); srt = np.sort(sc_cal, 0); nc = len(srt)
        a = np.full((B, H), A_ALPHA); out = np.zeros_like(aR); pend = {}
        for i in range(n):
            lev = np.clip(1 - a, 0, 1); idx = np.clip(np.ceil(lev * (nc + 1)).astype(int) - 1, 0, nc - 1)
            mult = np.take_along_axis(srt, idx[None], 0)[0]; out[i] = W[base][i] * mult
            pend[i] = out[i]
            o = i - H
            if o >= 0:
                err = (aR[o] > pend.pop(o)).astype(float); a = a + 0.005 * (A_ALPHA - err)
        W[base + "_aci"] = out
    a0 = 5.0; b0 = a0 * (R[cal] ** 2).mean(0); a_ = np.full((B, H), a0); b_ = b0.copy(); Wb = np.zeros_like(aR)
    for i in range(n):
        o = i - H
        if o >= 0:
            a_ = lam * a_ + (1 - lam) * a0 + 0.5; b_ = lam * b_ + (1 - lam) * b0 + 0.5 * R[o] ** 2
        nu = 2 * a_; scale = np.sqrt(b_ / a_); Wb[i] = student.ppf(1 - A_ALPHA / 2, nu) * scale
    W["bayes"] = Wb
    te = slice(nv + 720, n - 168 - H)
    fut = np.stack([L[i:i + 168].mean(0) for i in range(n - 168)] + [L[-168:].mean(0)] * 168)
    harm = np.zeros((n, B), bool)
    for b in range(B):
        th = np.quantile(fut[te, b], 0.9); harm[te, b] = fut[te, b] >= th
    if os.environ.get("UQ_RETURN"): return aR, W, nv, tr
    print(f"\n{data} H{H} s{seed}: target coverage 90 %")
    print(f"  {'method':10} cover  cover@harm  width/split  winkler")
    ws = W["split"][te].mean()
    for k, w in W.items():
        cov = aR[te] <= w[te]; ch = cov[harm[te]].mean()
        wink = (2 * w[te] + (2 / A_ALPHA) * np.clip(aR[te] - w[te], 0, None)).mean()
        print(f"  {k:10} {cov.mean():.3f}  {ch:.3f}       {w[te].mean() / ws:.3f}       {wink:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ETTh1")
