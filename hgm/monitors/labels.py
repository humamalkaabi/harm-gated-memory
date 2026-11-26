from hgm import paths as HP
import argparse, json, os, sys
import numpy as np, torch
from scipy.stats import spearmanr
from hgm.data import load
from hgm.models.forecaster import BenchMem, views
from hgm.models.controllers import Ctrl
torch.set_grad_enabled(False); torch.set_num_threads(1)


def ridge_fit(X, Y, lam=1.0):
    A = X.T @ X + lam * np.eye(X.shape[1]); return np.linalg.solve(A, X.T @ Y)


def windows(x, a, b, Lr, Hr):
    idx = np.arange(a, b - Hr + 1)
    idx = idx[idx - Lr >= 0]
    X = np.stack([x[i - Lr:i] for i in idx]); Y = np.stack([x[i:i + Hr] for i in idx])
    return np.c_[X, np.ones(len(X))], Y


def labels(z, ts, W, Lr=96, Hr=24):
    N = z.shape[1]; dC = np.full((len(ts), N), np.nan); dX = np.full((len(ts), N), np.nan)
    for i, t in enumerate(ts):
        for c in range(N):
            x = z[:, c]
            Xp, Yp = windows(x, t - W, t, Lr, Hr)
            Xa, Ya = windows(x, t, t + W // 2, Lr, Hr); Xb, Yb = windows(x, t + W // 2, t + W, Lr, Hr)
            Bp, Ba = ridge_fit(Xp, Yp), ridge_fit(Xa, Ya)
            mp, ma = ((Xb @ Bp - Yb) ** 2).mean(), ((Xb @ Ba - Yb) ** 2).mean()
            dC[i, c] = (mp - ma) / (ma + 1e-8)
            pre, post = x[t - W:t], x[t:t + W]
            dX[i, c] = abs(post.mean() - pre.mean()) / (pre.std() + 1e-8) + abs(np.log((post.std() + 1e-8) / (pre.std() + 1e-8)))
    return dC, dX


def auroc(pos, neg, n_boot=500, seed=0):
    pos, neg = np.asarray(pos), np.asarray(neg)
    def a(p, q):
        r = np.argsort(np.argsort(np.r_[p, q])) + 1
        return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q))
    rng = np.random.default_rng(seed); bs = [a(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return float(a(pos, neg)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(data, H, seed, out=HP.DIAG):
    os.makedirs(out, exist_ok=True); fn = f"{out}/{data}_H{H}_s{seed}.json"
    if os.path.exists(fn): return
    z, tr, va, end = load(data); zt = torch.tensor(z.T)
    m = BenchMem(L=336, H=H); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_mem_linear_s{seed}.pt")); m.eval()
    B = zt.shape[0]; M, S = m.init_state(B); buf = {}
    T1 = end - H + 1; n = T1 - tr
    sig = {k: np.full((n, B), np.nan, np.float32) for k in ("s", "w", "e_raw", "nov")}
    Eh = np.full((n, H, B), np.nan, np.float32)
    for i, t in enumerate(range(tr, T1)):
        tok, xn, mu, sd = views(zt, t, m.L, m.P)
        row = zt[:, t - 1]
        for h in range(1, H + 1):
            if t - h in buf: Eh[i, h - 1] = ((buf[t - h][:, h - 1] - row) ** 2).numpy()
        yn, M, S, info = m.step(tok, xn, None, M, S, torch.ones(B))
        buf[t] = yn * sd + mu; buf.pop(t - H - 1, None)
        sig["s"][i] = info["s"].numpy(); sig["w"][i] = info["w"].numpy()
        sig["nov"][i] = (((row[:, None] - mu) / sd).abs()[:, 0]).numpy()
    vs = slice(H, va - tr)
    mu_h = np.nanmean(Eh[vs], axis=(0, 2)); sd_h = np.nanstd(Eh[vs], axis=(0, 2)) + 1e-6
    sig["e"] = np.nanmean((Eh - mu_h[None, :, None]) / sd_h[None, :, None], axis=1)
    W = 2 * 336 if data != "ETTh2" else 2 * 168
    W = min(W, 1344)
    ts = np.arange(va + W, end - W - H, max(H, 24))
    dC, dX = labels(z, ts, W)
    agg = {k: np.stack([np.nanmean(sig[k][t - tr:t - tr + H], 0) for t in ts]) for k in ("s", "w", "e", "nov")}
    pos = dC >= np.nanpercentile(dC, 90)
    neg = (dC <= np.nanpercentile(dC, 50)) & (dX >= np.nanpercentile(dX, 75))
    res = dict(data=data, H=H, seed=seed, n_times=len(ts), n_var=B, n_pos=int(pos.sum()), n_neg=int(neg.sum()), W=W, auroc={}, spearman={})
    for k, v in agg.items():
        res["auroc"][k] = auroc(v[pos], v[neg])
        ok = ~np.isnan(v) & ~np.isnan(dC)
        res["spearman"][k] = dict(dC=float(spearmanr(v[ok], dC[ok])[0]), dX=float(spearmanr(v[ok], dX[ok])[0]))
    res["spearman_dC_dX"] = float(spearmanr(dC.ravel(), dX.ravel(), nan_policy="omit")[0])
    json.dump(res, open(fn, "w"))
    print(data, H, seed, {k: round(v[0], 3) for k, v in res["auroc"].items()},
          {k: (round(v["dC"], 2), round(v["dX"], 2)) for k, v in res["spearman"].items()}, res["n_pos"], res["n_neg"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--H", type=int, default=24); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); main(a.data, a.H, a.seed)
