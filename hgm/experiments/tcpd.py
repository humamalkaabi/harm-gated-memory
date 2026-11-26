from hgm import paths as HP
import json, glob, numpy as np
from scipy.stats import wilcoxon
from hgm.monitors.labels import auroc

P = 5; rng = np.random.default_rng(0); d = 16
R1 = rng.standard_normal((d, P)); R2 = rng.standard_normal((d, P))
ann = json.load(open(HP.TCPD + "/annotations.json"))


def signals(y):
    n = len(y); e = np.full(n, np.nan); s = np.full(n, np.nan); nov = np.full(n, np.nan); risk = np.full(n, np.nan)
    th = np.zeros(P + 1); Pm = np.eye(P + 1) * 100.0; lam = 0.98
    M = np.zeros((d, d)); S = np.zeros((d, d)); errs, grads = [], []
    for t in range(P, n):
        past = y[:t]; mu, sd = past.mean(), past.std() + 1e-8
        x = np.r_[y[t - P:t], 1.0]
        yh = x @ th; err = (y[t] - yh) ** 2
        if len(errs) > 3: e[t] = np.log((err + 1e-12) / (np.median(errs) + 1e-12))
        errs.append(err)
        if len(errs) > 10: risk[t] = np.log((np.mean(errs[-5:]) + 1e-12) / (np.median(errs[:-5]) + 1e-12))
        Kg = Pm @ x / (lam + x @ Pm @ x); th = th + Kg * (y[t] - yh); Pm = (Pm - np.outer(Kg, x @ Pm)) / lam
        xn = (y[t - P + 1:t + 1] - mu) / sd
        k = R1 @ xn; k = k / (np.linalg.norm(k) + 1e-8); v = R2 @ xn
        G = np.outer(M @ k - v, k); g = np.linalg.norm(G)
        if len(grads) > 3: s[t] = np.log((g + 1e-12) / (np.median(grads) + 1e-12))
        grads.append(g)
        S = 0.5 * S - 0.5 * G; M = 0.95 * M + S
        nov[t] = abs(y[t] - mu) / sd
    return dict(e=e, s=s, nov=nov, risk=risk)


rows = []
for f in sorted(glob.glob(HP.TCPD + "/datasets/*/*.json")):
    D = json.load(open(f)); name = D["name"]
    if D["n_obs"] < 100 or D["n_dim"] != 1: continue
    y = np.array([np.nan if v is None else v for v in D["series"][0]["raw"]], float)
    if np.isnan(y).any(): y = np.interp(np.arange(len(y)), np.flatnonzero(~np.isnan(y)), y[~np.isnan(y)])
    cps = sorted({c for a in ann[name].values() for c in a})
    if not cps: continue
    n = len(y); t = np.arange(n); dist = np.min([np.where(t >= c, t - c, 10 ** 6) for c in cps], 0)
    near = np.min([np.abs(t - c) for c in cps], 0)
    pos = (dist <= 5) & (t >= 15); neg = (near > 10) & (t >= 15)
    if pos.sum() < 3 or neg.sum() < 10: continue
    sg = signals(y)
    r = {k: auroc(v[pos], v[neg], n_boot=200)[0] for k, v in sg.items()}
    rows.append((name, int(pos.sum()), r)); print(f"{name:20} pos {pos.sum():3}  " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))
E = np.array([r["e"] for _, _, r in rows]); Sg = np.array([r["s"] for _, _, r in rows]); N = np.array([r["nov"] for _, _, r in rows]); Rk = np.array([r["risk"] for _, _, r in rows])
print(f"\n{len(rows)} series | mean AUROC  error {E.mean():.3f}  surprise {Sg.mean():.3f}  novelty {N.mean():.3f}  risk(5) {np.nanmean(Rk):.3f}")
print(f"error > surprise in {(E > Sg).sum()}/{len(rows)} series, Wilcoxon p = {wilcoxon(E - Sg, alternative='greater').pvalue:.4f}")
json.dump(dict(rows=rows), open(HP.DIAG + "/tcpd_v2.json", "w"))
