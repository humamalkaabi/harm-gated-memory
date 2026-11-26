from hgm import paths as HP
import json, numpy as np
from hgm.monitors.labels import auroc
rng = np.random.default_rng(42); P, H, T0, TAU, T1, NP = 4, 24, 600, 800, 1000, 300
d = 16; R1 = rng.standard_normal((d, P)); R2 = rng.standard_normal((d, P))


def series(phi, seed_noise, tau=None, phi2=None, jump=0.0):
    g = np.random.default_rng(seed_noise); y = np.zeros(T1)
    for t in range(P, T1):
        c = phi2 if (tau is not None and t >= tau) else phi
        y[t] = c @ y[t - P:t][::-1] + (jump if (tau is not None and t >= tau) else 0.0) + g.standard_normal() * 0.5
    return y


def signals(y, coef, ref):
    n = len(y); L = np.full(n, np.nan); rev = np.full(n, np.nan)
    fc = {}
    for o in range(T0, n - H):
        hist = list(y[o - P:o]); out = []
        for h in range(H):
            v = coef @ np.array(hist[-P:][::-1]); out.append(v); hist.append(v)
        fc[o] = np.array(out)
    for t in range(T0 + H, n):
        o = t - H + 1; rev[t] = ((fc[o] - y[o:o + H]) ** 2).mean() if o in fc else np.nan
    risk = np.full(n, np.nan); err = rev.copy(); nov = np.full(n, np.nan); sur = np.full(n, np.nan)
    for t in range(T0 + H + 24, n):
        risk[t] = np.log(np.nanmean(rev[t - 23:t + 1]) / ref)
        nov[t] = abs(y[t - 23:t + 1].mean() - y[t - 123:t - 23].mean()) / (y[t - 123:t - 23].std() + 1e-8)
    M = np.zeros((d, d)); S = np.zeros((d, d)); mu, sd = y[:T0].mean(), y[:T0].std()
    for t in range(P, n):
        xn = (y[t - P + 1:t + 1] - mu) / sd; k = R1 @ xn; k /= np.linalg.norm(k) + 1e-8; v = R2 @ xn
        G = np.outer(M @ k - v, k); sur[t] = np.linalg.norm(G); S = 0.5 * S - 0.5 * G; M = 0.95 * M + S
    return dict(risk=risk, err=err, surprise=sur, novelty=nov)


phi = np.array([0.5, 0.2, 0.1, 0.05]); out = {k: {} for k in ("risk", "err", "surprise", "novelty")}
SA, SB = [], []
for i in range(NP):
    phi2 = phi.copy(); phi2[0] = 0.5 + 0.3 * rng.standard_normal()
    jump = rng.choice([-1, 1]) * (0.8 + 0.4 * rng.random())
    yA = series(phi, 1000 + i); yB = series(phi, 1000 + i, TAU, phi2, jump)
    assert np.allclose(yA[:TAU], yB[:TAU])
    X = np.stack([yA[t - P:t][::-1] for t in range(P, T0)]); coef = np.linalg.lstsq(X, yA[P:T0], rcond=None)[0]
    ref = 0.25 * H
    SA.append(signals(yA, coef, 1.0)); SB.append(signals(yB, coef, 1.0))
for k in range(-24, 97, 4):
    t = TAU + k
    for s in out:
        a = np.array([x[s][t] for x in SA]); b = np.array([x[s][t] for x in SB])
        out[s][k] = 0.5 if np.allclose(a, b, equal_nan=True) else round(auroc(b, a, n_boot=1)[0], 3)
json.dump(out, open(HP.DIAG + "/e20.json", "w"))
for s in out:
    print(f"{s:9}", "  ".join(f"k={k}:{v}" for k, v in out[s].items() if k in (-24, -4, 0, 4, 12, 24, 48, 96)))
