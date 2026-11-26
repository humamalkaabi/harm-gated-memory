import glob, json, os, sys
import numpy as np
from scipy.stats import t as student
from hgm import paths as HP
from hgm.data import load
from hgm.experiments.harm_detection import SPD, auc

AL = 0.1


def _losses(data, seed, H):
    z, tr, va, end = load(data)
    Yh = np.load(f"{HP.POLICY}/{data}_H{H}_s{seed}.npz")["A"]; n, B, _ = Yh.shape
    tgt = np.stack([z[t:t + H].T for t in range(tr, tr + n)])
    return z, tr, va, Yh, tgt, n, B


def multiscale(data, seed, H=24):
    z, tr, va, Yh, tgt, n, B = _losses(data, seed, H)
    L = ((Yh - tgt) ** 2).mean(2); nv = va - tr; lval = L[H:nv - H].mean(0)
    d = SPD[data]; D, W = d, 7 * d
    scales = {"1d": D, "3d": 3 * D, "7d": W, "14d": 14 * D, "28d": 28 * D}; mx = max(scales.values())
    out = {}
    for lw in (1, 7, 28):
        ts = np.arange(nv + mx + H, n - D - max(W, lw * D) - H, d)
        if len(ts) < 40: continue
        S = {k: [] for k in scales}; R, dX = [], []
        for t in ts:
            a = t + D; rv = a - H
            for k, w in scales.items(): S[k].append(np.log(L[rv - w:rv].mean(0) / lval))
            R.append(np.log(L[a:a + lw * D].mean(0) / lval))
            dX.append(np.abs(z[tr + a:tr + a + W].mean(0) - z[tr + t - W:tr + t].mean(0)) / (z[tr + t - W:tr + t].std(0) + 1e-6))
        S = {k: np.array(v) for k, v in S.items()}; R = np.array(R); dX = np.array(dX)
        Z = {k: (v - v[:30].mean(0)) / (v[:30].std(0) + 1e-9) for k, v in S.items()}
        S["mean_of_scales"] = np.mean([Z[k] for k in scales], 0); S["max_of_scales"] = np.max([Z[k] for k in scales], 0)
        pos = np.zeros_like(R, bool); neg = np.zeros_like(R, bool)
        for b in range(B):
            pos[:, b] = R[:, b] >= np.quantile(R[:, b], 0.9)
            neg[:, b] = (R[:, b] <= np.median(R[:, b])) & (dX[:, b] >= np.quantile(dX[:, b], 0.75))
        ok = [b for b in range(B) if pos[:, b].sum() >= 5 and neg[:, b].sum() >= 5]
        out[f"label_{lw}d"] = {k: round(float(np.nanmean([auc(v[:, b][pos[:, b]], v[:, b][neg[:, b]]) for b in ok])), 4)
                               for k, v in S.items()}
    return out


def pareto(data, seed, H=24, max_var=64):
    z, tr, va, Yh, tgt, n, B = _losses(data, seed, H)
    if B > max_var:
        keep = np.sort(np.random.default_rng(0).choice(B, max_var, replace=False))
        Yh, tgt, B = Yh[:, keep], tgt[:, keep], max_var
    R = tgt - Yh; aR = np.abs(R); nv = va - tr; L = (R ** 2).mean(2); cal = slice(H, nv - H)

    def sigma(hl):
        lam = 0.5 ** (1 / hl); sig = np.zeros((n, B)); s2 = L[:max(1, nv - H)].mean(0)
        for i in range(n):
            if i - H >= 0: s2 = lam * s2 + (1 - lam) * L[i - H]
            sig[i] = np.sqrt(s2)
        return sig

    def aci(base, g):
        sc = np.sort(aR[cal] / (base[cal] + 1e-12), 0); nc = len(sc); a = np.full((B, H), AL); out = np.zeros_like(aR); pend = {}
        for i in range(n):
            idx = np.clip(np.ceil(np.clip(1 - a, 0, 1) * (nc + 1)).astype(int) - 1, 0, nc - 1)
            out[i] = base[i] * np.take_along_axis(sc, idx[None], 0)[0]; pend[i] = out[i]
            if i - H >= 0: a = a + g * (AL - (aR[i - H] > pend.pop(i - H)).astype(float))
        return out

    def pid(base, lr):
        sc = aR / (base + 1e-12); cal_s = sc[cal]
        rng = np.quantile(cal_s, 0.99, axis=0) - np.quantile(cal_s, 0.01, axis=0) + 1e-12
        q = np.quantile(cal_s, 1 - AL, axis=0); csum = np.zeros((B, H)); qt = np.zeros_like(aR)
        for i in range(n):
            o = i - H
            if o >= 0:
                e = (sc[o] > qt[o]).astype(float) - AL; q = q + lr * rng * e; csum = csum + e
            k = max(i - H + 1, 1)
            x = np.clip(csum * np.log(k + 1) / ((k + 1) * 1.0), -1.5, 1.5)
            qt[i] = np.maximum(q + rng * np.tan(x), 0)
        return qt * base

    def dtaci(base, gammas=(0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128), I=100):
        sc = np.sort(aR[cal] / (base[cal] + 1e-12), 0); nc = len(sc); K = len(gammas); g = np.array(gammas)[:, None, None]
        a = np.full((K, B, H), AL); w = np.ones((K, B, H)) / K; out = np.zeros_like(aR); lev = {}
        eta = np.sqrt(3 / I) * np.sqrt((np.log(K * I) + 2) / ((1 - AL) ** 2 * AL ** 3 + AL ** 2 * (1 - AL) ** 3)); sg = 1 / (2 * I)
        for i in range(n):
            ab = (w * a).sum(0)
            idx = np.clip(np.ceil(np.clip(1 - ab, 0, 1) * (nc + 1)).astype(int) - 1, 0, nc - 1)
            out[i] = base[i] * np.take_along_axis(sc, idx[None], 0)[0]; lev[i] = a.copy()
            o = i - H
            if o >= 0:
                s = aR[o] / (base[o] + 1e-12)
                beta = 1 - (np.sum(sc <= s[None], 0) / (nc + 1))
                ak = lev.pop(o); d = beta[None] - ak
                pin = AL * d - np.minimum(0, d)
                w = w * np.exp(-eta * pin); w = w / w.sum(0, keepdims=True); w = (1 - sg) * w + sg / K
                a = a + g * (AL - (beta[None] < ak).astype(float))
        return out

    W = {}; sig = sigma(168)
    for k in (0, 0.25, 0.5, 0.75, 1.0):
        s = sig ** k; base = s[..., None] * np.quantile(aR[cal] / s[cal][..., None], 1 - AL, axis=0)[None]
        W[f"risk^{k}"] = base
        for g in (0.001, 0.005, 0.02): W[f"risk^{k}+aci{g}"] = aci(base, g)
    sf = np.maximum(sig, np.quantile(sig[cal], 0.25, axis=0))
    base = sf[..., None] * np.quantile(aR[cal] / sf[cal][..., None], 1 - AL, axis=0)[None]
    W["risk_floor"] = base; W["risk_floor+aci0.005"] = aci(base, 0.005)
    one = np.ones((n, B, 1)); rsk = sig[..., None] * np.ones((1, 1, H))
    for lr in (0.01, 0.05, 0.1):
        W[f"pid{lr}"] = pid(one * np.ones((1, 1, H)), lr); W[f"risk+pid{lr}"] = pid(rsk, lr)
    W["dtaci"] = dtaci(one * np.ones((1, 1, H))); W["risk+dtaci"] = dtaci(rsk)
    for hl in (48, 168, 672):
        lam = 0.5 ** (1 / hl); a0 = 5.0; b0 = a0 * (R[cal] ** 2).mean(0); a_ = np.full((B, H), a0); b_ = b0.copy(); Wb = np.zeros_like(aR)
        for i in range(n):
            if i - H >= 0: a_ = lam * a_ + (1 - lam) * a0 + 0.5; b_ = lam * b_ + (1 - lam) * b0 + 0.5 * R[i - H] ** 2
            Wb[i] = student.ppf(1 - AL / 2, 2 * a_) * np.sqrt(b_ / a_)
        W[f"bayes_hl{hl}"] = Wb
    te = slice(nv + 720, n - 168 - H)
    fut = np.stack([L[i:i + 168].mean(0) for i in range(n - 168)] + [L[-168:].mean(0)] * 168)
    harm = np.zeros((n, B), bool)
    for b in range(B): harm[te, b] = fut[te, b] >= np.quantile(fut[te, b], 0.9)
    ws = W["risk^0"][te].mean(); rows = {}
    for k, w in W.items():
        cov = aR[te] <= w[te]
        rows[k] = dict(cover=float(cov.mean()), cover_harm=float(cov[harm[te]].mean()), width=float(w[te].mean() / ws),
                       winkler=float((2 * w[te] + (2 / AL) * np.clip(aR[te] - w[te], 0, None)).mean()))
    ref = rows["risk^0+aci0.005"]["winkler"]
    for r in rows.values(): r["winkler_vs_aci"] = r["winkler"] / ref
    cand = {k: r for k, r in rows.items() if r["cover"] >= 0.88}
    for k, r in rows.items():
        r["pareto"] = k in cand and not any(o["cover_harm"] >= r["cover_harm"] and o["width"] <= r["width"]
                                            and (o["cover_harm"] > r["cover_harm"] or o["width"] < r["width"]) for o in cand.values())
    return rows


def gate_bound(data, seed, H=24, warm=720, max_var=64):
    from hgm.experiments.utility import streams
    Y, tgt = streams(data, H, seed); n, B, _ = tgt.shape
    keep = np.sort(np.random.default_rng(0).choice(B, max_var, replace=False)) if B > max_var else np.arange(B)
    Y = {k: v[:, keep] for k, v in Y.items()}; tgt = tgt[:, keep]; B = len(keep)
    L = {k: ((v - tgt) ** 2).mean(2) for k, v in Y.items()}; out = {}
    idx = np.arange(warm + 672 + H, n)
    for X in ("A", "B", "C"):
        d = L["P"] - L[X]; V = ((Y[X] - Y["P"]) ** 2).mean(2) + 1e-12
        cs = np.concatenate([np.zeros((1, B)), np.cumsum(d, 0)])
        feat = [np.ones((len(idx), B)), V[idx]] + [(cs[idx - H] - cs[idx - H - w]) / w for w in (24, 168, 672)]
        F = np.stack(feat, -1); mh = np.zeros((len(idx), B))
        for b in range(B):
            beta = np.linalg.lstsq(F[:, b], d[idx, b], rcond=None)[0]; mh[:, b] = F[:, b] @ beta
        Vi = V[idx]; lp, lx = L["P"][idx], L[X][idx]
        loss = lambda w: (1 - w) * lp + w * lx - w * (1 - w) * Vi
        w_or = np.clip(0.5 + mh / (2 * Vi), 0, 1)
        grid = np.linspace(0, 1, 101); lc = np.array([loss(c).mean(0) for c in grid]); cb = grid[lc.argmin(0)]
        l_half, l_best = loss(0.5).mean(), np.mean([lc[:, b].min() for b in range(B)])
        out[X] = dict(realised_oracle_vs_avg_pct=round(100 * (l_half - loss(w_or).mean()) / l_half, 3),
                      realised_oracle_vs_best_static_pct=round(100 * (l_best - loss(w_or).mean()) / l_best, 3),
                      r2_of_m=round(float(1 - ((d[idx] - mh) ** 2).sum() / ((d[idx] - d[idx].mean(0)) ** 2).sum()), 4))
    return out


def run(data, seed, H=24):
    out = f"{HP.DIAG}/sens_{data}_H{H}_s{seed}.json"
    if os.path.exists(out): return
    res = {"data": data, "seed": seed, "H": H, "multiscale": multiscale(data, seed, H), "pareto": pareto(data, seed, H)}
    try: res["gate_bound"] = gate_bound(data, seed, H)
    except FileNotFoundError as e: print("gate bound skipped:", e, flush=True)
    json.dump(res, open(out, "w"), indent=1); print("wrote", out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2: run(sys.argv[1], int(sys.argv[2]))
    else:
        for f in sorted(glob.glob(f"{HP.POLICY}/*_H*_s*.npz")):
            d, h, s = os.path.basename(f)[:-4].split("_"); run(d, int(s[1:]), int(h[1:]))
