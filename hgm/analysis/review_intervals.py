import json, os, sys
import numpy as np
os.environ["UQ_RETURN"] = "1"
from hgm import paths as HP
from hgm.uncertainty import intervals as UQ
from hgm.experiments.harm_detection import SPD

A_ALPHA = 0.1


def aci(base, aR, cal, n, H, alpha_t, gamma=0.005):
    B, Hh = aR.shape[1], aR.shape[2]
    srt = np.sort(aR[cal] / (base[cal] + 1e-12), 0); nc = len(srt)
    a = np.full((B, Hh), alpha_t); out = np.zeros_like(aR); pend = {}
    for i in range(n):
        lev = np.clip(1 - a, 0, 1); idx = np.clip(np.ceil(lev * (nc + 1)).astype(int) - 1, 0, nc - 1)
        out[i] = base[i] * np.take_along_axis(srt, idx[None], 0)[0]; pend[i] = out[i]
        o = i - H
        if o >= 0:
            a = a + gamma * (alpha_t - (aR[o] > pend.pop(o)).astype(float))
    return out


def pid(base, aR, cal, n, H, lr=0.05, scorecaster=False, hl=168):
    B, Hh = aR.shape[1], aR.shape[2]; sc = aR / (base + 1e-12); cal_s = sc[cal]
    rng = np.quantile(cal_s, 0.99, axis=0) - np.quantile(cal_s, 0.01, axis=0) + 1e-12
    q = np.quantile(cal_s, 1 - A_ALPHA, axis=0); csum = np.zeros((B, Hh)); qt = np.zeros_like(aR)
    lam = 0.5 ** (1 / hl); mcal = cal_s.mean(0); ew = mcal.copy()
    for i in range(n):
        o = i - H
        if o >= 0:
            e = (sc[o] > qt[o]).astype(float) - A_ALPHA; q = q + lr * rng * e; csum = csum + e
            ew = lam * ew + (1 - lam) * sc[o]
        k = max(i - H + 1, 1); x = np.clip(csum * np.log(k + 1) / (k + 1.0), -1.5, 1.5)
        qt[i] = np.maximum(q + rng * np.tan(x) + ((ew - mcal) if scorecaster else 0.0), 0)
    return qt * base


def main(data, seed, H=24, path=None):
    aR, W, nv, tr = UQ.main(data, H, seed, path=path)
    n, B, _ = aR.shape; cal = slice(H, nv - H); te = slice(nv + 720, n - 168 - H)
    L = (aR ** 2).mean(2)
    def harm_mask(win):
        fut = np.stack([L[i:i + win].mean(0) for i in range(n - win)] + [L[-win:].mean(0)] * win); h = np.zeros((n, B), bool)
        for b in range(B):
            th = np.quantile(fut[te, b], 0.9); h[te, b] = fut[te, b] >= th
        return h
    masks = {"h3_168": harm_mask(168), "h1_week": harm_mask(7 * SPD[data])}
    base_split = W["split"]; target = W["risk_aci"][te].mean()
    lo, hi = 1e-4, A_ALPHA
    for _ in range(14):
        mid = np.sqrt(lo * hi); w = aci(base_split, aR, cal, n, H, mid)[te].mean()
        if w > target: lo = mid
        else: hi = mid
    alpha_m = np.sqrt(lo * hi); Wm = aci(base_split, aR, cal, n, H, alpha_m)
    out = {"data": data, "seed": seed, "H": H, "alpha_matched": round(float(alpha_m), 5), "rows": {}}
    rows = [("split", W["split"]), ("risk_only", W["risk"]), ("bayes", W["bayes"]), ("aci", W["split_aci"]),
            ("risk_aci", W["risk_aci"]), ("aci_width_matched", Wm), ("pid", pid(W["split"], aR, cal, n, H)),
            ("pid_scorecaster", pid(W["split"], aR, cal, n, H, scorecaster=True)), ("risk_pid", pid(W["risk"], aR, cal, n, H))]
    for k, w in rows:
        cov = aR[te] <= w[te]; row = {"cover": round(float(cov.mean()), 4), "width": round(float(w[te].mean()), 5),
                                      "winkler": round(float((2 * w[te] + (2 / A_ALPHA) * np.clip(aR[te] - w[te], 0, None)).mean()), 5)}
        for name, h in masks.items(): row["cover_" + name] = round(float(cov[h[te]].mean()), 4)
        out["rows"][k] = row
    json.dump(out, open(f"{HP.DIAG}/h3review_{data}_s{seed}_H{H}.json", "w"), indent=1); print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 24)
