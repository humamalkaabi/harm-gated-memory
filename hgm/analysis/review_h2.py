import json, os, sys
import numpy as np
from scipy.stats import rankdata
from hgm import paths as HP
from hgm.data import load
from hgm.experiments.utility import streams
from hgm.experiments.harm_detection import SPD

WARM = 720


def ewma_revealed(d, H, hl=168):
    lam = 0.5 ** (1 / hl); n, B = d.shape; m = np.zeros((n, B)); cur = np.zeros(B)
    for i in range(n):
        o = i - H
        if o >= 0: cur = lam * cur + (1 - lam) * d[o]
        m[i] = cur
    return m


def deployable_choice(losses, H, warm=WARM):
    c = {k: v[:warm - H].mean() for k, v in losses.items()}
    return min(c, key=c.get)


def spearman_cols(a, b):
    ra = np.apply_along_axis(rankdata, 0, a); rb = np.apply_along_axis(rankdata, 0, b)
    ra -= ra.mean(0); rb -= rb.mean(0)
    return (ra * rb).sum(0) / np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0) + 1e-12)


def main(data, seed, H=24, gate_dir=None):
    z, tr, va, end = load(data); d_ = SPD[data]; W = 7 * d_
    Y, tgt = streams(data, H, seed, gate_dir); n, B, _ = tgt.shape
    loss = {k: ((v - tgt) ** 2).mean(2) for k, v in Y.items()}
    G = np.load(f"{gate_dir or HP.POLICY}/{data}_H{H}_s{seed}.npz")["A"]; nv = va - tr
    tA = np.stack([z[t:t + H].T for t in range(tr, tr + G.shape[0])]); LA = ((G - tA) ** 2).mean(2)
    lval = LA[H:nv - H].mean(0); LAt = loss["A"]
    def risk_at(i):
        rv = i - H
        return np.log(LAt[max(0, rv - W):rv].mean(0) / lval) if rv > 0 else np.zeros(B)
    r = np.array([risk_at(i) for i in range(n)])
    rv_val = np.array([np.log(LA[max(H, j - H - W):j - H].mean(0) / lval) for j in range(H + W + H, nv)])
    thr = np.quantile(rv_val, 0.9, axis=0)
    ts = np.arange(WARM, n - W, d_)
    Rf = np.array([np.log(LAt[i:i + W].mean(0) / lval) for i in ts]); harm = Rf >= np.quantile(Rf, 0.9, axis=0)
    out = {"data": data, "seed": seed, "H": H, "pairs": {}}
    sl = slice(WARM, n)
    for X in ("A", "B", "C"):
        dP = loss["P"] - loss[X]
        U = np.array([dP[i:i + W].mean(0) for i in ts]); rt = r[ts]
        rho = spearman_cols(rt, U)
        u_h = np.nanmean(np.where(harm, U, np.nan), 0); u_o = np.nanmean(np.where(~harm, U, np.nan), 0)
        fP, fX = Y["P"], Y[X]; V = ((fX - fP) ** 2).mean(2) + 1e-12
        m = ewma_revealed(dP, H); w = np.clip(0.5 + m / (2 * V), 0, 1)
        mix = lambda ww: (((ww[..., None] * fX + (1 - ww[..., None]) * fP) - tgt) ** 2).mean(2)
        mse = lambda L_: L_[sl].mean()
        base = mse(loss["P"])
        res = {"P": base, "X": mse(loss[X]), "avg": mse(mix(np.full((n, B), 0.5))), "v_gate": mse(mix(w)),
               "harm_trigger": mse(np.where(r > thr, loss[X], loss["P"]))}
        res["deployable_static"] = res[deployable_choice({"P": loss["P"], "X": loss[X], "avg": mix(np.full((n, B), 0.5))}, H)]
        res["best_static_hindsight"] = min(res["P"], res["X"], res["avg"])
        rel = {k: round(float(100 * (v / base - 1)), 3) for k, v in res.items()}
        out["pairs"][f"P_{X}"] = {"spearman_risk_utility_median": round(float(np.median(rho)), 3),
                                  "spearman_risk_utility_iqr": [round(float(np.quantile(rho, .25)), 3), round(float(np.quantile(rho, .75)), 3)],
                                  "utility_at_harm_minus_other_median": round(float(np.median(u_h - u_o)), 5),
                                  "share_variates_utility_higher_at_harm": round(float(np.mean(u_h > u_o)), 3),
                                  "mse_rel_to_P_pct": rel}
    os.makedirs(HP.DIAG, exist_ok=True)
    json.dump(out, open(f"{HP.DIAG}/h2review_{data}_s{seed}.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 24)
