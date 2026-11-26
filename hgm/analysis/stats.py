from hgm import paths as HP
import os, sys, json, numpy as np
from scipy.stats import spearmanr, rankdata
from hgm.experiments import utility as U
from hgm.policies.gates import gate
os.environ["UQ_RETURN"] = "1"
from hgm.uncertainty import intervals as dev_uq
RNG = np.random.default_rng(1); NB = 2000; H = 24


def auc(p, n):
    if len(p) == 0 or len(n) == 0: return np.nan
    r = rankdata(np.r_[p, n]); return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def boot_blocks(n, size):
    blocks = [np.arange(i, min(i + size, n)) for i in range(0, n, size)]
    for _ in range(NB):
        yield np.concatenate([blocks[j] for j in RNG.integers(0, len(blocks), len(blocks))])


def h6(data, seed, gate_dir):
    Y, tgt = U.streams(data, H, seed, gate_dir); n, B, _ = tgt.shape; warm = 720
    Ls = {k: ((v - tgt) ** 2).mean(2) for k, v in Y.items()}; out = {"pred": {}, "gates": {}}
    for X in ("A", "B", "C"):
        d = Ls["P"] - Ls[X]
        for W in (24, 168):
            idx = np.arange(warm + W + H, n - W, 24)
            pa = np.stack([d[i - H - W:i - H].mean(0) for i in idx]); fa = np.stack([d[i:i + W].mean(0) for i in idx])
            def stat(rows):
                a = np.nanmean([auc(pa[rows, b][fa[rows, b] > 0], pa[rows, b][fa[rows, b] <= 0]) for b in range(B)])
                r = np.nanmean([spearmanr(pa[rows, b], fa[rows, b])[0] for b in range(B)])
                return a, r
            pt = stat(np.arange(len(idx))); bs = np.array([stat(r) for r in boot_blocks(len(idx), 7)])
            ci_a = np.nanpercentile(bs[:, 0], [5, 95]); ci_r = np.nanpercentile(bs[:, 1], [5, 95])
            ok = bool(ci_a[0] > 0.45 and ci_a[1] < 0.55 and ci_r[0] > -0.10 and ci_r[1] < 0.10)
            bs21 = np.array([stat(r) for r in boot_blocks(len(idx), 21)])
            ca21 = np.nanpercentile(bs21[:, 0], [5, 95]); cr21 = np.nanpercentile(bs21[:, 1], [5, 95])
            ok21 = bool(ca21[0] > 0.45 and ca21[1] < 0.55 and cr21[0] > -0.10 and cr21[1] < 0.10)
            out["pred"][f"P-{X} W{W}"] = dict(auroc=round(pt[0], 3), auroc_ci90=np.round(ci_a, 3).tolist(),
                                              rho=round(pt[1], 3), rho_ci90=np.round(ci_r, 3).tolist(), tost=ok,
                                              auroc_ci90_b21=np.round(ca21, 3).tolist(), rho_ci90_b21=np.round(cr21, 3).tolist(), tost_b21=ok21)
        sc = slice(warm, None); avg = (Y["P"] + Y[X]) / 2; la = ((avg - tgt) ** 2).mean((1, 2))[sc]
        wX = {}
        for b in range(B):
            g = U.gates_pair(Ls["P"][:, b], Ls[X][:, b], H, warm)
            for k in g: wX.setdefault(k, np.zeros((n, B)))[:, b] = g[k]
        def cands():
            for k, w in wX.items():
                w32 = w.astype(np.float32)[..., None]; yield k, (1 - w32) * Y["P"] + w32 * Y[X]
            yield "hedge", gate(Y, tgt, H, ["P", X], "hedge", 0.1, nval=warm)[0].astype(np.float32)
        for k, pr in cands():
            lg = ((pr - tgt) ** 2).mean((1, 2))[sc]
            imp = lambda r: (la[r].mean() - lg[r].mean()) / la[r].mean()
            bs = np.array([imp(r) for r in boot_blocks(len(la), 168)])
            ub = float(np.percentile(bs, 95)); ub975 = float(np.percentile(bs, 97.5))
            out["gates"][f"{k}(P,{X})"] = dict(improvement_pct=round(100 * imp(np.arange(len(la))), 2), upper95_pct=round(100 * ub, 2), supported=ub < 0.005,
                                               upper975_pct=round(100 * ub975, 2), supported975=ub975 < 0.005)
    return out


def h7(data, seed, path):
    aR, Wd, nv, tr = dev_uq.main(data, H, seed, path=path)
    n, B, _ = aR.shape; R = None
    G = dict(np.load(path)) if path else None
    from hgm.data import load
    z, tr_, va, end = load(data); A = np.load(path)["A"]; tgt = np.stack([z[t:t + H].T for t in range(tr_, tr_ + n)])
    L = ((A - tgt) ** 2).mean(2); te = slice(nv + 720, n - 168 - H)
    fut = np.stack([L[i:i + 168].mean(0) for i in range(n - 168)] + [L[-168:].mean(0)] * 168)
    harm = np.zeros((n, B), bool)
    for b in range(B):
        th = np.quantile(fut[te, b], 0.9); harm[te, b] = fut[te, b] >= th
    cov = {k: (aR <= Wd[k]).mean(2) for k in ("split_aci", "risk_aci")}
    wink = {k: (2 * Wd[k] + (2 / 0.1) * np.clip(aR - Wd[k], 0, None)).mean(2) for k in ("split_aci", "risk_aci")}
    idx = np.arange(te.start, te.stop)
    def stat(r):
        o = idx[r]; hm = harm[o]
        dc = (cov["risk_aci"][o][hm]).mean() - (cov["split_aci"][o][hm]).mean()
        wr = wink["risk_aci"][o].mean() / wink["split_aci"][o].mean()
        return dc, wr
    pt = stat(np.arange(len(idx))); bs = np.array([stat(r) for r in boot_blocks(len(idx), 168)])
    lo = float(np.percentile(bs[:, 0], 5)); hi = float(np.percentile(bs[:, 1], 95))
    lo975 = float(np.percentile(bs[:, 0], 2.5)); hi975 = float(np.percentile(bs[:, 1], 97.5))
    return dict(cov_gain=round(pt[0], 3), cov_gain_lower95=round(lo, 3), winkler_ratio=round(pt[1], 4), winkler_upper95=round(hi, 4),
                coverage_ok=lo > 0, width_ok=hi < 1.05, cov_gain_lower975=round(lo975, 3), winkler_upper975=round(hi975, 4),
                coverage_ok975=lo975 > 0, width_ok975=hi975 < 1.05)


if __name__ == "__main__":
    data, seed = sys.argv[1], int(sys.argv[2])
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    gd = HP.POLICY
    path = f"{HP.POLICY}/{data}_H{H}_s{seed}.npz"
    res = {"data": data, "seed": seed, "H": H}
    if os.path.exists(f"{HP.ONLINETSF}/preds/{data}_{H}_RL_Proceed.npz") and not os.environ.get("HGM_SKIP_H6"): res["H6"] = h6(data, seed, gd)
    res["H7"] = h7(data, seed, path)
    json.dump(res, open(f"{HP.DIAG}/stats_{data}_s{seed}.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
