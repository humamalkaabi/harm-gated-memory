from hgm import paths as HP
import sys, json, numpy as np, torch
from river import drift
from river.drift import binary as bdrift
from hgm.data import load
from hgm.models.forecaster import BenchMem
from hgm.monitors.memory_stream import run as run_mem
torch.set_grad_enabled(False); torch.set_num_threads(1)
SPD = dict(ETTh1=24, ETTh2=24, ETTm1=96, Weather=144, WeatherClean=144, ECL=24, Traffic=24, ETTm2=96)
RNG = np.random.default_rng(0)


def auc(p, n):
    p, n = p[~np.isnan(p)], n[~np.isnan(n)]
    if len(p) == 0 or len(n) == 0: return np.nan
    allv = np.r_[p, n]; r = allv.argsort().argsort() + 1.0
    from scipy.stats import rankdata
    r = rankdata(allv)
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def river_alarms(stream, make, n, binary=False):
    det = make(); out = np.zeros(n, bool)
    for i, x in enumerate(stream):
        if np.isnan(x): continue
        det.update(int(x) if binary else float(x))
        if det.drift_detected: out[i] = True
    return out


def label_variants(data, seed, H, S, R, dX, B, nt, n_boot=500):
    keys = ["risk", "err_day", "surprise", "novelty"]; out = {"data": data, "seed": seed, "H": H, "variants": {}}
    blocks = [np.arange(i, min(i + 7, nt)) for i in range(0, nt, 7)]
    for q in (0.80, 0.90, 0.95):
        for negdef in ("registered", "all", "lowchange"):
            pos = np.zeros_like(R, bool); neg = np.zeros_like(R, bool)
            for b in range(B):
                pos[:, b] = R[:, b] >= np.quantile(R[:, b], q)
                lowR = R[:, b] <= np.median(R[:, b])
                neg[:, b] = (lowR & (dX[:, b] >= np.quantile(dX[:, b], 0.75))) if negdef == "registered" else \
                    (lowR & (dX[:, b] <= np.quantile(dX[:, b], 0.25))) if negdef == "lowchange" else ~pos[:, b]
            ok = [b for b in range(B) if pos[:, b].sum() >= 5 and neg[:, b].sum() >= 5]
            def score(idx):
                return {k: np.nanmean([auc(S[k][idx, b][pos[idx, b]], S[k][idx, b][neg[idx, b]]) for b in ok]) for k in keys}
            base = score(np.arange(nt)); boots = []
            for _ in range(n_boot):
                idx = np.concatenate([blocks[j] for j in RNG.integers(0, len(blocks), len(blocks))]); boots.append(score(idx))
            res = {}
            for k in keys:
                bv = np.array([x[k] for x in boots]); res[k] = [round(float(base[k]), 3), round(float(np.nanpercentile(bv, 2.5)), 3), round(float(np.nanpercentile(bv, 97.5)), 3)]
                if k != "risk":
                    dv = np.array([x["risk"] - x[k] for x in boots]); res["risk_minus_" + k] = [round(float(base["risk"] - base[k]), 3), round(float(np.nanpercentile(dv, 2.5)), 3)]
            out["variants"][f"q{int(q*100)}_{negdef}"] = res
    from scipy.stats import rankdata
    rk = {k: np.apply_along_axis(rankdata, 0, np.nan_to_num(S[k], nan=-np.inf)) for k in keys}; rR = np.apply_along_axis(rankdata, 0, R)
    def rho(idx):
        o = {}
        for k in keys:
            a_, b_ = rk[k][idx], rR[idx]; a_ = a_ - a_.mean(0); b_ = b_ - b_.mean(0)
            o[k] = float(np.nanmean((a_ * b_).sum(0) / np.sqrt((a_ ** 2).sum(0) * (b_ ** 2).sum(0) + 1e-12)))
        return o
    base = rho(np.arange(nt)); boots = [rho(np.concatenate([blocks[j] for j in RNG.integers(0, len(blocks), len(blocks))])) for _ in range(n_boot)]
    res = {}
    for k in keys:
        bv = np.array([x[k] for x in boots]); res[k] = [round(base[k], 3), round(float(np.percentile(bv, 2.5)), 3), round(float(np.percentile(bv, 97.5)), 3)]
        if k != "risk":
            dv = np.array([x["risk"] - x[k] for x in boots]); res["risk_minus_" + k] = [round(base["risk"] - base[k], 3), round(float(np.percentile(dv, 2.5)), 3)]
    out["spearman"] = res
    json.dump(out, open(f"{HP.DIAG}/e5v2_{data}_s{seed}.json", "w"), indent=1)
    print("wrote", f"{HP.DIAG}/e5v2_{data}_s{seed}.json", flush=True)
    return out


def run_mem_full(m, zt, t0, T1):
    from hgm.models.forecaster import views
    B = zt.shape[0]; M, S_ = m.init_state(B); n = T1 - t0; g = torch.ones(B)
    s = np.zeros((n, B), np.float32); w = np.zeros((n, B), np.float32)
    for i, t in enumerate(range(t0, T1)):
        tok, xn, mu, sd = views(zt, t, m.L, m.P); _, M, S_, info = m.step(tok, xn, None, M, S_, g)
        s[i] = info["s"].numpy(); w[i] = info["w"].numpy()
    return s, w


def target_memory(z, tr, n, H, L=336, d=32, theta=0.1, alpha=1e-3, seed=0):
    B = z.shape[1]; P = np.random.default_rng(seed).standard_normal((d, L)).astype(np.float32) / np.sqrt(L)
    M = np.zeros((B, H, d), np.float32); s = np.full((n, B), np.nan, np.float32); K = {}
    for i in range(n):
        o = tr + i; win = z[o - L:o]; mu, sd = win.mean(0), win.std(0) + 1e-5
        k = ((win - mu) / sd).T @ P.T; k /= np.linalg.norm(k, axis=1, keepdims=True) + 1e-9; K[i] = (k, mu, sd)
        j = i - H
        if j >= 0:
            kj, muj, sdj = K.pop(j); v = ((z[tr + j:tr + j + H] - muj) / sdj).T
            err = np.einsum("bhd,bd->bh", M, kj) - v
            s[i] = np.linalg.norm(err, axis=1)
            M = (1 - alpha) * M - theta * err[:, :, None] * kj[:, None, :]
    return s


def review_v3(data, seed, H, z, tr, A, L, lval, s, ts, D, W, S, R, dX, B, nt):
    import os
    n = A.shape[0]; zt = torch.tensor(z.T)
    m = BenchMem(L=336, H=H); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_mem_linear_s{seed}.pt")); m.eval()
    _, mom = run_mem_full(m, zt, tr, tr + n)
    deep = f"{HP.CKPT}/{data}_H{H}_mem_mlp_s{seed}.pt"; sd_, md_ = None, None
    if os.path.exists(deep):
        md = BenchMem(L=336, H=H, mem="mlp"); md.load_state_dict(torch.load(deep)); md.eval(); sd_, md_ = run_mem_full(md, zt, tr, tr + n)
    sta = target_memory(z, tr, n, H)
    extra = {k: [] for k in ["surprise_week", "surprise_max", "surprise_mom", "target_mem"] + (["deep_surprise", "deep_mom"] if sd_ is not None else [])}
    for t in ts:
        a = t + D
        extra["surprise_week"].append(np.nanmean(s[a - W:a], 0)); extra["surprise_max"].append(np.nanmax(s[a - D:a], 0))
        extra["surprise_mom"].append(np.nanmean(mom[a - D:a], 0)); extra["target_mem"].append(np.nanmean(sta[a - D:a], 0))
        if sd_ is not None:
            extra["deep_surprise"].append(np.nanmean(sd_[a - D:a], 0)); extra["deep_mom"].append(np.nanmean(md_[a - D:a], 0))
    S = dict(S); S.update({k: np.array(v, float) for k, v in extra.items()})
    pos = np.zeros_like(R, bool); neg = np.zeros_like(R, bool); ons = np.zeros_like(R, bool)
    for b in range(B):
        pos[:, b] = R[:, b] >= np.quantile(R[:, b], 0.9)
        neg[:, b] = (R[:, b] <= np.median(R[:, b])) & (dX[:, b] >= np.quantile(dX[:, b], 0.75))
        ons[:, b] = pos[:, b] & ~np.r_[False, pos[:-1, b]]
    ok = [b for b in range(B) if pos[:, b].sum() >= 5 and neg[:, b].sum() >= 5]
    keys = list(S)
    from scipy.stats import rankdata
    U = {k: np.apply_along_axis(lambda x: rankdata(np.nan_to_num(x, nan=-np.inf)) / len(x), 0, S[k]) for k in keys}
    def score(idx):
        o = {k: np.nanmean([auc(S[k][idx, b][pos[idx, b]], S[k][idx, b][neg[idx, b]]) for b in ok]) for k in keys}
        for k in keys:
            u = U[k][idx][:, ok]; o["onset_" + k] = auc(u[ons[idx][:, ok]], u[neg[idx][:, ok]])
        return o
    blk = 21; blocks = [np.arange(i, min(i + blk, nt)) for i in range(0, nt, blk)]
    nb = 2000 if nt * len(ok) < 20000 else 1000
    base = score(np.arange(nt)); boots = [score(np.concatenate([blocks[j] for j in RNG.integers(0, len(blocks), len(blocks))])) for _ in range(nb)]
    out = {"data": data, "seed": seed, "H": H, "block": blk, "n_boot": nb, "variates": len(ok), "n_onsets": int(ons[:, ok].sum()),
           "auroc": {}, "diff_vs_risk": {}, "p_one_sided": {}, "holm": {}, "spearman": {}}
    for k in base:
        bv = np.array([x[k] for x in boots]); out["auroc"][k] = [round(float(base[k]), 3), round(float(np.nanpercentile(bv, 2.5)), 3), round(float(np.nanpercentile(bv, 97.5)), 3)]
    for pre in ("", "onset_"):
        for k in keys:
            if k == "risk": continue
            dv = np.array([x[pre + "risk"] - x[pre + k] for x in boots])
            out["diff_vs_risk"][pre + k] = [round(float(base[pre + "risk"] - base[pre + k]), 3), round(float(np.nanpercentile(dv, 2.5)), 3)]
            out["p_one_sided"][pre + k] = float((1 + np.sum(dv <= 0)) / (1 + len(dv)))
    fam = ["err_day", "surprise", "novelty", "d3a", "cusum", "ph", "adwin", "kswin", "ddm", "hddm"]
    fam = [k for k in fam if k in out["p_one_sided"]]; p = np.array([out["p_one_sided"][k] for k in fam]); order = np.argsort(p)
    adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    out["holm"] = {k: round(float(a_), 4) for k, a_ in zip(fam, adj)}
    rk = {k: np.apply_along_axis(rankdata, 0, np.nan_to_num(S[k], nan=-np.inf)) for k in keys}; rR = np.apply_along_axis(rankdata, 0, R)
    def rho(idx):
        o = {}
        for k in keys:
            a_, b_ = rk[k][idx] - rk[k][idx].mean(0), rR[idx] - rR[idx].mean(0)
            o[k] = float(np.nanmean((a_ * b_).sum(0) / np.sqrt((a_ ** 2).sum(0) * (b_ ** 2).sum(0) + 1e-12)))
        return o
    rb = rho(np.arange(nt)); rbs = [rho(np.concatenate([blocks[j] for j in RNG.integers(0, len(blocks), len(blocks))])) for _ in range(min(nb, 1000))]
    for k in keys:
        bv = np.array([x[k] for x in rbs]); out["spearman"][k] = [round(rb[k], 3), round(float(np.percentile(bv, 2.5)), 3), round(float(np.percentile(bv, 97.5)), 3)]
        if k != "risk":
            dv = np.array([x["risk"] - x[k] for x in rbs]); out["spearman"]["risk_minus_" + k] = [round(rb["risk"] - rb[k], 3), round(float(np.percentile(dv, 2.5)), 3)]
    json.dump(out, open(f"{HP.DIAG}/e5v3_{data}_s{seed}.json", "w"), indent=1)
    print("wrote", f"{HP.DIAG}/e5v3_{data}_s{seed}.json", flush=True)
    return out


def main(data, seed, H=24, gate_dir=HP.POLICY):
    z, tr, va, end = load(data); d = SPD[data]; W = 7 * d; D = d
    f = f"{gate_dir}/{data}_H{H}_s{seed}.npz"
    A = np.load(f)["A"]; n, B, _ = A.shape
    tgt = np.stack([z[t:t + H].T for t in range(tr, tr + n)])
    L = ((A - tgt) ** 2).mean(2)
    nv = va - tr; vL = L[H:nv - H]; lval = vL.mean(0); q90 = np.quantile(vL, 0.9, axis=0)
    mu_v, sd_v = vL.mean(0), vL.std(0) + 1e-12
    rev = np.full((n, B), np.nan); rev[H:] = L[:n - H]
    m = BenchMem(L=336, H=H); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_mem_linear_s{seed}.pt")); m.eval()
    zt = torch.tensor(z.T); _, s = run_mem(m, zt, tr, tr + n)
    C = np.zeros((n, B)); c = np.zeros(B)
    for i in range(n):
        x = np.nan_to_num((rev[i] - mu_v) / sd_v); c = np.maximum(0, c + x - 0.5); C[i] = c
    alarms = {}
    import os
    variants_only = bool(os.environ.get("HGM_LABEL_VARIANTS"))
    makers = {"ph": (lambda: drift.PageHinkley(), False), "adwin": (lambda: drift.ADWIN(), False),
              "kswin": (lambda: drift.KSWIN(seed=0), False), "ddm": (lambda: bdrift.DDM(), True), "hddm": (lambda: bdrift.HDDMA(), True)}
    if variants_only: makers = {}
    for k, (mk, isb) in makers.items():
        al = np.zeros((n, B), bool)
        for b in range(B):
            st = (rev[:, b] > q90[b]).astype(float) if isb else (rev[:, b] - mu_v[b]) / sd_v[b]
            st = np.where(np.isnan(rev[:, b]), np.nan, st)
            al[:, b] = river_alarms(st, mk, n, isb)
        alarms[k] = np.cumsum(al, 0)
    ts = np.arange(nv + W + H, n - D - W - H, d)
    S = {k: [] for k in ["risk", "err_day", "surprise", "novelty", "d3a", "cusum"] + list(makers)}
    R, dX = [], []
    for t in ts:
        a = t + D; rv = a - H
        S["risk"].append(np.log(L[rv - W:rv].mean(0) / lval))
        S["err_day"].append(np.log(L[rv - D:rv].mean(0) / lval))
        S["surprise"].append(np.nanmean(s[a - D:a], 0))
        xp, xw = z[tr + a - D - W:tr + a - D], z[tr + a - D:tr + a]
        S["novelty"].append(np.abs(xw.mean(0) - xp.mean(0)) / (xp.std(0) + 1e-6) + np.abs(np.log((xw.std(0) + 1e-6) / (xp.std(0) + 1e-6))))
        rec, old = L[rv - 4 * D:rv], L[max(H, rv - 4 * D - 4 * W):rv - 4 * D]
        S["d3a"].append((rec.mean(0) - old.mean(0)) / (old.std(0) / np.sqrt(len(rec)) + 1e-12))
        S["cusum"].append(C[a - 1])
        for k in makers:
            S[k].append(alarms[k][a - 1] - alarms[k][a - 1 - W])
        R.append(np.log(L[a:a + W].mean(0) / lval))
        dX.append(np.abs(z[tr + a:tr + a + W].mean(0) - z[tr + t - W:tr + t].mean(0)) / (z[tr + t - W:tr + t].std(0) + 1e-6))
    S = {k: np.array(v, float) for k, v in S.items()}; R = np.array(R); dX = np.array(dX); nt = len(ts)
    if variants_only:
        return label_variants(data, seed, H, S, R, dX, B, nt)
    if os.environ.get("HGM_V3"):
        return review_v3(data, seed, H, z, tr, A, L, lval, s, ts, D, W, S, R, dX, B, nt)
    pos = np.zeros_like(R, bool); neg = np.zeros_like(R, bool)
    for b in range(B):
        pos[:, b] = R[:, b] >= np.quantile(R[:, b], 0.9)
        neg[:, b] = (R[:, b] <= np.median(R[:, b])) & (dX[:, b] >= np.quantile(dX[:, b], 0.75))
    ok = [b for b in range(B) if pos[:, b].sum() >= 5 and neg[:, b].sum() >= 5]

    def score(idx):
        return {k: np.nanmean([auc(v[idx, b][pos[idx, b]], v[idx, b][neg[idx, b]]) for b in ok]) for k, v in S.items()}
    base = score(np.arange(nt))
    blocks = [np.arange(i, min(i + 7, nt)) for i in range(0, nt, 7)]
    boots = []
    for _ in range(2000 if nt * len(ok) < 20000 else 500):
        idx = np.concatenate([blocks[j] for j in RNG.integers(0, len(blocks), len(blocks))]); boots.append(score(idx))
    out = {"data": data, "seed": seed, "n_points": nt, "variates": len(ok), "auroc": {}, "diff_vs_risk": {}}
    for k in S:
        bv = np.array([x[k] for x in boots]); dv = np.array([x["risk"] - x[k] for x in boots])
        out["auroc"][k] = [round(base[k], 3), round(float(np.nanpercentile(bv, 2.5)), 3), round(float(np.nanpercentile(bv, 97.5)), 3)]
        if k != "risk":
            out["diff_vs_risk"][k] = [round(base["risk"] - base[k], 3), round(float(np.nanpercentile(dv, 2.5)), 3)]
    import os
    if os.environ.get("SAVE_SERIES"):
        np.savez_compressed(f"{HP.DIAG}/e5series_{data}_s{seed}.npz", ts=ts, R=R, dX=dX, pos=pos, neg=neg, s=s, L=L, lval=lval,
                            **{"S_" + k: v for k, v in S.items()}, tr=tr, va=va, H=H, d=d)
        return
    json.dump(out, open(f"{HP.DIAG}/e5_{data}_s{seed}.json", "w"))
    print(data, seed, "points", nt, "variates", len(ok))
    for k, v in out["auroc"].items():
        dd = out["diff_vs_risk"].get(k, ["", ""])
        print(f"  {k:9} AUROC {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]   risk-minus {dd[0]} (lower {dd[1]})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
