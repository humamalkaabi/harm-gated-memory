from hgm import paths as HP
import sys, os, json, numpy as np, torch
from scipy.stats import rankdata
from hgm.data import load
from hgm.models.forecaster import BenchMem
from hgm.models.controllers import Ctrl, Zero
from hgm.monitors.memory_stream import run as run_frozen
from hgm.online import run as run_online
from hgm.experiments.harm_detection import auc, SPD
torch.set_grad_enabled(False); torch.set_num_threads(1)
RNG = np.random.default_rng(0)
ALPHA = 0.05


def eb_lcb_stream(x, alpha=ALPHA):
    n, B = x.shape; out = np.full((n, B), np.nan)
    i = np.zeros(B); sx = np.zeros(B); ssq = np.zeros(B); mu_prev = np.full(B, 0.5); var_prev = np.full(B, 0.25)
    sl = np.zeros(B); slx = np.zeros(B); spen = np.zeros(B); la = np.log(1 / alpha)
    for t in range(n):
        xt = x[t]; ok = ~np.isnan(xt)
        if ok.any():
            ii = i[ok] + 1
            lam = np.minimum(np.sqrt(2 * la / (var_prev[ok] * ii * np.log1p(ii) + 1e-12)), 0.5)
            v = (xt[ok] - mu_prev[ok]) ** 2
            sl[ok] += lam; slx[ok] += lam * xt[ok]; spen[ok] += v * (-np.log1p(-lam) - lam)
            i[ok] = ii; sx[ok] += xt[ok]
            mu_new = (0.5 + sx[ok]) / (ii + 1)
            ssq[ok] += (xt[ok] - mu_new) ** 2
            mu_prev[ok] = mu_new; var_prev[ok] = (0.25 + ssq[ok]) / (ii + 1)
        seen = i > 0
        out[t, seen] = (slx[seen] - la - spen[seen]) / sl[seen]
    return out


def eb_lcb_window(xw, alpha=ALPHA):
    return eb_lcb_stream(xw, alpha)[-1]


def eb_ucb_fixed(x, alpha=ALPHA):
    m = x.shape[0]; la = np.log(2 / alpha)
    return x.mean(0) + np.sqrt(2 * x.var(0, ddof=1) * la / m) + 7 * la / (3 * (m - 1))


def evaluate(L, s, z, tr, nv, H, d, nb_main, nb_rho):
    n, B = L.shape; W = 7 * d; D = d
    lval = L[H:nv - H].mean(0)
    vL = L[H:nv - H]; c = np.quantile(vL, 0.99, axis=0) + 1e-12
    xb = np.minimum(L / c, 1.0)
    ucb = eb_ucb_fixed(xb[H:nv - H])
    rev = np.full((n, B), np.nan); rev[nv + H:] = xb[nv:n - H]
    lcb_cum = eb_lcb_stream(rev)
    ts = np.arange(nv + W + H, n - D - W - H, d)
    keys = ["risk", "err_day", "surprise", "novelty", "pr_cum", "pr_win"]
    S = {k: [] for k in keys}; R, dX = [], []
    for t in ts:
        a = t + D; rv = a - H
        S["risk"].append(np.log(L[rv - W:rv].mean(0) / lval))
        S["err_day"].append(np.log(L[rv - D:rv].mean(0) / lval))
        S["surprise"].append(np.nanmean(s[a - D:a], 0))
        xp, xw = z[tr + a - D - W:tr + a - D], z[tr + a - D:tr + a]
        S["novelty"].append(np.abs(xw.mean(0) - xp.mean(0)) / (xp.std(0) + 1e-6) + np.abs(np.log((xw.std(0) + 1e-6) / (xp.std(0) + 1e-6))))
        S["pr_cum"].append(lcb_cum[a - 1] - ucb)
        S["pr_win"].append(eb_lcb_window(xb[rv - W:rv]) - ucb)
        R.append(np.log(L[a:a + W].mean(0) / lval))
        dX.append(np.abs(z[tr + a:tr + a + W].mean(0) - z[tr + t - W:tr + t].mean(0)) / (z[tr + t - W:tr + t].std(0) + 1e-6))
    S = {k: np.array(v, float) for k, v in S.items()}; R = np.array(R); dX = np.array(dX); nt = len(ts)
    pos = np.zeros_like(R, bool); neg = np.zeros_like(R, bool)
    for b in range(B):
        pos[:, b] = R[:, b] >= np.quantile(R[:, b], 0.9)
        neg[:, b] = (R[:, b] <= np.median(R[:, b])) & (dX[:, b] >= np.quantile(dX[:, b], 0.75))
    ok = [b for b in range(B) if pos[:, b].sum() >= 5 and neg[:, b].sum() >= 5]

    def score(idx):
        return {k: np.nanmean([auc(S[k][idx, b][pos[idx, b]], S[k][idx, b][neg[idx, b]]) for b in ok]) for k in keys}
    rk = {k: np.apply_along_axis(rankdata, 0, np.nan_to_num(S[k], nan=-np.inf)) for k in keys}
    rR = np.apply_along_axis(rankdata, 0, R)

    def rho(idx):
        o = {}
        for k in keys:
            a_, b_ = rk[k][idx] - rk[k][idx].mean(0), rR[idx] - rR[idx].mean(0)
            o[k] = float(np.nanmean((a_ * b_).sum(0) / np.sqrt((a_ ** 2).sum(0) * (b_ ** 2).sum(0) + 1e-12)))
        return o
    res = {"variates": len(ok), "n_points": nt}
    for blk in (7, 21):
        blocks = [np.arange(i, min(i + blk, nt)) for i in range(0, nt, blk)]
        draw = lambda: np.concatenate([blocks[j] for j in RNG.integers(0, len(blocks), len(blocks))])
        base = score(np.arange(nt)); boots = [score(draw()) for _ in range(nb_main)]
        r = {"auroc": {}, "diff_vs_risk": {}}
        for k in keys:
            bv = np.array([x[k] for x in boots])
            r["auroc"][k] = [round(float(base[k]), 3), round(float(np.nanpercentile(bv, 2.5)), 3), round(float(np.nanpercentile(bv, 97.5)), 3)]
            if k != "risk":
                dv = np.array([x["risk"] - x[k] for x in boots])
                r["diff_vs_risk"][k] = [round(float(base["risk"] - base[k]), 3), round(float(np.nanpercentile(dv, 2.5)), 3)]
        rb = rho(np.arange(nt)); rbs = [rho(draw()) for _ in range(nb_rho)]; r["spearman"] = {}
        for k in keys:
            bv = np.array([x[k] for x in rbs])
            r["spearman"][k] = [round(rb[k], 3), round(float(np.percentile(bv, 2.5)), 3), round(float(np.percentile(bv, 97.5)), 3)]
            if k != "risk":
                dv = np.array([x["risk"] - x[k] for x in rbs])
                r["spearman"]["risk_minus_" + k] = [round(rb["risk"] - rb[k], 3), round(float(np.percentile(dv, 2.5)), 3)]
        res[f"block{blk}"] = r
    return res


def lr_on_validation(m, zt, tr, va, ctrl_cls, write):
    vm = {l: float(np.nanmean(run_online(m, zt, tr, va, ctrl_cls(None, None), write=write, gd_lr=l)["mse"])) for l in (1e-5, 1e-4, 1e-3)}
    return min(vm, key=vm.get), vm


def main(data, seed, H=24):
    out_fn = f"{HP.DIAG}/e5same_{data}_s{seed}.json"
    if os.path.exists(out_fn): print("exists", out_fn); return
    z, tr, va, end = load(data); zt = torch.tensor(z.T); T1 = end - H + 1; n = T1 - tr; nv = va - tr; d = SPD[data]
    tgt = np.stack([z[t:t + H].T for t in range(tr, T1)])
    loss = lambda Y: ((Y - tgt) ** 2).mean(2)
    big = data in ("ECL", "Traffic"); nb_main = 500 if big else 1000; nb_rho = 300 if big else 1000
    out = {"data": data, "seed": seed, "H": H, "alpha": ALPHA, "refs": {}, "test_mse": {}}
    te = slice(nv, n)

    def ld(kind, mem="linear"):
        m = BenchMem(L=336, H=H, mem=mem); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_{kind}_{mem}_s{seed}.pt")); m.eval(); return m
    mm = ld("mem")
    Y, s = run_frozen(mm, zt, tr, T1); Lf = loss(Y)
    out["test_mse"]["mem_frozen"] = float(Lf[te].mean())
    out["refs"]["mem_frozen"] = evaluate(Lf, s, z, tr, nv, H, d, nb_main, nb_rho); print(data, seed, "mem_frozen done", flush=True)
    s_frozen = s; del Y
    lr, vm = lr_on_validation(mm, zt, tr, va, Ctrl, True)
    r = run_online(mm, zt, tr, T1, Ctrl(None, None), write=True, gd_lr=lr, keep_yhat=True)
    Lg = loss(r["Y"]); out["test_mse"]["mem_gd"] = float(Lg[te].mean()); out["lr_mem_gd"] = lr
    out["refs"]["mem_gd"] = evaluate(Lg, r["s"], z, tr, nv, H, d, nb_main, nb_rho); print(data, seed, "mem_gd done", flush=True)
    del r
    if os.path.exists(f"{HP.CKPT}/{data}_H{H}_mem_mlp_s{seed}.pt"):
        md = ld("mem", "mlp"); Y, sd_ = run_frozen(md, zt, tr, T1); Ld = loss(Y)
        out["test_mse"]["deep_frozen"] = float(Ld[te].mean())
        out["refs"]["deep_frozen"] = evaluate(Ld, sd_, z, tr, nv, H, d, nb_main, nb_rho); print(data, seed, "deep_frozen done", flush=True)
        del Y
    pol = f"{HP.POLICY}/{data}_H{H}_s{seed}.npz"
    if os.path.exists(pol):
        A = np.load(pol)["A"]
    else:
        mp = ld("plain"); lrA, _ = lr_on_validation(mp, zt, tr, va, Zero, False)
        A = run_online(mp, zt, tr, T1, Zero(None, None), write=False, gd_lr=lrA, keep_yhat=True)["Y"]
    La = loss(A); out["test_mse"]["A"] = float(La[te].mean())
    out["refs"]["A"] = evaluate(La, s_frozen, z, tr, nv, H, d, nb_main, nb_rho); print(data, seed, "A done", flush=True)
    json.dump(out, open(out_fn, "w"), indent=1); print("wrote", out_fn, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 24)
