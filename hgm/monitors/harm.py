from hgm import paths as HP
import sys, json, numpy as np, torch
from hgm.monitors import deviation as D
from hgm.data import load
from hgm.models.forecaster import BenchMem
from hgm.monitors.regret import online_errors_fs, zs
from hgm.monitors.labels import auroc
torch.set_grad_enabled(False); torch.set_num_threads(1)
SPD = dict(ETTh1=24, ETTh2=24, ETTm1=96, Weather=144, WeatherClean=144, ECL=24)


def lead_avg(E):
    return np.nanmean(E, 1)


def eb_bounds(x, delta=0.05):
    n = x.shape[0]; mx = np.nanmax(x, 0) + 1e-12; u = x / mx
    m = np.nanmean(u, 0); v = np.nanvar(u, 0, ddof=1); L = np.log(2 / delta)
    r = np.sqrt(2 * v * L / n) + 7 * L / (3 * (n - 1))
    return (m - r) * mx, (m + r) * mx


def main(data="ETTh1", H=24, seed=1, out=True):
    z, tr, va, end = load(data); zt = torch.tensor(z.T); T1 = end - H + 1; d = SPD[data]; W = 7 * d; Dl = d
    m = BenchMem(L=336, H=H); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_mem_linear_s{seed}.pt")); m.eval()
    p = BenchMem(L=336, H=H); p.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_plain_linear_s{seed}.pt")); p.eval()
    vs = slice(H, va - tr)
    Eb = online_errors_fs(m, zt, tr, T1, 1.0); Ef = online_errors_fs(m, zt, tr, T1, 5.0)
    Ep, _ = D.online_errors(p, zt, tr, T1, False); _, s = D.online_errors(m, zt, tr, T1, True)
    rb, rf = lead_avg(Eb), lead_avg(Ef)
    eb, ef, ep = D.stdz(Eb, vs), D.stdz(Ef, vs), D.stdz(Ep, vs)
    sig = dict(s=s, e=eb, cus_x=D.cusum(eb - ep, 0.25), reg=D.cusum(zs(eb - ef, vs), 0.25))
    n = rb.shape[0]; B = rb.shape[1]
    fo_b = np.full((n, B), np.nan); fo_f = np.full((n, B), np.nan)
    for o in range(n - H):
        idx = o + np.arange(1, H + 1)
        fo_b[o] = np.nanmean(Eb[idx, np.arange(H)], 0); fo_f[o] = np.nanmean(Ef[idx, np.arange(H)], 0)
    ts = np.arange(va - tr + W, n - Dl - W - H, d)
    rows = {k: [] for k in ["s", "e", "cus_x", "reg", "d3a", "cs", "cs_rel", "nov", "pr_src", "lr_4d", "lr_w", "lr_ew"]}; lab = []; labB = []; labV = []; dX = []
    vr = np.nanmean(fo_b[H:va - tr - H], 0) + 1e-12
    vrb = rb[H:va - tr]; _, hiV = eb_bounds(vrb); mV = np.nanmean(vrb, 0) + 1e-12
    lam = 0.5 ** (1 / (2 * Dl)); lr = np.log(np.nan_to_num(rb, nan=1.0) / mV + 1e-12); EW = np.zeros_like(lr); c = np.zeros(lr.shape[1])
    for i in range(len(lr)): c = lam * c + (1 - lam) * lr[i]; EW[i] = c
    for t in ts:
        a = t + Dl
        for k in ["s", "e", "cus_x", "reg"]: rows[k].append(np.nanmean(sig[k][a - Dl:a], 0))
        rec = rb[a - 4 * Dl:a]; past = rb[max(H, a - 4 * Dl - W * 4):a - 4 * Dl]
        rows["d3a"].append((np.nanmean(rec, 0) - np.nanmean(past, 0)) / (np.nanstd(past, 0) / np.sqrt(len(rec)) + 1e-12))
        lo, _ = eb_bounds(rec); _, hi = eb_bounds(rb[a - 4 * Dl - W:a - 4 * Dl])
        rows["cs"].append(lo - hi); rows["cs_rel"].append(lo / (hi + 1e-12))
        x_past = z[tr + t - W:tr + t]; x_new = z[tr + t:tr + a]
        nv = np.abs(x_new.mean(0) - x_past.mean(0)) / (x_past.std(0) + 1e-6) + np.abs(np.log((x_new.std(0) + 1e-6) / (x_past.std(0) + 1e-6)))
        rows["nov"].append(nv)
        rows["pr_src"].append(lo / hiV); rows["lr_4d"].append(np.log(np.nanmean(rec, 0) / mV))
        rows["lr_w"].append(np.log(np.nanmean(rb[a - W:a], 0) / mV)); rows["lr_ew"].append(EW[a - 1])
        fut = np.nanmean(fo_b[a:a + W], 0); pre = np.nanmean(fo_b[t - W:t], 0)
        lab.append(np.log((fut + 1e-12) / (pre + 1e-12))); labB.append(np.log(np.nanmean(fo_b[a:a + W], 0) / (np.nanmean(fo_f[a:a + W], 0) + 1e-12)))
        labV.append(np.log((fut + 1e-12) / vr))
        dX.append(np.abs(z[tr + a:tr + a + W].mean(0) - z[tr + t - W:tr + t].mean(0)) / (z[tr + t - W:tr + t].std(0) + 1e-6))
    R = {k: np.array(v).ravel() for k, v in rows.items()}; y = np.array(lab).ravel(); yB = np.array(labB).ravel(); dX = np.array(dX).ravel()
    res = {}
    neg = (y <= np.nanmedian(y)) & (dX >= np.nanpercentile(dX, 75))
    for q in (90, 95):
        pos = y >= np.nanpercentile(y, q)
        res[f"harm_top{100 - q}"] = {k: round(auroc(v[pos], v[neg], n_boot=100)[0], 3) for k, v in R.items()}
    yV = np.array(labV).ravel()
    for q in (90, 95):
        pos = yV >= np.nanpercentile(yV, q); negV = (yV <= np.nanmedian(yV)) & (dX >= np.nanpercentile(dX, 75))
        res[f"harmV_top{100 - q}"] = {k: round(auroc(v[pos], v[negV], n_boot=100)[0], 3) for k, v in R.items()}
    posB = yB >= np.nanpercentile(yB, 90); negB = yB <= np.nanmedian(yB)
    res["actR_top10"] = {k: round(auroc(v[posB], v[negB], n_boot=100)[0], 3) for k, v in R.items()}
    print(data, seed, "W", W, "n", len(y), "pos", int((y >= np.nanpercentile(y, 90)).sum()), "neg", int(neg.sum()))
    for kk, vv in res.items(): print(" ", kk, vv, flush=True)
    vid = np.tile(np.arange(B), len(ts))
    def macro(y, negrule, q):
        out = {k: [] for k in R}
        for b in range(B):
            mb = vid == b; yb = y[mb]; db = dX[mb]
            pos = yb >= np.nanpercentile(yb, q)
            neg = (yb <= np.nanmedian(yb)) & (db >= np.nanpercentile(db, 75)) if negrule else yb <= np.nanmedian(yb)
            if pos.sum() < 5 or neg.sum() < 5: continue
            for k, v in R.items(): out[k].append(auroc(v[mb][pos], v[mb][neg], n_boot=1)[0])
        return {k: round(float(np.mean(v)), 3) for k, v in out.items()}
    for nm, yy, nr in (("harm", y, True), ("harmV", yV, True), ("actR", yB, False)):
        res[f"W_{nm}_top10"] = macro(yy, nr, 90); print("  within-var", nm, res[f"W_{nm}_top10"], flush=True)
    print("  pos share per variate (harmV top10):", np.bincount(vid[yV >= np.nanpercentile(yV, 90)], minlength=B))
    if out: json.dump(res, open(f"{HP.DIAG}/harm_{data}_s{seed}.json", "w"))
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ETTh1", seed=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
