from hgm import paths as HP
import sys, os, json, numpy as np, torch
from scipy.stats import rankdata
from hgm.data import load
from hgm.models.forecaster import BenchMem
from hgm.monitors.memory_stream import run as run_frozen
from hgm.experiments.harm_detection import auc, SPD
from hgm.experiments.same_model import evaluate
torch.set_grad_enabled(False); torch.set_num_threads(1)
P_TOK, D_RAW = 16, 16


def raw_memory_surprise(z, t0, t1, seed=0):
    g = np.random.default_rng(seed); R1 = g.standard_normal((D_RAW, P_TOK)); R2 = g.standard_normal((D_RAW, P_TOK))
    B = z.shape[1]; M = np.zeros((B, D_RAW, D_RAW)); S = np.zeros_like(M); out = np.full((t1 - t0, B), np.nan)
    for t in range(P_TOK, t1):
        x = z[t - P_TOK + 1:t + 1].T
        k = x @ R1.T; k /= np.linalg.norm(k, axis=1, keepdims=True) + 1e-8; v = x @ R2.T
        err = np.einsum("bij,bj->bi", M, k) - v; G = err[:, :, None] * k[:, None, :]
        if t >= t0: out[t - t0] = np.linalg.norm(G, axis=(1, 2))
        S = 0.5 * S - 0.5 * G; M = 0.95 * M + S
    return out


def change_scores(sig, tgt):
    au, rho = [], []
    for b in range(tgt.shape[1]):
        hi = tgt[:, b] >= np.quantile(tgt[:, b], 0.75); lo = tgt[:, b] <= np.quantile(tgt[:, b], 0.5)
        au.append(auc(sig[hi, b], sig[lo, b]))
        a, c = rankdata(sig[:, b]), rankdata(tgt[:, b]); a -= a.mean(); c -= c.mean()
        rho.append((a * c).sum() / np.sqrt((a ** 2).sum() * (c ** 2).sum() + 1e-12))
    return [round(float(np.nanmean(au)), 3), round(float(np.nanmean(rho)), 3)]


def main(data, seed, H=24):
    out_fn = f"{HP.DIAG}/e1change_{data}_s{seed}.json"
    if os.path.exists(out_fn): print("exists", out_fn); return
    z, tr, va, end = load(data); zt = torch.tensor(z.T); T1 = end - H + 1; n = T1 - tr; nv = va - tr; d = SPD[data]
    W, D = 7 * d, d
    m = BenchMem(L=336, H=H); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_mem_linear_s{seed}.pt")); m.eval()
    Y, s_bench = run_frozen(m, zt, tr, T1)
    s_raw = raw_memory_surprise(z, tr, T1, seed)
    ts = np.arange(nv + W + H, n - D - W - H, d)
    sig = {"bench_surprise": [], "raw_surprise": [], "novelty": []}; recent, future, level, vol = [], [], [], []
    for t in ts:
        a = t + D
        sig["bench_surprise"].append(np.nanmean(s_bench[a - D:a], 0)); sig["raw_surprise"].append(np.nanmean(s_raw[a - D:a], 0))
        xp, xw = z[tr + a - D - W:tr + a - D], z[tr + a - D:tr + a]
        sig["novelty"].append(np.abs(xw.mean(0) - xp.mean(0)) / (xp.std(0) + 1e-6) + np.abs(np.log((xw.std(0) + 1e-6) / (xp.std(0) + 1e-6))))
        recent.append(np.abs(xw.mean(0) - xp.mean(0)) / (xp.std(0) + 1e-6))
        level.append(np.abs(xw.mean(0))); vol.append(xw.std(0))
        future.append(np.abs(z[tr + a:tr + a + W].mean(0) - z[tr + t - W:tr + t].mean(0)) / (z[tr + t - W:tr + t].std(0) + 1e-6))
    sig = {k: np.array(v) for k, v in sig.items()}; recent, future, level, vol = map(np.array, (recent, future, level, vol))
    res = {"data": data, "seed": seed, "H": H, "change": {}, "harm": {}}
    for k, v in sig.items():
        res["change"][k] = {"recent_change": change_scores(v, recent), "future_change": change_scores(v, future),
                            "recent_level": change_scores(v, level), "recent_volatility": change_scores(v, vol)}
    tgt = np.stack([z[t:t + H].T for t in range(tr, T1)])
    big = data in ("ECL", "Traffic"); nb = 300 if big else 1000
    Lf = ((Y - tgt) ** 2).mean(2)
    win = np.lib.stride_tricks.sliding_window_view(z[tr - P_TOK + 1:T1], P_TOK, axis=0)
    s_vol = win.std(-1)
    cand = (("bench_surprise", s_bench), ("raw_surprise", s_raw), ("volatility16", s_vol))
    res["harm"]["mem_frozen"] = {k: evaluate(Lf, s, z, tr, nv, H, d, nb, nb)["block7"] for k, s in cand}
    pol = f"{HP.POLICY}/{data}_H{H}_s{seed}.npz"
    if os.path.exists(pol):
        LA = ((np.load(pol)["A"] - tgt) ** 2).mean(2)
        res["harm"]["A"] = {k: evaluate(LA, s, z, tr, nv, H, d, nb, nb)["block7"] for k, s in cand}
    json.dump(res, open(out_fn, "w"), indent=1); print("wrote", out_fn, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 24)
