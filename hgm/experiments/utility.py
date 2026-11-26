from hgm import paths as HP
import sys, numpy as np
from scipy.stats import norm, spearmanr
from hgm.data import load
from hgm.policies.gates import gate
from hgm.monitors.labels import auroc


def streams(data, H=24, seed=1, gate_dir=None, tag="RL_Proceed"):
    z, tr, va, end = load(data)
    P = np.load(f"{HP.ONLINETSF}/preds/{data}_{H}_{tag}.npz")["pred"].transpose(0, 2, 1)
    f = f"{gate_dir or HP.POLICY}/{data}_H{H}_s{seed}.npz"
    G = dict(np.load(f)); off = va - tr; n = len(P)
    Y = {"P": P, **{k: G[k][off:off + n] for k in ("A", "B", "C")}}
    tgt = np.stack([z[t:t + H].T for t in range(va, va + n)])
    return Y, tgt


def kalman_utility(d_blocks, q, r):
    m, v = 0.0, 1.0 * r; M = np.zeros(len(d_blocks)); V = np.zeros(len(d_blocks))
    for k, y in enumerate(d_blocks):
        v = v + q
        if not np.isnan(y):
            K = v / (v + r); m = m + K * (y - m); v = (1 - K) * v
        M[k], V[k] = m, v
    return M, V


def gates_pair(lP, lX, H, warm, W=168, beta=1.0, thr=0.9):
    n = len(lP); d = lP - lX
    nb = n // H; blk = d[:nb * H].reshape(nb, H).mean(1)
    nbr = max(2, (warm + 1) // H - 1)
    r = np.var(blk[:nbr]) + 1e-12
    out = {}
    for name, q_rel in (("bayes", 0.05),):
        M, V = kalman_utility(blk, q_rel * r, r)
        p = np.zeros(n); mu = np.zeros(n)
        for i in range(n):
            k = (i + 1) // H - 2
            if k >= 0: p[i] = 1 - norm.cdf(0, M[k], np.sqrt(V[k])); mu[i] = M[k]
        out["bayes_sw"] = (p > thr).astype(float); out["bayes_mix"] = p
    lam = 0.5 ** (1 / (W / H)); em = 0.0; ev = r; L = np.zeros(n); kk = -1
    for i in range(n):
        k = (i + 1) // H - 2
        while kk < k:
            kk += 1; x = blk[kk]; em = lam * em + (1 - lam) * x; ev = lam * ev + (1 - lam) * (x - em) ** 2
        neff = (1 + lam) / (1 - lam)
        L[i] = em - beta * np.sqrt(ev / neff) if k >= 0 else -1
    out["lcb"] = (L > 0).astype(float)
    return out


def main(data="ETTh1", H=24, seed=1, gate_dir=None, others=("A", "B", "C")):
    Y, tgt = streams(data, H, seed, gate_dir); n, B, _ = tgt.shape; warm = 720; sc = slice(warm, None)
    Ls = {k: ((v - tgt) ** 2).mean(2) for k, v in Y.items()}
    mse = lambda pred: float(((pred[sc] - tgt[sc]) ** 2).mean())
    ref = mse(Y["P"]); print(f"\n{data} H{H} s{seed}  reference PROCEED MSE {ref:.4f}")
    for X in others:
        print(f"  standalone {X}: {100 * (mse(Y[X]) / ref - 1):+.2f}%")
    for X in others:
        for W in (24, 168):
            pa, fa = [], []
            for i in range(warm + W + H, n - W, 24):
                d = Ls["P"] - Ls[X]
                pa.append(d[i - H - W:i - H].mean(0)); fa.append(d[i:i + W].mean(0))
            pa, fa = np.array(pa), np.array(fa)
            rho = np.nanmean([spearmanr(pa[:, b], fa[:, b])[0] for b in range(B)])
            au = np.nanmean([auroc(pa[fa[:, b] > 0, b], pa[fa[:, b] <= 0, b], n_boot=1)[0] for b in range(B)
                             if 5 < (fa[:, b] > 0).sum() < len(fa) - 5])
            print(f"  predictability P vs {X}, W={W:3}: Spearman {rho:+.3f}  sign-AUROC {au:.3f}  (future share X better {np.mean(fa > 0):.2f})")
    res = {}
    for X in others:
        wX = {k: np.zeros((n, B)) for k in ("bayes_sw", "bayes_mix", "lcb")}
        for b in range(B):
            g = gates_pair(Ls["P"][:, b], Ls[X][:, b], H, warm)
            for k in wX: wX[k][:, b] = g[k]
        res[f"avg(P,{X})"] = mse(0.5 * (Y["P"] + Y[X]))
        o, _ = gate(Y, tgt, H, ["P", X], "hedge", 0.1, nval=warm); res[f"hedge(P,{X})"] = mse(o)
        for k, w in wX.items():
            pred = (1 - w[..., None]) * Y["P"] + w[..., None] * Y[X]; res[f"{k}(P,{X})"] = mse(pred)
            res[f"{k}(P,{X})_share"] = float(w[sc].mean())
    o, _ = gate(Y, tgt, H, ["P", "A", "B", "C"], "hedge", 0.1, nval=warm); res["hedge(all)"] = mse(o)
    res["avg(P,A,B)"] = mse((Y["P"] + Y["A"] + Y["B"]) / 3)
    for k, v in res.items():
        if k.endswith("_share"): continue
        print(f"  {k:18} {100 * (v / ref - 1):+.2f}%   (deployed share of X {res.get(k + '_share', float('nan')):.2f})")
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ETTh1")
