from hgm import paths as HP
import sys, numpy as np, torch
from hgm.data import load
from hgm.models.forecaster import BenchMem, views
from hgm.monitors.labels import auroc
from hgm.monitors.harm import SPD
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
torch.set_grad_enabled(False); torch.set_num_threads(1)


def run(m, zt, t0, T1):
    B = zt.shape[0]; H = m.H; M, S = m.init_state(B); n = T1 - t0
    Y = np.zeros((n, B, H), np.float32); s = np.zeros((n, B), np.float32); g = torch.ones(B)
    for i, t in enumerate(range(t0, T1)):
        tok, xn, mu, sd = views(zt, t, m.L, m.P); yn, M, S, info = m.step(tok, xn, None, M, S, g)
        Y[i] = (yn * sd + mu).numpy(); s[i] = info["s"].numpy()
    return Y, s


def main(data="ETTh1", H=24, seed=1):
    z, tr, va, end = load(data); zt = torch.tensor(z.T); T1 = end - H + 1; d = SPD[data]; W = 7 * d; Dl = d
    m = BenchMem(L=336, H=H); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_mem_linear_s{seed}.pt")); m.eval()
    t0 = 336; Y, s = run(m, zt, t0, T1); n, B, _ = Y.shape
    tgt = np.stack([z[t:t + H].T for t in range(t0, T1)]); L = ((Y - tgt) ** 2).mean(2)
    ref_tr = L[:tr - t0 - H].mean(0); ref_va = L[tr - t0:va - t0 - H].mean(0)
    def feats(a, ref):
        rv = a - H; f = []
        f.append(s[a - Dl:a].mean(0) / (s[max(0, a - W):a].mean(0) + 1e-9))
        for w in (Dl, 4 * Dl, W): f.append(np.log(L[rv - w:rv].mean(0) / ref))
        xp, xw = z[t0 + a - W - Dl:t0 + a - Dl], z[t0 + a - Dl:t0 + a]
        f.append(np.abs(xw.mean(0) - xp.mean(0)) / (xp.std(0) + 1e-6)); f.append(np.abs(np.log((xw.std(0) + 1e-6) / (xp.std(0) + 1e-6))))
        f = np.stack(f, 1)
        inter = np.stack([f[:, 0] * f[:, 3], f[:, 4] * f[:, 3], f[:, 5] * f[:, 3], f[:, 0] * f[:, 1]], 1)
        return np.c_[f, inter]
    Xtr, ytr = [], []
    for a in range(W + Dl + H, tr - t0 - W - H, d // 2):
        Xtr.append(feats(a, ref_tr)); ytr.append(np.log(L[a:a + W].mean(0) / ref_tr))
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    ts = np.arange(va - t0 + W, n - Dl - W - H, d); Xte = []; yte = []; dX = []
    for t in ts:
        a = t + Dl; Xte.append(feats(a, ref_va)); yte.append(np.log(L[a:a + W].mean(0) / ref_va))
        dX.append(np.abs(z[t0 + a:t0 + a + W].mean(0) - z[t0 + t - W:t0 + t].mean(0)) / (z[t0 + t - W:t0 + t].std(0) + 1e-6))
    Xte = np.stack(Xte); yte = np.array(yte); dX = np.array(dX)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    preds = {"lr_w": Xte[:, :, 3], "surprise": Xte[:, :, 0]}
    rg = Ridge(alpha=1.0).fit((Xtr - mu) / sd, ytr); preds["head_ridge"] = rg.predict(((Xte - mu) / sd).reshape(-1, Xtr.shape[1])).reshape(len(ts), B)
    gb = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05).fit(Xtr, ytr)
    preds["head_gbm"] = gb.predict(Xte.reshape(-1, Xtr.shape[1])).reshape(len(ts), B)
    done = np.zeros(len(ts), bool)
    on = np.zeros((len(ts), B)); XA = list((Xtr - mu) / sd); YA = list(ytr); k0 = len(XA)
    for j, t in enumerate(ts):
        a = t + Dl
        for jj in range(j):
            if ts[jj] + Dl + W + H <= a and not done[jj]:
                XA.extend((Xte[jj] - mu) / sd); YA.extend(yte[jj]); done[jj] = True
        w = np.r_[np.ones(k0) * 0.2, np.ones(len(XA) - k0)]
        rr = Ridge(alpha=1.0).fit(np.array(XA), np.array(YA), sample_weight=w); on[j] = rr.predict((Xte[j] - mu) / sd)
    preds["head_online"] = on
    res = {k: [] for k in preds}
    for b in range(B):
        yb = yte[:, b]; pos = yb >= np.percentile(yb, 90); neg = (yb <= np.median(yb)) & (dX[:, b] >= np.percentile(dX[:, b], 75))
        if pos.sum() < 5 or neg.sum() < 5: continue
        for k, v in preds.items(): res[k].append(auroc(v[pos, b], v[neg, b], n_boot=1)[0])
    print(data, seed, "train samples", len(ytr), {k: round(float(np.mean(v)), 3) for k, v in res.items()},
          "ridge coef", np.round(rg.coef_, 2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ETTh1")
