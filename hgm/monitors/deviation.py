from hgm import paths as HP
import sys, json, numpy as np, torch
from hgm.data import load
from hgm.models.forecaster import BenchMem, views
from hgm.monitors.labels import labels, auroc
torch.set_grad_enabled(False); torch.set_num_threads(1)


def online_errors(m, zt, tr, T1, write):
    B = zt.shape[0]; H = m.H; M, S = m.init_state(B); buf = {}; n = T1 - tr
    Eh = np.full((n, H, B), np.nan, np.float32); s = np.full((n, B), np.nan, np.float32)
    g = torch.full((B,), 1.0 if write else 0.0)
    for i, t in enumerate(range(tr, T1)):
        tok, xn, mu, sd = views(zt, t, m.L, m.P); row = zt[:, t - 1]
        for h in range(1, H + 1):
            if t - h in buf: Eh[i, h - 1] = ((buf[t - h][:, h - 1] - row) ** 2).numpy()
        yn, M, S, info = m.step(tok, xn, None, M, S, g)
        buf[t] = yn * sd + mu; buf.pop(t - H - 1, None); s[i] = info["s"].numpy()
    return Eh, s


def stdz(Eh, vs, hmax=None):
    mu = np.nanmean(Eh[vs], axis=(0, 2)); sd = np.nanstd(Eh[vs], axis=(0, 2)) + 1e-6
    Z = (Eh - mu[None, :, None]) / sd[None, :, None]
    if hmax: Z = Z[:, :hmax]
    return np.nanmean(Z, axis=1)


def cusum(x, k):
    C = np.zeros_like(x); c = np.zeros(x.shape[1])
    for i in range(len(x)):
        c = np.maximum(0, c + np.nan_to_num(x[i]) - k); C[i] = c
    return C


def main(data="ETTh1", H=24, seed=1):
    z, tr, va, end = load(data); zt = torch.tensor(z.T); T1 = end - H + 1
    ld = lambda k: (lambda m: (m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_{k}_linear_s{seed}.pt")), m.eval(), m)[-1])(BenchMem(L=336, H=H))
    Em, s = online_errors(ld("mem"), zt, tr, T1, True)
    Ep, _ = online_errors(ld("plain"), zt, tr, T1, False)
    vs = slice(H, va - tr)
    e = stdz(Em, vs); ep = stdz(Ep, vs); es = stdz(Em, vs, max(1, H // 4)); x = e - ep
    sig = dict(s=s, e=e, e_short=es, x=x, cus_e=cusum(e, 0.5), cus_x=cusum(x, 0.25))
    out = {}
    for lab, (W, Lr, Hr) in {"ridge96": (672, 96, 24), "ridge48": (1008, 48, 12)}.items():
        ts = np.arange(va + W, end - W - H, 24)
        dC, dX = labels(z, ts, W, Lr, Hr)
        pos = dC >= np.nanpercentile(dC, 90); neg = (dC <= np.nanpercentile(dC, 50)) & (dX >= np.nanpercentile(dX, 75))
        agg = {k: np.stack([np.nanmean(v[t - tr:t - tr + H], 0) for t in ts]) for k, v in sig.items()}
        out[lab] = {k: round(auroc(v[pos], v[neg])[0], 3) for k, v in agg.items()}
        print(lab, int(pos.sum()), int(neg.sum()), out[lab], flush=True)
    json.dump(out, open(f"{HP.DIAG}/dev_{data}_H{H}.json", "w"))


if __name__ == "__main__":
    main()
