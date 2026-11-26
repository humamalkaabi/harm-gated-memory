from hgm import paths as HP
import sys, numpy as np, torch
from hgm.monitors import deviation as D
from hgm.data import load
from hgm.models.forecaster import BenchMem, views
from hgm.monitors.labels import labels, auroc
torch.set_grad_enabled(False); torch.set_num_threads(1)


def online_errors_fs(m, zt, tr, T1, fs_val):
    B = zt.shape[0]; H = m.H; M, S = m.init_state(B); buf = {}; n = T1 - tr
    Eh = np.full((n, H, B), np.nan, np.float32); g = torch.ones(B); fs = torch.full((B,), float(fs_val))
    for i, t in enumerate(range(tr, T1)):
        tok, xn, mu, sd = views(zt, t, m.L, m.P); row = zt[:, t - 1]
        for h in range(1, H + 1):
            if t - h in buf: Eh[i, h - 1] = ((buf[t - h][:, h - 1] - row) ** 2).numpy()
        yn, M, S, _ = m.step(tok, xn, None, M, S, g, fs=fs)
        buf[t] = yn * sd + mu; buf.pop(t - H - 1, None)
    return Eh


def zs(x, vs):
    return (x - np.nanmean(x[vs], 0)) / (np.nanstd(x[vs], 0) + 1e-6)


def ewma(x, hl):
    lam = 0.5 ** (1 / hl); o = np.zeros_like(x); c = np.zeros(x.shape[1])
    for i in range(len(x)):
        c = lam * c + (1 - lam) * np.nan_to_num(x[i]); o[i] = c
    return o


def main(data="ETTh1", H=24, seed=1):
    z, tr, va, end = load(data); zt = torch.tensor(z.T); T1 = end - H + 1
    m = BenchMem(L=336, H=H); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_mem_linear_s{seed}.pt")); m.eval()
    p = BenchMem(L=336, H=H); p.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_plain_linear_s{seed}.pt")); p.eval()
    vs = slice(H, va - tr)
    E = {f: online_errors_fs(m, zt, tr, T1, f) for f in (0.2, 1.0, 5.0, 20.0)}
    Ep, s = D.online_errors(p, zt, tr, T1, False)
    e = {f: D.stdz(v, vs) for f, v in E.items()}; ep = D.stdz(Ep, vs)
    _, s = D.online_errors(m, zt, tr, T1, True)
    sig = dict(s=s, ps=e[1.0], cus_x=D.cusum(e[1.0] - ep, 0.25))
    for f in (5.0, 20.0):
        d = zs(e[0.2] - e[f], vs)
        sig[f"reg{int(f)}"] = d; sig[f"cus_reg{int(f)}"] = D.cusum(d, 0.25); sig[f"ew_reg{int(f)}"] = ewma(d, 24)
        dr = zs(ep - e[f], vs); sig[f"cus_regp{int(f)}"] = D.cusum(dr, 0.25)
    print({f: float(np.nanmean(v[va - tr:])) for f, v in E.items()}, "plain", float(np.nanmean(Ep[va - tr:])))
    for lab, (W, Lr, Hr) in {"ridge96": (672, 96, 24), "ridge48": (1008, 48, 12)}.items():
        ts = np.arange(va + W, end - W - H, 24); dC, dX = labels(z, ts, W, Lr, Hr)
        neg = (dC <= np.nanpercentile(dC, 50)) & (dX >= np.nanpercentile(dX, 75))
        agg = {k: np.stack([np.nanmean(v[t - tr:t - tr + H], 0) for t in ts]) for k, v in sig.items()}
        print(f"\n== {data} labels {lab} ==   AUROC top5% | top10%")
        for k in sig:
            r = [auroc(agg[k][dC >= np.nanpercentile(dC, qq)], agg[k][neg], n_boot=100)[0] for qq in (95, 90)]
            print(f"  {k:11} {r[0]:.3f} | {r[1]:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ETTh1")
