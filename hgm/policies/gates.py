from hgm import paths as HP
import sys, numpy as np, torch, json
from hgm.data import load
from hgm.models.forecaster import BenchMem
from hgm.online import run, EMC, FROZEN
from hgm.models.controllers import Ctrl, Zero
torch.set_grad_enabled(False); torch.set_num_threads(1)


def experts(data, H, seed, lr=None):
    z, tr, va, end = load(data); zt = torch.tensor(z.T); T1 = end - H + 1
    def ld(k):
        m = BenchMem(L=336, H=H); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_{k}_linear_s{seed}.pt")); m.eval(); return m
    mm, mp = ld("mem"), ld("plain")
    cal = run(mm, zt, tr, va, Ctrl(None, None), keep_eps=True); Ev = cal["E"][H:]
    mu = torch.tensor(np.nanmean(Ev, axis=(0, 2))); sdv = torch.tensor(np.nanstd(Ev, axis=(0, 2)) + 1e-6)
    et = np.nanmean((Ev - mu.numpy()[None, :, None]) / sdv.numpy()[None, :, None], axis=1); u = float(np.nanpercentile(et, 95))
    if lr is None:
        vm = {l: float(np.nanmean(run(mp, zt, tr, va, Zero(None, None), write=False, gd_lr=l)["mse"])) for l in (1e-5, 1e-4, 1e-3)}
        lr = min(vm, key=vm.get)
    P = {k: FROZEN[k] for k in ("K", "kappa", "nb", "Rt", "R", "recall_at", "margin", "snap_every", "max_event")}
    Y = {}
    Y["A"] = run(mp, zt, tr, T1, Zero(None, None), write=False, gd_lr=lr, keep_yhat=True)["Y"]
    Y["B"] = run(mm, zt, tr, T1, EMC(mu, sdv, u, arbitrate=False, **P), gd_lr=lr, keep_yhat=True)["Y"]
    Y["C"] = run(mm, zt, tr, T1, EMC(mu, sdv, u, arbitrate=False, **P), keep_yhat=True)["Y"]
    tgt = np.stack([z[t:t + H].T for t in range(tr, T1)])
    return Y, tgt, tr, va, T1, lr


def gate(Y, tgt, H, names, kind, par, nval=None, clip=None):
    n, B, _ = tgt.shape; K = len(names)
    L = np.stack([((Y[k] - tgt) ** 2).mean(2) for k in names])
    if nval is not None:
        L = L / (L[0, H:nval - H].mean(0)[None, None] + 1e-12)
        if clip: L = np.minimum(L, clip)
    out = np.empty((n, B, H)); w_hist = np.zeros((n, B))
    ew = np.zeros((K, B)); use = np.zeros(B, int); cum = np.zeros((K, B))
    lam = 0.5 ** (1 / par) if kind in ("switch", "safe") else 0.5 ** (1 / 2000)
    for i in range(n):
        o = i - H
        if o >= 0:
            ew = lam * ew + (1 - lam) * L[:, o]; cum = 0.5 ** (1 / 2000) * cum + L[:, o]
        if i <= H:
            wt = np.zeros((K, B)); wt[0] = 1
        elif kind == "switch":
            wt = np.zeros((K, B)); wt[np.argmin(ew, 0), np.arange(B)] = 1
        elif kind == "hedge":
            a = -par * (cum - cum.min(0)); wt = np.exp(a); wt /= wt.sum(0)
        elif kind == "fshare":
            pass
        elif kind == "safe":
            r = ew[1] / (ew[0] + 1e-12)
            use = np.where(use == 1, (r < 1.0).astype(int), (r < 1 - 0.1).astype(int))
            wt = np.zeros((K, B)); wt[0] = 1 - use; wt[1] = use
        if kind == "fshare":
            if i == 0: fw = np.full((K, B), 1.0 / K); fw[:] = 0; fw[0] = 1
            if o >= 0:
                fw = fw * np.exp(-par[0] * np.minimum(L[:, o], 50.0)); fw /= fw.sum(0)
                fw = (1 - par[1]) * fw + par[1] / K
            wt = fw
        out[i] = np.einsum("kb,kbh->bh", wt, np.stack([Y[k][i] for k in names]))
        w_hist[i] = 1 - wt[0]
    return out, w_hist


def main(data="ETTh1", H=24, seed=1):
    Y, tgt, tr, va, T1, lr = experts(data, H, seed)
    te = slice(va - tr, None); r = {}
    for k in Y: r[k] = float(((Y[k][te] - tgt[te]) ** 2).mean())
    for names in (["A", "B"], ["A", "B", "C"], ["A", "C"]):
        for kind, ps in (("switch", (24, 168, 720)), ("hedge", (0.1, 1.0, 10.0)), ("safe", (168, 720))):
            if kind == "safe" and len(names) != 2: continue
            for p in ps:
                o, w = gate(Y, tgt, H, names, kind, p)
                r[f"{''.join(names)}_{kind}{p}"] = (float(((o[te] - tgt[te]) ** 2).mean()), round(float(w[te].mean()), 2))
    print(data, H, seed, "lr", lr); [print(" ", k, v) for k, v in r.items()]
    np.savez_compressed(f"{HP.POLICY}/{data}_H{H}_s{seed}.npz", **{k: v for k, v in Y.items()})
    return r


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ETTh1", int(sys.argv[2]) if len(sys.argv) > 2 else 24)
