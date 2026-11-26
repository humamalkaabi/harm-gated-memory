from hgm import paths as HP
import argparse, json, os, sys, time, copy
import numpy as np, torch
from hgm.data import load
from hgm.models.forecaster import BenchMem, views
from hgm.models.controllers import Ctrl, Zero, RandomForget, EventMemoryController

torch.set_grad_enabled(False); torch.set_num_threads(1)
FROZEN = dict(K=3, kappa=4.0, nb=4, Rt=16, R=24, recall_at=8, margin=0.1, snap_every=8, max_event=150, arb_margin=0.2, arb_min=6)


class EMC(EventMemoryController):
    def _score(self, model, Mc, hist, zt, t, lo, rho=None):
        B = zt.shape[0]; tot = torch.zeros(B); cnt = torch.zeros(B)
        for j, (tj, q, h, xn, mu, sd) in enumerate(reversed(hist), start=1):
            yc = (model.predict(Mc, q, h, xn, None, rho) * sd + mu)[:, :j]
            tgt = zt[:, tj:tj + j]
            m = (torch.arange(tj, tj + j)[None, :] >= lo[:, None]).float()
            tot = tot + (((yc - tgt) ** 2) * m).sum(1); cnt = cnt + m.sum(1)
        return tot / cnt.clamp(min=1), cnt


class EMCA(EMC):
    def __init__(self, *a, arb_margin=0.0, **k):
        super().__init__(*a, arbitrate=False, **k); self.am = arb_margin; self.rho = None; self.nby = 0; self.nst = 0

    def decide(self, t, s, e):
        g, fs = super().decide(t, s, e)[:2]
        if self.rho is None: self.rho = torch.ones_like(s)
        self.nby += (self.rho < 1).float().sum().item(); self.nst += s.numel()
        return g, fs, self.rho

    def after(self, t, model, M, S, hist, zt):
        M, S = super().after(t, model, M, S, hist, zt)
        if len(hist) >= self.Rt:
            B = zt.shape[0]; lo = torch.full((B,), t - self.Rt + 1)
            sm, _ = self._score(model, M, hist, zt, t, lo, rho=torch.ones(B))
            sb, _ = self._score(model, M, hist, zt, t, lo, rho=torch.zeros(B))
            self.rho = torch.where(sb < sm * (1 - self.am), torch.zeros(B), torch.ones(B))
        return M, S


def run(model, zt, t0, t1, ctrl, write=True, gd_lr=0.0, keep_eps=False, keep_yhat=False):
    model = copy.deepcopy(model) if gd_lr > 0 else model
    B = zt.shape[0]; H = model.H; L = model.L
    M, S = model.init_state(B)
    opt = torch.optim.Adam(model.parameters(), lr=gd_lr) if gd_lr > 0 else None
    buf, gdbuf, hist = {}, {}, []
    from collections import deque
    hist = deque(maxlen=getattr(ctrl, "R", 24))
    n = t1 - t0
    mse = np.full(n, np.nan, np.float32); mae = np.full(n, np.nan, np.float32)
    E = np.full((n, H, B), np.nan, np.float32) if keep_eps else None
    Y = np.full((n, B, H), np.nan, np.float32) if keep_yhat else None
    rec_e = np.zeros((n, B), np.float32); rec_s = np.zeros((n, B), np.float32)
    g1 = torch.full((B,), 1.0 if write else 0.0)
    for i, t in enumerate(range(t0, t1)):
        tok, xn, mu, sd = views(zt, t, L, model.P)
        row = zt[:, t - 1]
        eps = [((buf[t - h][:, h - 1] - row) ** 2) if (t - h) in buf else None for h in range(1, H + 1)]
        if keep_eps:
            for h, e in enumerate(eps):
                if e is not None: E[i, h] = e.numpy()
        feats = model.features(tok)
        s_now = model.assoc_grad(M, feats[2], feats[3])[1]
        et = ctrl.etilde(eps, B)
        dec = ctrl.decide(t, s_now, et)
        g, fs = dec[0] * g1, dec[1]
        rho = dec[2] if len(dec) > 2 else None
        yn, M, S, info = model.step(tok, xn, None, M, S, g, fs, feats=feats, rho=rho)
        yhat = yn * sd + mu
        M, S = ctrl.after(t, model, M, S, hist, zt)
        hist.append((t, feats[1], feats[0], xn, mu, sd))
        buf[t] = yhat; buf.pop(t - H - 1, None)
        if t + H <= zt.shape[1]:
            y = zt[:, t:t + H]
            mse[i] = ((yhat - y) ** 2).mean().item(); mae[i] = (yhat - y).abs().mean().item()
        if keep_yhat: Y[i] = yhat.numpy()
        rec_e[i] = et.numpy(); rec_s[i] = s_now.numpy()
        if opt is not None:
            r = model.read(M, feats[1]).detach()
            gdbuf[t] = (tok, xn, mu, sd, r)
            o = t - H
            if o in gdbuf:
                tk, x_, m_, s_, r_ = gdbuf.pop(o)
                with torch.enable_grad():
                    h_ = model.enc(tk[:, -model.P:])
                    p = (model.lin(x_) + model.head(torch.cat([r_, h_], -1))) * s_ + m_
                    loss = ((p - zt[:, o:o + H]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
    return dict(mse=mse, mae=mae, e=rec_e, s=rec_s, E=E, Y=Y)


def main(data, H, seed, mem="linear", out="results"):
    os.makedirs(out, exist_ok=True)
    fn = f"{out}/{data}_H{H}_{mem}_s{seed}.json"
    if os.path.exists(fn): return
    z, tr, va, end = load(data)
    zt = torch.tensor(z.T)
    def ld(kind):
        m = BenchMem(L=336, H=H, mem=mem); m.load_state_dict(torch.load(f"{HP.CKPT}/{data}_H{H}_{kind}_{mem}_s{seed}.pt")); m.eval(); return m
    mm, mp = ld("mem"), ld("plain")
    T1 = end - H + 1
    res = dict(data=data, H=H, seed=seed, mem=mem, frozen=FROZEN, n_var=int(z.shape[1]), arms={}, ctrl={})
    ti = time.time()
    cal = run(mm, zt, tr, va, Ctrl(None, None), keep_eps=True)
    Ev = cal["E"][H:]
    mu = torch.tensor(np.nanmean(Ev, axis=(0, 2))); sdv = torch.tensor(np.nanstd(Ev, axis=(0, 2)) + 1e-6)
    et = np.nanmean((Ev - mu.numpy()[None, :, None]) / sdv.numpy()[None, :, None], axis=1)
    u = float(np.nanpercentile(et, 95)); s95 = float(np.nanpercentile(cal["s"][H:], 95))
    res.update(u=u, s95=s95)
    lrs = [1e-5, 1e-4, 1e-3]
    vm = {lr: float(np.nanmean(run(mp, zt, tr, va, Zero(None, None), write=False, gd_lr=lr)["mse"])) for lr in lrs}
    lr = min(vm, key=vm.get); res["gd_lr"] = lr; res["gd_val"] = vm
    P = {k: FROZEN[k] for k in ("K", "kappa", "nb", "Rt", "R", "recall_at", "margin", "snap_every", "max_event")}
    test = slice(va - tr, T1 - tr)
    arms = {}
    arms["none"] = run(mp, zt, tr, T1, Zero(None, None), write=False)
    arms["gd"] = run(mp, zt, tr, T1, Zero(None, None), write=False, gd_lr=lr)
    arms["base"] = run(mm, zt, tr, T1, Ctrl(mu, sdv))
    for name, mk in [("F", lambda: EMC(mu, sdv, u, recall=False, arbitrate=False, **P)),
                     ("FR", lambda: EMC(mu, sdv, u, arbitrate=False, **P)),
                     ("FR_s", lambda: EMC(mu, sdv, s95, signal="surprise", arbitrate=False, **P))]:
        c = mk(); arms[name] = run(mm, zt, tr, T1, c)
        res["ctrl"][name] = dict(events=c.n_events, recalls=c.n_recall, boost=c.boost_steps / max(c.steps, 1))
    arms["rnd"] = run(mm, zt, tr, T1, RandomForget(mu, sdv, res["ctrl"]["FR"]["boost"], FROZEN["kappa"], seed=777))
    c = EMC(mu, sdv, u, arbitrate=False, **P); arms["FR_gd"] = run(mm, zt, tr, T1, c, gd_lr=lr)
    val = slice(H, va - tr)
    for k, r in arms.items():
        res["arms"][k] = dict(val_mse=float(np.nanmean(r["mse"][val])), mse=float(np.nanmean(r["mse"][test])), mae=float(np.nanmean(r["mae"][test])),
                              mse_curve=np.nanmean(r["mse"][test].reshape(-1)[: (T1 - va) // 200 * 200].reshape(-1, 200), 1).tolist())
    res["sec"] = time.time() - ti
    json.dump(res, open(fn, "w"))
    print(data, H, seed, "val", {k: round(v["val_mse"], 4) for k, v in res["arms"].items()}, "test", {k: round(v["mse"], 4) for k, v in res["arms"].items()}, f"{res['sec']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--H", type=int); ap.add_argument("--seed", type=int)
    ap.add_argument("--mem", default="linear"); ap.add_argument("--out", default="results")
    a = ap.parse_args(); main(a.data, a.H, a.seed, a.mem, a.out)
