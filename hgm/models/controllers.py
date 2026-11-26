from collections import deque
import numpy as np
import torch
from hgm.models.memory import make_views

L, H, P, D_U = 24, 24, 4, 4


class Ctrl:
    def __init__(self, mu=None, sd=None):
        self.mu, self.sd = mu, sd

    def etilde(self, eps, B):
        vals = [((e - self.mu[h]) / self.sd[h]) if self.mu is not None else e for h, e in enumerate(eps) if e is not None]
        return torch.stack(vals).mean(0) if vals else torch.zeros(B)

    def decide(self, t, s, e):
        return torch.ones_like(s), None

    def after(self, t, model, M, S, hist, zt):
        return M, S


class Zero(Ctrl):
    def decide(self, t, s, e): return torch.zeros_like(s), None


class Persist(Ctrl):
    def __init__(self, mu, sd, thr, K=3, kappa=4.0, signal="error"):
        super().__init__(mu, sd); self.thr, self.K, self.kappa, self.signal = thr, K, kappa, signal
        self.c = None; self.boost_steps = 0; self.steps = 0

    def _count(self, s, e):
        trig = ((e if self.signal == "error" else s) > self.thr).float()
        self.c = trig if self.c is None else (self.c + 1) * trig
        return self.c

    def decide(self, t, s, e):
        c = self._count(s, e)
        on = c >= self.K
        self.boost_steps += on.float().sum().item(); self.steps += on.numel()
        return torch.ones_like(s), torch.where(on, torch.full_like(s, self.kappa), torch.ones_like(s))


class RandomForget(Ctrl):
    def __init__(self, mu, sd, p, kappa=4.0, seed=0):
        super().__init__(mu, sd); self.p, self.kappa = p, kappa; self.g = torch.Generator().manual_seed(seed)

    def decide(self, t, s, e):
        on = torch.rand(s.shape, generator=self.g) < self.p
        return torch.ones_like(s), torch.where(on, torch.full_like(s, self.kappa), torch.ones_like(s))


class RecallOrForget(Persist):
    def __init__(self, mu, sd, thr, K=3, kappa=4.0, nb=4, R=12, Rt=6, margin=0.1, snap_every=8, forget=True, exclude_new=False, recall_at=None):
        super().__init__(mu, sd, thr, K, kappa, "error")
        self.nb, self.R, self.Rt, self.margin, self.snap_every, self.forget = nb, R, Rt, margin, snap_every, forget
        self.snaps = deque(maxlen=4); self.bank = None; self.valid = None; self.ptr = None
        self.recalled = None; self.n_recall = 0; self.n_events = 0; self.exclude_new = exclude_new; self.recall_at = recall_at or K

    def decide(self, t, s, e):
        c = self._count(s, e)
        if self.recalled is None: self.recalled = torch.zeros_like(s, dtype=torch.bool)
        self.recalled &= c > 0
        self.fire = c == self.K
        self.decide_now = c == self.recall_at
        on = (c >= self.K) & ~self.recalled & torch.tensor(self.forget)
        self.boost_steps += on.float().sum().item(); self.steps += on.numel()
        return torch.ones_like(s), torch.where(on, torch.full_like(s, self.kappa), torch.ones_like(s))

    def _score(self, model, Mc, hist, zt, t):
        tot, cnt = 0.0, 0
        lo = t - self.Rt + 1
        for j, (tj, q, h, win, fut) in enumerate(reversed(hist), start=1):
            a = max(1, lo - tj)
            if a > j: continue
            yc = model.predict(Mc, q, h, win, fut)[:, a - 1:j]
            tgt = zt[:, tj + a:tj + 1 + j, D_U]
            tot = tot + ((yc - tgt) ** 2).sum(1); cnt += j - a + 1
        return tot / max(cnt, 1)

    def after(self, t, model, M, S, hist, zt):
        B = M[0].shape[0]
        if t % self.snap_every == 0:
            self.snaps.append([x.clone() for x in M])
        if self.bank is None:
            self.bank = [torch.zeros((B, self.nb) + x.shape[1:]) for x in M]
            self.valid = torch.zeros(B, self.nb, dtype=torch.bool); self.ptr = torch.zeros(B, dtype=torch.long)
        if self.fire.any() and len(self.snaps) > 0:
            self.n_events += int(self.fire.sum())
            idx = self.fire.nonzero().squeeze(-1)
            pre = self.snaps[0]
            if not hasattr(self, "last_slot"): self.last_slot = torch.full((B,), -1, dtype=torch.long)
            self.last_slot[idx] = self.ptr[idx]
            for bi, x in zip(self.bank, pre):
                bi[idx, self.ptr[idx]] = x[idx]
            self.valid[idx, self.ptr[idx]] = True
            self.ptr[idx] = (self.ptr[idx] + 1) % self.nb
        fire = self.decide_now
        if not fire.any() or len(hist) < self.R:
            return M, S
        new_slot = getattr(self, "last_slot", torch.full((B,), -1, dtype=torch.long))
        cur = self._score(model, M, hist, zt, t)
        best = cur.clone(); best_k = torch.full((B,), -1, dtype=torch.long)
        for k in range(self.nb):
            Mc = [bi[:, k] for bi in self.bank]
            sc = self._score(model, Mc, hist, zt, t)
            ok = self.valid[:, k]
            if self.exclude_new:
                ok = ok & ~(fire & (new_slot == k))
            sc = torch.where(ok, sc, torch.full_like(sc, float("inf")))
            better = sc < best
            best = torch.where(better, sc, best); best_k = torch.where(better, torch.full_like(best_k, k), best_k)
        take = fire & (best_k >= 0) & (best < cur * (1 - self.margin))
        if take.any():
            ti = take.nonzero().squeeze(-1)
            for mi, si, bi in zip(M, S, self.bank):
                mi[ti] = bi[ti, best_k[ti]]
                si[ti] = 0.0
            self.recalled |= take
            self.n_recall += int(take.sum())
        return M, S


def run(model, z, ctrl, freeze_from=None):
    B, T, _ = z.shape
    zt = torch.tensor(z)
    M, S = model.init_state(B)
    err = np.full((B, T), np.nan, np.float32)
    rec = {k: np.full((B, T), np.nan, np.float32) for k in ("s", "w", "f", "g", "fs", "e", "rho")}
    buf = {}
    hist = deque(maxlen=getattr(ctrl, "R", 12))
    for t in range(L - 1, T - H):
        ctx, win, fut, y = make_views(zt, t, L, P, H, D_U)
        c_t = zt[:, t, D_U]
        eps = [((buf[t - h][:, h - 1] - c_t) ** 2) if (t - h) in buf else None for h in range(1, H + 1)]
        feats = model.features(ctx)
        s_now = model.assoc_grad(M, feats[2], feats[3])[1]
        et = ctrl.etilde(eps, B)
        dec = ctrl.decide(t, s_now, et)
        g, fs = dec[0], dec[1]
        rho = dec[2] if len(dec) > 2 else None
        if freeze_from is not None and t >= freeze_from:
            g = torch.zeros_like(g)
        yhat, M, S, info = model.step(ctx, win, fut, M, S, g, fs, feats=feats, rho=rho)
        M, S = ctrl.after(t, model, M, S, hist, zt)
        hist.append((t, feats[1], feats[0], win, fut))
        buf[t] = yhat
        buf.pop(t - H - 1, None)
        err[:, t] = ((yhat - y) ** 2).mean(1).numpy()
        for kk in ("s", "w", "f"):
            rec[kk][:, t] = info[kk].numpy()
        rec["g"][:, t] = g.numpy()
        rec["fs"][:, t] = (fs if fs is not None else torch.ones_like(g)).numpy()
        rec["e"][:, t] = et.numpy()
        rec["rho"][:, t] = (rho if rho is not None else torch.ones_like(g)).numpy()
    rec["err"] = err
    return rec


class EventMemoryController(Ctrl):
    def __init__(self, mu, sd, thr, K=3, kappa=4.0, forget=True, recall=True, arbitrate=True,
                 nb=4, Rt=16, recall_at=8, margin=0.1, snap_every=8, max_event=150, R=24, signal="error", arb_margin=0.0, arb_min=2):
        super().__init__(mu, sd)
        self.signal = signal; self.arb_margin, self.arb_min = arb_margin, arb_min
        self.thr, self.K, self.kappa, self.forget, self.recall, self.arbitrate = thr, K, kappa, forget, recall, arbitrate
        self.nb, self.Rt, self.recall_at, self.margin, self.snap_every, self.max_event, self.R = nb, Rt, recall_at, margin, snap_every, max_event, R
        self.c = None; self.snaps = deque(maxlen=4); self.bank = None
        self.n_events = 0; self.n_recall = 0; self.bypass_steps = 0; self.steps = 0; self.boost_steps = 0

    def _init(self, s):
        B = s.shape[0]
        self.c = torch.zeros(B); self.mode = torch.zeros(B, dtype=torch.bool); self.age = torch.zeros(B, dtype=torch.long)
        self.t_on = torch.zeros(B, dtype=torch.long); self.recalled = torch.zeros(B, dtype=torch.bool)
        self.rho_next = torch.ones(B); self.last_slot = torch.full((B,), -1, dtype=torch.long)

    def decide(self, t, s, e):
        if self.c is None: self._init(s)
        if self.signal == "surprise":
            trig = (s > self.thr).float()
        else:
            trig = (e > self.thr).float() if e.numel() == s.numel() else torch.zeros_like(s)
        self.c = (self.c + 1) * trig
        self.enter = (~self.mode) & (self.c >= self.K)
        self.t_on = torch.where(self.enter, torch.full_like(self.t_on, t) - self.c.long() + 1, self.t_on)
        self.mode = self.mode | self.enter
        self.age = torch.where(self.enter, torch.zeros_like(self.age), self.age + self.mode.long())
        self.decide_recall = self.mode & (self.age == max(self.recall_at - self.K, 1))
        on = self.mode & ~self.recalled & torch.tensor(self.forget)
        fs = torch.where(on, torch.full_like(s, self.kappa), torch.ones_like(s))
        rho = torch.where(self.mode, self.rho_next, torch.ones_like(s)) if self.arbitrate else None
        self.boost_steps += on.float().sum().item(); self.steps += s.numel()
        if rho is not None: self.bypass_steps += (rho < 1).float().sum().item()
        return torch.ones_like(s), fs, rho

    def _score(self, model, Mc, hist, zt, t, lo, rho=None):
        B = zt.shape[0]
        tot = torch.zeros(B); cnt = torch.zeros(B)
        for j, (tj, q, h, win, fut) in enumerate(reversed(hist), start=1):
            yc = model.predict(Mc, q, h, win, fut, rho)[:, :j]
            tgt = zt[:, tj + 1:tj + 1 + j, D_U]
            times = torch.arange(tj + 1, tj + 1 + j)[None, :]
            m = (times >= lo[:, None]).float()
            tot = tot + (((yc - tgt) ** 2) * m).sum(1); cnt = cnt + m.sum(1)
        return tot / cnt.clamp(min=1), cnt

    def after(self, t, model, M, S, hist, zt):
        B = M[0].shape[0]
        if t % self.snap_every == 0:
            self.snaps.append([x.clone() for x in M])
        if self.bank is None:
            self.bank = [torch.zeros((B, self.nb) + x.shape[1:]) for x in M]
            self.valid = torch.zeros(B, self.nb, dtype=torch.bool); self.ptr = torch.zeros(B, dtype=torch.long)
        if self.enter.any():
            self.n_events += int(self.enter.sum())
            if self.recall and len(self.snaps) > 0:
                idx = self.enter.nonzero().squeeze(-1)
                pre = self.snaps[0]
                self.last_slot[idx] = self.ptr[idx]
                for bi, x in zip(self.bank, pre):
                    bi[idx, self.ptr[idx]] = x[idx]
                self.valid[idx, self.ptr[idx]] = True
                self.ptr[idx] = (self.ptr[idx] + 1) % self.nb
        lo = torch.maximum(self.t_on, torch.full_like(self.t_on, t - self.Rt + 1))
        if self.recall and self.decide_recall.any() and len(hist) > 0:
            fire = self.decide_recall
            cur, _ = self._score(model, M, hist, zt, t, lo)
            best = cur.clone(); best_k = torch.full((B,), -1, dtype=torch.long)
            for k in range(self.nb):
                sc, _ = self._score(model, [bi[:, k] for bi in self.bank], hist, zt, t, lo)
                ok = self.valid[:, k] & ~(self.last_slot == k)
                sc = torch.where(ok, sc, torch.full_like(sc, float("inf")))
                better = sc < best
                best = torch.where(better, sc, best); best_k = torch.where(better, torch.full_like(best_k, k), best_k)
            take = fire & (best_k >= 0) & (best < cur * (1 - self.margin))
            if take.any():
                ti = take.nonzero().squeeze(-1)
                for mi, si, bi in zip(M, S, self.bank):
                    mi[ti] = bi[ti, best_k[ti]]; si[ti] = 0.0
                self.recalled |= take; self.n_recall += int(take.sum())
        if self.mode.any():
            if self.arbitrate:
                sm, n = self._score(model, M, hist, zt, t, lo)
                sb, _ = self._score(model, M, hist, zt, t, lo, rho=torch.zeros(B))
                enough = n >= self.arb_min
                resume = self.mode & enough & (sm <= sb)
                bypass = self.mode & enough & (sb < sm * (1 - self.arb_margin))
                self.rho_next = torch.where(bypass, torch.zeros(B), torch.ones(B))
                end = resume
            else:
                end = self.mode & (self.c == 0)
            end = end | (self.mode & (self.age >= self.max_event))
            self.mode = self.mode & ~end
            self.recalled = self.recalled & self.mode
            self.rho_next = torch.where(self.mode, self.rho_next, torch.ones(B))
        return M, S
