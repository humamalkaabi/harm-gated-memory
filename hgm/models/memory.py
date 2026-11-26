import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gelu_grad(a):
    cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
    pdf = torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    return cdf + a * pdf


class MemForecaster(nn.Module):
    def __init__(self, n_ch=5, n_u=4, L=24, H=24, P=4, d=32, hid=128, alpha_max=0.05, theta_max=1.0,
                 mem="linear", dm=64):
        super().__init__()
        self.n_ch, self.n_u, self.L, self.H, self.P, self.d = n_ch, n_u, L, H, P, d
        self.alpha_max, self.theta_max, self.mem, self.dm = alpha_max, theta_max, mem, dm
        self.rule = os.environ.get("HGM_MEMORY", "titans")
        assert self.rule in ("titans", "gdn", "ttt"), self.rule
        assert self.rule == "titans" or mem == "linear", "gdn / ttt are defined for the linear memory"
        self.enc = nn.Sequential(nn.Linear(P * n_ch, hid), nn.GELU(), nn.Linear(hid, d))
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.gates = nn.Linear(d, 3)
        nn.init.constant_(self.gates.bias, 0.0)
        head_in = d + d + L * n_ch + H * n_u
        self.head = nn.Sequential(nn.Linear(head_in, hid), nn.GELU(), nn.Linear(hid, H))
        if mem == "mlp":
            self.W1_0 = nn.Parameter(torch.randn(dm, d) / math.sqrt(d))
            self.theta_max, self.gclip = 0.3, 1.0

    def init_state(self, B, device="cpu"):
        if self.mem == "linear":
            M = [torch.zeros(B, self.d, self.d, device=device)]
        else:
            M = [self.W1_0.unsqueeze(0).expand(B, -1, -1).clone(), torch.zeros(B, self.d, self.dm, device=device)]
        S = [torch.zeros_like(x) for x in M]
        return M, S

    def read(self, M, x):
        if self.mem == "linear":
            return torch.bmm(M[0], x.unsqueeze(-1)).squeeze(-1)
        a = torch.bmm(M[0], x.unsqueeze(-1)).squeeze(-1)
        return torch.bmm(M[1], F.gelu(a).unsqueeze(-1)).squeeze(-1)

    def assoc_grad(self, M, k, v):
        if self.mem == "linear":
            err = torch.bmm(M[0], k.unsqueeze(-1)).squeeze(-1) - v
            G = [err.unsqueeze(-1) * k.unsqueeze(1)]
        else:
            a = torch.bmm(M[0], k.unsqueeze(-1)).squeeze(-1)
            p = F.gelu(a)
            err = torch.bmm(M[1], p.unsqueeze(-1)).squeeze(-1) - v
            G2 = err.unsqueeze(-1) * p.unsqueeze(1)
            da = torch.bmm(M[1].transpose(1, 2), err.unsqueeze(-1)).squeeze(-1) * _gelu_grad(a)
            G1 = da.unsqueeze(-1) * k.unsqueeze(1)
            G = [G1, G2]
        s = torch.sqrt(sum((g.flatten(1) ** 2).sum(1) for g in G))
        return G, s

    def features(self, ctx_tok):
        h = self.enc(ctx_tok)
        q = F.normalize(self.Wq(h), dim=-1)
        k = F.normalize(self.Wk(h), dim=-1)
        v = self.Wv(h)
        gt = torch.sigmoid(self.gates(h))
        return h, q, k, v, gt

    def predict(self, M, q, h, win, fut_u, rho=None):
        r = self.read(M, q)
        if rho is not None:
            r = r * rho[:, None]
        return self.head(torch.cat([r, h, win, fut_u], -1))

    def surprise(self, ctx_tok, M):
        h, q, k, v, _ = self.features(ctx_tok)
        return self.assoc_grad(M, k, v)[1]

    def step(self, ctx_tok, win, fut_u, M, S, g, fs=None, feats=None, rho=None):
        h, q, k, v, gt = feats if feats is not None else self.features(ctx_tok)
        yhat = self.predict(M, q, h, win, fut_u, rho)
        theta = self.theta_max * gt[:, 0]
        eta = gt[:, 1]
        alpha = self.alpha_max * gt[:, 2]
        if fs is not None:
            alpha = (alpha * fs).clamp(max=1.0)
        G, s = self.assoc_grad(M, k, v)
        if self.rule != "titans":
            M2, S2, info = self._delta_step(M, S, k, v, theta, alpha, g, s)
            info.update(q=q, h=h)
            return yhat, M2, S2, info
        if self.mem == "mlp":
            c = (self.gclip / s.clamp(min=1e-8)).clamp(max=1.0)[:, None, None]
            G = [gi * c for gi in G]
        gb = g[:, None, None]
        S_new, M_new = [], []
        wsq = torch.zeros_like(s); fsq = torch.zeros_like(s)
        for Mi, Si, Gi in zip(M, S, G):
            Sn = gb * (eta[:, None, None] * Si - theta[:, None, None] * Gi)
            fo = gb * alpha[:, None, None] * Mi
            S_new.append(Sn); M_new.append(Mi + Sn - fo)
            wsq = wsq + (Sn.flatten(1) ** 2).sum(1); fsq = fsq + (fo.flatten(1) ** 2).sum(1)
        return yhat, M_new, S_new, dict(s=s, w=wsq.sqrt(), f=fsq.sqrt(), q=q, h=h)

    def _delta_step(self, M, S, k, v, theta, alpha, g, s):
        Mi, gb = M[0], g[:, None, None]
        a = (1.0 - alpha) if self.rule == "gdn" else torch.ones_like(alpha)
        Md = a[:, None, None] * Mi
        err = torch.bmm(Md, k.unsqueeze(-1)).squeeze(-1) - v
        wr = -theta[:, None, None] * err.unsqueeze(-1) * k.unsqueeze(1)
        fo = Mi - Md
        Mn = Mi + gb * (wr - fo)
        info = dict(s=s, w=(gb * wr).flatten(1).norm(dim=1), f=(gb * fo).flatten(1).norm(dim=1))
        return [Mn], S, info


def make_views(z, t, L, P, H, n_u):
    ctx = z[:, t - P + 1:t + 1].flatten(1)
    win = z[:, t - L + 1:t + 1].flatten(1)
    fut = z[:, t + 1:t + 1 + H, :n_u].flatten(1)
    y = z[:, t + 1:t + 1 + H, n_u]
    return ctx, win, fut, y
