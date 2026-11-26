from hgm import paths as HP
import sys
import torch, torch.nn as nn
from hgm.models.memory import MemForecaster


class BenchMem(MemForecaster):
    def __init__(self, L=336, H=24, P=16, d=32, hid=128, mem="linear", alpha_max=0.05, assoc="auto", S=16, head="mlp"):
        super().__init__(n_ch=1, n_u=0, L=L, H=H, P=P, d=d, hid=hid, mem=mem, alpha_max=alpha_max)
        self.assoc, self.S = assoc, S
        if assoc == "pred":
            self.Wv2 = nn.Linear(S, d)
        self.lin = nn.Linear(L, H)
        self.head_kind = head
        if head == "mlp":
            self.head = nn.Sequential(nn.Linear(2 * d, hid), nn.GELU(), nn.Linear(hid, H))
        else:
            self.head = nn.Sequential(nn.Linear(d, H))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def features(self, xn):
        F_ = torch.nn.functional
        h = self.enc(xn[:, -self.P:])
        q = F_.normalize(self.Wq(h), dim=-1)
        if self.assoc == "auto":
            k = F_.normalize(self.Wk(h), dim=-1); v = self.Wv(h)
        else:
            hk = self.enc(xn[:, -self.S - self.P:-self.S])
            k = F_.normalize(self.Wk(hk), dim=-1); v = self.Wv2(xn[:, -self.S:])
        gt = torch.sigmoid(self.gates(h))
        return h, q, k, v, gt

    def predict(self, M, q, h, win, fut_u=None, rho=None):
        r = self.read(M, q)
        if rho is not None:
            r = r * rho[:, None]
        if self.head_kind == "mlp":
            return self.lin(win) + self.head(torch.cat([r, h], -1))
        return self.lin(win) + self.head(r)


def revin(x):
    m = x.mean(1, keepdim=True); s = x.std(1, keepdim=True) + 1e-5
    return (x - m) / s, m, s


def views(zt, t, L, P):
    x = zt[:, t - L:t]
    xn, m, s = revin(x)
    return xn, xn, m, s
