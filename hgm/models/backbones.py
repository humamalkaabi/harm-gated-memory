import torch, torch.nn as nn


def _revin(x):
    m = x.mean(-1, keepdim=True); s = x.std(-1, keepdim=True) + 1e-5
    return (x - m) / s, m, s


class PatchTST(nn.Module):
    def __init__(self, L=336, H=24, patch=16, stride=8, d_model=128, n_heads=16, layers=3, d_ff=256, dropout=0.2):
        super().__init__()
        self.L, self.H, self.patch, self.stride = L, H, patch, stride
        self.n_patch = (L - patch) // stride + 2
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.empty(self.n_patch, d_model).uniform_(-0.02, 0.02))
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(self.n_patch * d_model, H)

    def forward(self, x):
        xn, m, s = _revin(x)
        xp = torch.cat([xn, xn[:, -1:].expand(-1, self.stride)], 1)
        p = xp.unfold(1, self.patch, self.stride)
        h = self.encoder(self.drop(self.embed(p) + self.pos))
        return self.head(h.flatten(1)) * s + m


class ITransformer(nn.Module):
    def __init__(self, L=336, H=24, d_model=128, n_heads=8, layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.L, self.H = L, H
        self.embed = nn.Linear(L, d_model)
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers, norm=nn.LayerNorm(d_model))
        self.proj = nn.Linear(d_model, H)

    def forward(self, x):
        single = x.dim() == 2
        if single: x = x[None]
        xn, m, s = _revin(x)
        h = self.encoder(self.drop(self.embed(xn)))
        y = self.proj(h) * s + m
        return y[0] if single else y


def build(name, L=336, H=24):
    return {"patchtst": PatchTST, "itransformer": ITransformer}[name](L=L, H=H)


CHANNEL_INDEPENDENT = {"patchtst": True, "itransformer": False}
