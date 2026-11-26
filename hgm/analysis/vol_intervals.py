import json, os, sys
import numpy as np
os.environ["UQ_RETURN"] = "1"
from hgm import paths as HP
from hgm.data import load
from hgm.uncertainty import intervals as UQ
from hgm.analysis.review_intervals import aci, A_ALPHA
from hgm.experiments.harm_detection import SPD

P = 16


def main(data, seed, H=24, path=None):
    aR, W, nv, tr = UQ.main(data, H, seed, path=path)
    n, B, Hh = aR.shape; cal = slice(H, nv - H); te = slice(nv + 720, n - 168 - H)
    z = load(data)[0]
    win = np.lib.stride_tricks.sliding_window_view(z[tr - P:tr + n - 1], P, axis=0)[:n]
    vol = win.std(-1)
    floor = 0.1 * np.median(vol[cal], axis=0) + 1e-6
    base = np.repeat(np.maximum(vol, floor)[:, :, None], Hh, axis=2)
    srt = np.sort(aR[cal] / base[cal], 0); nc = len(srt)
    idx = min(int(np.ceil((1 - A_ALPHA) * (nc + 1))) - 1, nc - 1)
    med = np.median(vol[cal], axis=0) + 1e-6
    rob = np.repeat(np.sqrt(vol ** 2 + med ** 2)[:, :, None], Hh, axis=2)
    rows = {"vol_only": base * srt[idx][None], "vol_aci": aci(base, aR, cal, n, H, A_ALPHA),
            "vol_aci_rob": aci(rob, aR, cal, n, H, A_ALPHA),
            "aci": W["split_aci"], "risk_aci": W["risk_aci"]}
    L = (aR ** 2).mean(2)
    fut = np.stack([L[i:i + 168].mean(0) for i in range(n - 168)] + [L[-168:].mean(0)] * 168); harm = np.zeros((n, B), bool)
    for b in range(B):
        th = np.quantile(fut[te, b], 0.9); harm[te, b] = fut[te, b] >= th
    out = {"data": data, "seed": seed, "H": H, "rows": {}}
    for k, w in rows.items():
        cov = aR[te] <= w[te]
        out["rows"][k] = {"cover": round(float(cov.mean()), 4), "cover_h3_168": round(float(cov[harm[te]].mean()), 4),
                          "width": round(float(w[te].mean()), 5),
                          "winkler": round(float((2 * w[te] + (2 / A_ALPHA) * np.clip(aR[te] - w[te], 0, None)).mean()), 5)}
    json.dump(out, open(f"{HP.DIAG}/h3vol_{data}_s{seed}_H{H}.json", "w"), indent=1); print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 24)
