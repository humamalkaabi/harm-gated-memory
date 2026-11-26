from hgm import paths as HP
import gzip, numpy as np, pandas as pd

import os
ROOT = HP.DATA


def load(name):
    if name == "ECL":
        x = np.loadtxt(gzip.open(f"{ROOT}/electricity.txt.gz"), delimiter=",", dtype=np.float64)
    elif name == "Traffic":
        x = np.loadtxt(gzip.open(f"{ROOT}/traffic.txt.gz"), delimiter=",", dtype=np.float64)
    else:
        f = {"ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv", "ETTm1": "ETTm1.csv", "Weather": "weather.csv", "WeatherClean": "weather.csv",
             "ETTm2": "ETTm2.csv"}[name]
        x = pd.read_csv(f"{ROOT}/{f}").drop(columns=["date"]).values.astype(np.float64)
        if name == "WeatherClean":
            bad = x == -9999.0
            for j in np.where(bad.any(0))[0]:
                col = pd.Series(np.where(bad[:, j], np.nan, x[:, j])); x[:, j] = col.ffill().bfill().values
    if name.startswith("ETTh"):
        tr, va, end = 4 * 30 * 24, 5 * 30 * 24, 20 * 30 * 24
    elif name.startswith("ETTm"):
        tr, va, end = 4 * 30 * 24 * 4, 5 * 30 * 24 * 4, 20 * 30 * 24 * 4
    else:
        n = len(x); tr = int(n * 0.2); te = int(n * 0.75); va = n - te; end = n
    x = x[:end]
    mu, sd = x[:tr].mean(0), x[:tr].std(0)
    sd[sd == 0] = 1.0
    z = ((x - mu) / sd).astype(np.float32)
    return z, tr, va, end


if __name__ == "__main__":
    for n in ["ETTh2", "ETTm1", "Weather", "ECL"]:
        z, tr, va, end = load(n)
        print(n, z.shape, tr, va, end, float(np.abs(z[va:]).mean()))
