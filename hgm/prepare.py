import glob, gzip, os, shutil, subprocess, time, urllib.request
import numpy as np, pandas as pd
from hgm import paths as HP

ETT = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{}.csv"
LAI = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/{0}/{0}.txt.gz"
ONLINETSF_URL, ONLINETSF_COMMIT = "https://github.com/SJTU-DMTai/OnlineTSF", "0a86461"
TCPD_URL = "https://github.com/alan-turing-institute/TCPD"


def _download(url, out, tries=8):
    if os.path.exists(out) and os.path.getsize(out) > 0: return
    part = out + ".part"
    for k in range(tries):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(part, out); print("downloaded", os.path.basename(out)); return
        except Exception as e:
            print(f"  retry {k + 1}/{tries} {os.path.basename(out)}: {e}"); time.sleep(5)
    raise RuntimeError(f"download failed: {url}")


def data():
    for f in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"): _download(ETT.format(f), f"{HP.DATA}/{f}.csv")
    for f in ("electricity", "traffic"): _download(LAI.format(f), f"{HP.DATA}/{f}.txt.gz")
    assert os.path.exists(f"{HP.DATA}/weather.csv"), "data/raw/weather.csv is part of the repository"


def _git(*a, cwd=None):
    subprocess.run(["git", *a], cwd=cwd, check=True)


def onlinetsf():
    T = HP.ONLINETSF
    if not os.path.isdir(f"{T}/.git"):
        if os.path.isdir(T): shutil.rmtree(T)
        _git("clone", "-q", ONLINETSF_URL, T)
        _git("checkout", "-q", ONLINETSF_COMMIT, cwd=T); _git("checkout", "-q", "-b", "change-is-not-harm", cwd=T)
        _git("-c", "user.name=hgm", "-c", "user.email=hgm@local", "am", "-q",
             *sorted(glob.glob(f"{HP.THIRD}/onlinetsf_patches/*.patch")), cwd=T)
    os.makedirs(f"{T}/dataset", exist_ok=True); os.makedirs(f"{T}/preds", exist_ok=True); os.makedirs(f"{T}/logs/online", exist_ok=True)
    for f in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"):
        if not os.path.exists(f"{T}/dataset/{f}.csv"): shutil.copy(f"{HP.DATA}/{f}.csv", f"{T}/dataset/{f}.csv")
    for name, src, start, freq in (("electricity", "electricity", "2016-07-01", "h"), ("traffic", "traffic", "2016-07-01", "h")):
        out = f"{T}/dataset/{name}.csv"
        if os.path.exists(out): continue
        x = np.loadtxt(gzip.open(f"{HP.DATA}/{src}.txt.gz"), delimiter=",")
        df = pd.DataFrame(x, columns=[str(i) for i in range(x.shape[1] - 1)] + ["OT"])
        df.insert(0, "date", pd.date_range(start, periods=len(df), freq=freq).strftime("%Y-%m-%d %H:%M:%S"))
        df.to_csv(out, index=False); print("wrote", out, df.shape)


def tcpd():
    if not os.path.isdir(f"{HP.TCPD}/.git"):
        if os.path.isdir(HP.TCPD): shutil.rmtree(HP.TCPD)
        _git("clone", "-q", "--depth", "1", TCPD_URL, HP.TCPD)


def check():
    import importlib.util as u
    miss = [m for m in ("torch", "numpy", "pandas", "scipy", "sklearn", "matplotlib", "tqdm", "river", "einops", "tensorboardX", "transformers")
            if u.find_spec(m) is None]
    if miss: raise SystemExit(f"missing packages: {' '.join(miss)}   ->   pip install -r requirements.txt")


def all_():
    check(); data(); onlinetsf(); tcpd()
    import torch; print("torch", torch.__version__, "| cores", os.cpu_count(), "\nSETUP_OK")
