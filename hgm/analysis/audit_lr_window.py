import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hgm.data import load
from hgm.online import run, Zero
from hgm.models.forecaster import BenchMem
from hgm.paths import CKPT
torch.set_num_threads(8)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "results", "audit_lr_window.jsonl")
done = set()
if os.path.exists(out):
    done = {(d["data"], d["H"], d["seed"]) for d in map(json.loads, open(out))}
jobs = sorted({(f.split("_")[0], int(f.split("_")[1][1:]), int(f.split("_s")[-1][:-3]))
               for f in os.listdir(CKPT) if "_plain_linear_" in f}, key=lambda j: (j[1], j[0], j[2]))
for data, H, seed in jobs:
    if (data, H, seed) in done: continue
    z, tr, va, end = load(data); zt = torch.tensor(z.T)
    mp = BenchMem(L=336, H=H); mp.load_state_dict(torch.load(f"{CKPT}/{data}_H{H}_plain_linear_s{seed}.pt")); mp.eval()
    mse = {l: run(mp, zt, tr, va, Zero(None, None), write=False, gd_lr=l)["mse"] for l in (1e-5, 1e-4, 1e-3)}
    ts = np.arange(tr, va); clean = ts + H <= va
    allm = {l: float(np.nanmean(v)) for l, v in mse.items()}
    clm = {l: float(np.nanmean(v[clean])) for l, v in mse.items()}
    rec = dict(data=data, H=H, seed=seed, n_val=int(len(ts)), n_leak=int((~clean).sum()),
               max_target_row_used=int(ts.max() + H - 1), va=int(va),
               lr_current=min(allm, key=allm.get), lr_clean=min(clm, key=clm.get), mse_all=allm, mse_clean=clm)
    rec["same"] = rec["lr_current"] == rec["lr_clean"]
    open(out, "a").write(json.dumps(rec) + "\n"); print(rec["data"], H, seed, rec["lr_current"], rec["lr_clean"], rec["same"], flush=True)
