from hgm import paths as HP
import sys, json, numpy as np
from hgm.policies.gates import experts, gate
data, H, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fn = f"{HP.POLICY}/{data}_H{H}_s{seed}.json"
import os; os.makedirs(HP.POLICY, exist_ok=True)
if os.path.exists(fn.replace(".json", ".npz")): sys.exit()
Y, tgt, tr, va, T1, lr = experts(data, H, seed)
te = slice(va - tr, None); r = {k: float(((Y[k][te] - tgt[te]) ** 2).mean()) for k in Y}
o, w = gate(Y, tgt, H, ["A", "B", "C"], "hedge", 0.1)
r["RG_raw"] = float(((o[te] - tgt[te]) ** 2).mean())
o, w = gate(Y, tgt, H, ["A", "B", "C"], "hedge", 0.1, nval=va - tr)
np.savez_compressed(fn.replace(".json", ".npz"), **Y)
r["RG"] = float(((o[te] - tgt[te]) ** 2).mean()); r["mem_weight"] = float(w[te].mean()); r["lr"] = lr
json.dump(r, open(fn, "w")); print(data, H, seed, r, flush=True)
