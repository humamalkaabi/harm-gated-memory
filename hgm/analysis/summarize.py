from hgm import paths as HP
import json, glob, numpy as np
rows = {}
for f in sorted(glob.glob(HP.DIAG + "/e5_*_s*.json")):
    r = json.load(open(f)); rows.setdefault(r["data"], []).append(r)
sigs = ["risk", "err_day", "ph", "adwin", "ddm", "hddm", "cusum", "d3a", "novelty", "kswin", "surprise"]
print(f"{'signal':9}" + "".join(f"{d:>16}" for d in rows))
for k in sigs:
    print(f"{k:9}" + "".join(f"{np.mean([r['auroc'][k][0] for r in rs]):8.3f} ({min(r['auroc'][k][0] for r in rs):.2f}-{max(r['auroc'][k][0] for r in rs):.2f})"[:16].rjust(16) for d, rs in rows.items()))
print("\nH5 superiority: worst (over seeds) lower 95% bound of risk minus signal")
pmin = []
for k in sigs[1:]:
    lows = {d: min(r["diff_vs_risk"][k][1] for r in rs) for d, rs in rows.items()}
    print(f"  {k:9}", "  ".join(f"{d} {v:+.3f}" for d, v in lows.items()), "  PASS" if all(v > 0 for v in lows.values()) else "  FAIL")
print("\nH5 surprise non-superiority: upper 95% bound of surprise AUROC < 0.55")
for d, rs in rows.items():
    ups = [r["auroc"]["surprise"][2] for r in rs]; pts = [r["auroc"]["surprise"][0] for r in rs]
    print(f"  {d:8} point {np.round(pts, 3)}  upper {np.round(ups, 3)}", "PASS" if max(ups) < 0.55 else "FAIL")
