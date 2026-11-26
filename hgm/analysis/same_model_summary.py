from hgm import paths as HP
import glob, json, os, numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(HP.DIAG.rstrip("/"))
runs = [json.load(open(f)) for f in sorted(glob.glob(f"{HP.DIAG}/e5same_*.json"))]
lines = []
def p(*a): lines.append(" ".join(str(x) for x in a))
p(f"runs: {len(runs)}  ({', '.join(sorted({r['data'] for r in runs}))})")
refs = ["mem_frozen", "mem_gd", "deep_frozen", "A"]
for ref in refs:
    R = [r for r in runs if ref in r["refs"]]
    if not R: continue
    p(f"\n=== reference: {ref}  ({len(R)} runs; confirmatory = without ETTh1: {sum(r['data'] != 'ETTh1' for r in R)})")
    mse = [(r["data"], r["seed"], r["test_mse"].get(ref), r["test_mse"].get("A")) for r in R]
    rel = [m / a - 1 for _, _, m, a in mse if m is not None and a is not None]
    if ref != "A" and rel: p(f"test MSE relative to policy A: median {np.median(rel)*100:+.1f}%  range {min(rel)*100:+.1f}% to {max(rel)*100:+.1f}%")
    for blk in ("block7", "block21"):
        p(f"-- bootstrap {blk}")
        for k in ["risk", "err_day", "surprise", "novelty", "pr_cum", "pr_win"]:
            au = [r["refs"][ref][blk]["auroc"][k][0] for r in R]
            ub = [r["refs"][ref][blk]["auroc"][k][2] for r in R]
            sp = [r["refs"][ref][blk]["spearman"][k][0] for r in R]
            row = f"  {k:9} AUROC {np.nanmin(au):.2f}-{np.nanmax(au):.2f} (median {np.nanmedian(au):.2f})  Spearman {np.nanmin(sp):+.2f} to {np.nanmax(sp):+.2f}"
            if k != "risk":
                win = [r["refs"][ref][blk]["diff_vs_risk"][k][1] > 0 for r in R]
                winc = [w for w, r in zip(win, R) if r["data"] != "ETTh1"]
                swin = [r["refs"][ref][blk]["spearman"]["risk_minus_" + k][1] > 0 for r in R]
                row += f" | risk > {k}: AUROC {sum(win)}/{len(win)} (confirmatory {sum(winc)}/{len(winc)}), Spearman {sum(swin)}/{len(swin)}"
            if k == "surprise":
                row += f" | upper bound < 0.55: {sum(u < 0.55 for u in ub)}/{len(ub)}"
            p(row)
    ds = sorted({r["data"] for r in R if r["data"] != "ETTh1"})
    for k in ["surprise", "err_day", "pr_cum", "pr_win"]:
        wins = 0
        for dname in ds:
            rr = [r for r in R if r["data"] == dname]
            m_risk = np.mean([r["refs"][ref]["block7"]["auroc"]["risk"][0] for r in rr])
            m_k = np.mean([r["refs"][ref]["block7"]["auroc"][k][0] for r in rr])
            wins += m_risk > m_k
        if ds: p(f"  datasets: risk > {k} on {wins}/{len(ds)} (exact one-sided sign test p = {binomtest(wins, len(ds), 0.5, 'greater').pvalue:.3f})")
txt = "\n".join(lines); print(txt)
open(os.path.join(ROOT, "same_model.txt"), "w").write(txt + "\n")
