from hgm import paths as HP
import glob, json, os, numpy as np
ROOT = os.path.dirname(HP.DIAG.rstrip("/")); L = []
def p(*a): L.append(" ".join(str(x) for x in a))
X = [json.load(open(f)) for f in sorted(glob.glob(f"{HP.DIAG}/h3vol_*_s*_H24.json"))]
p(f"{len(X)} runs")
p(f"{'run':16} " + "  ".join(f"{k:>24}" for k in ("aci", "risk_aci", "vol_only", "vol_aci")))
p(f"{'':16} " + "  ".join(f"{'cov_harm  width  Winkler':>24}" for _ in range(4)))
for x in X:
    r = x["rows"]; p(f"{x['data'] + ' s' + str(x['seed']):16} " + "  ".join(
        f"{r[k]['cover_h3_168']:8.3f} {r[k]['width']:6.3f} {r[k]['winkler']:8.3f}" for k in ("aci", "risk_aci", "vol_only", "vol_aci")))
def cmp(k, Y, tag):
    d = [x["rows"][k]["cover_h3_168"] - x["rows"]["risk_aci"]["cover_h3_168"] for x in Y]
    w = [x["rows"][k]["winkler"] / x["rows"]["risk_aci"]["winkler"] for x in Y]
    wd = [x["rows"][k]["width"] / x["rows"]["risk_aci"]["width"] for x in Y]
    p(f"{k:12} vs risk_aci [{tag}, {len(Y)} runs]: d cov_harm median {np.median(d):+.3f} (higher {sum(v > 0 for v in d)}); "
      f"Winkler ratio median {np.median(w):.3f} (better {sum(v < 1 for v in w)}); width ratio median {np.median(wd):.3f}")
XR = [x for x in X if "vol_aci_rob" in x["rows"]]
if XR:
    p("")
    clean = [x for x in XR if x["data"] not in ("ECL", "Weather")]
    cmp("vol_aci", X, "all"); cmp("vol_aci", [x for x in X if x["data"] not in ("ECL", "Weather")], "excl. ECL, Weather")
    cmp("vol_aci_rob", XR, "all with rob"); cmp("vol_aci_rob", clean, "with rob, excl. ECL, Weather")
    p("\nvol_aci_rob per run: cov_harm width Winkler")
    for x in XR:
        r = x["rows"]["vol_aci_rob"]; p(f"  {x['data'] + ' s' + str(x['seed']):16} {r['cover_h3_168']:.3f} {r['width']:.3f} {r['winkler']:.3f}")
if X:
    d_cov = [x["rows"]["vol_aci"]["cover_h3_168"] - x["rows"]["risk_aci"]["cover_h3_168"] for x in X]
    w_rat = [x["rows"]["vol_aci"]["winkler"] / x["rows"]["risk_aci"]["winkler"] for x in X]
    wd_rat = [x["rows"]["vol_aci"]["width"] / x["rows"]["risk_aci"]["width"] for x in X]
    p(f"\nvol-ACI minus risk-ACI, coverage during harm: median {np.median(d_cov):+.3f}, higher in {sum(v > 0 for v in d_cov)}/{len(X)}")
    p(f"vol-ACI / risk-ACI Winkler: median {np.median(w_rat):.3f}, lower (better) in {sum(v < 1 for v in w_rat)}/{len(X)}")
    p(f"vol-ACI / risk-ACI mean width: median {np.median(wd_rat):.3f}")
txt = "\n".join(L); print(txt); open(os.path.join(ROOT, "e6_report.txt"), "w").write(txt + "\n")
