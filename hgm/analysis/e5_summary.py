from hgm import paths as HP
import json, os, numpy as np

ROOT = os.path.dirname(HP.DIAG.rstrip("/")); L = []
def p(*a): L.append(" ".join(str(x) for x in a))
def j(f):
    f = os.path.join(HP.DIAG, f); return json.load(open(f)) if os.path.exists(f) else None
p(f"{'':8} {'dataset':13} {'risk':>6} {'pers.':>6} {'surpr.':>6} | {'rho risk':>8} {'rho surpr':>9} | {'cov gain':>8} {'lb':>6} {'Winkler':>7} {'ub':>6} {'ub975':>6} | {'risk-ACI':>8} {'w-matched':>9}")
for s in (1, 2, 3):
    for d in ("Weather", "WeatherClean"):
        e5, v2, st, h3 = j(f"e5_{d}_s{s}.json"), j(f"e5v2_{d}_s{s}.json"), j(f"stats_{d}_s{s}.json"), j(f"h3review_{d}_s{s}_H24.json")
        a = e5["auroc"] if e5 else {}
        rs = v2["spearman"] if v2 else {}
        h7 = st["H7"] if st else {}
        cm = (h3["rows"]["risk_aci"]["cover_h3_168"], h3["rows"]["aci_width_matched"]["cover_h3_168"]) if h3 else (np.nan, np.nan)
        f = lambda x, k=0: f"{x[k]:.3f}" if isinstance(x, list) else (f"{x:.3f}" if isinstance(x, (int, float)) else "  -  ")
        p(f"seed {s:3} {d:13} {f(a.get('risk')):>6} {f(a.get('err_day')):>6} {f(a.get('surprise')):>6} | {f(rs.get('risk')):>8} {f(rs.get('surprise')):>9} | "
          f"{f(h7.get('cov_gain')):>8} {f(h7.get('cov_gain_lower95')):>6} {f(h7.get('winkler_ratio')):>7} {f(h7.get('winkler_upper95')):>6} {f(h7.get('winkler_upper975')):>6} | "
          f"{cm[0]:8.3f} {cm[1]:9.3f}")
p("\nrisk/pers./surpr.: AUROC on the pre-specified labels; rho: label-free Spearman with future log-risk;"
  " cov gain / lb: coverage-during-harm gain of risk-scaled ACI over ACI and its lower 95% bound;"
  " Winkler: ratio to ACI with upper 95% (one-sided) and 97.5% bounds; last two: coverage during harm of risk-scaled ACI"
  " and of the width-matched ACI. Weather (as benchmarked) needs stats/h3review from `fix`.")
txt = "\n".join(L); print(txt); open(os.path.join(ROOT, "e5_report.txt"), "w").write(txt + "\n")
