from hgm import paths as HP
import glob, json, os, numpy as np

ROOT = os.path.join(HP.ROOT, "outputs")
dirs = [os.path.join(ROOT, "sensitivity"), HP.DIAG] + [os.path.join(ROOT, "phase4", v, "results") for v in ("titans_H48", "titans_H96")]
F = {}
for d in dirs:
    for f in glob.glob(os.path.join(d, "sens_*_H*_s*.json")): F.setdefault(os.path.basename(f), f)
S = [json.load(open(f)) for f in sorted(F.values())]
print(f"runs with saved streams: {len(S)}  (by H: " + ", ".join(f"H={h}: {sum(s['H'] == h for s in S)}" for h in (24, 48, 96)) + ")")

wins = [s for s in S if "label_7d" in s["multiscale"]]
best7 = sum(max(("1d", "3d", "7d", "14d", "28d"), key=lambda k: s["multiscale"]["label_7d"][k]) == "7d" for s in wins)
print(f"seven-day risk window best among 1-28 days: {best7}/{len(wins)}")

for h in (24, 48, 96):
    X = [s["pareto"] for s in S if s["H"] == h]
    dom = sum(p["risk^1.0+aci0.001"]["cover_harm"] >= p["risk^0+aci0.005"]["cover_harm"] and
              p["risk^1.0+aci0.001"]["winkler"] <= p["risk^0+aci0.005"]["winkler"] for p in X)
    print(f"H={h}: risk-scaled ACI (step 0.001) dominates ACI (0.005) in {dom}/{len(X)} runs")

for lab, b, r in (("ACI", "risk^0+aci0.005", "risk^1.0+aci0.005"), ("DtACI", "dtaci", "risk+dtaci"), ("PID", "pid0.05", "risk+pid0.05")):
    P = [s["pareto"] for s in S]
    up = sum(p[r]["cover_harm"] > p[b]["cover_harm"] for p in P)
    w = [p[r]["winkler"] / p[b]["winkler"] for p in P]
    print(f"{lab:6}: harm coverage up in {up}/{len(P)}; median gain {np.median([p[r]['cover_harm'] - p[b]['cover_harm'] for p in P]):.3f}; "
          f"median Winkler ratio {np.median(w):.3f}; runs with ratio > 1.5: " + ", ".join(f"{s['data']} H{s['H']} s{s['seed']}" for s, x in zip(S, w) if x > 1.5))

for h in (24, 48, 96):
    g = [v["realised_oracle_vs_best_static_pct"] for s in S if s["H"] == h and "gate_bound" in s for v in s["gate_bound"].values()]
    if g: print(f"H={h}: oracle gate vs best static, median {np.median(g):.2f}%, max {max(g):.2f}% over {len(g)} pairs")
allg = [v["realised_oracle_vs_best_static_pct"] for s in S if "gate_bound" in s for v in s["gate_bound"].values()]
print(f"all: max {max(allg):.2f}% over {len(allg)} (policy, run) pairs")
