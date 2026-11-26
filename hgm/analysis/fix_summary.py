from hgm import paths as HP
import glob, json, os, re, numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(HP.DIAG.rstrip("/")); lines = []
def p(*a): lines.append(" ".join(str(x) for x in a))
VARIANTS = [("titans_H24", HP.DIAG)] + [(v, os.path.join(ROOT, "phase4", v, "results")) for v in ("gdn_H24", "ttt_H24", "titans_H48", "titans_H96")]

p("== H2 gate rule (every gate's upper 95% bound of improvement over the static average < 0.5%) and best gate vs best static")
for name, R in VARIANTS:
    pre_dir = os.path.join(R, "precorrection")
    for tag, d in (("pre", pre_dir), ("corrected", R)):
        fs = [f for f in sorted(glob.glob(f"{d}/stats_*_s*.json")) if "H6" in json.load(open(f))]
        if not fs: continue
        ok = 0; ub_max = []
        for f in fs:
            g = json.load(open(f))["H6"]["gates"]; ok += all(v["supported"] for v in g.values())
            ub_max.append(max(v["upper95_pct"] for v in g.values()))
        p(f"  {name:11} {tag:9}: rule met in {ok}/{len(fs)} runs; max upper bound median {np.median(ub_max):.2f}%")
    diffs = []
    for f in glob.glob(f"{R}/stats_*_s*.json"):
        f0 = os.path.join(pre_dir, os.path.basename(f))
        if os.path.exists(f0) and "H6" in json.load(open(f)) and "H6" in json.load(open(f0)):
            a, b = json.load(open(f0))["H6"]["gates"], json.load(open(f))["H6"]["gates"]
            diffs += [abs(a[k]["improvement_pct"] - b[k]["improvement_pct"]) for k in a if k in b]
    if diffs: p(f"  {name:11} largest change of any gate's improvement: {max(diffs):.3f} percentage points (median {np.median(diffs):.4f})")
    big = []
    for f in glob.glob(f"{R}/stats_*_s*.json"):
        f0 = os.path.join(pre_dir, os.path.basename(f))
        if os.path.exists(f0) and "H6" in json.load(open(f)) and "H6" in json.load(open(f0)):
            a_, b_ = json.load(open(f0))["H6"]["gates"], json.load(open(f))["H6"]["gates"]
            big += [(abs(a_[k]["improvement_pct"] - b_[k]["improvement_pct"]), os.path.basename(f)[6:-5], k,
                     a_[k]["improvement_pct"], b_[k]["improvement_pct"]) for k in a_ if k in b_]
    for x in sorted(big, reverse=True)[:3]:
        if x[0] > 0.1: p(f"      {x[1]:16} {x[2]:22} improvement {x[3]:+.2f}% -> {x[4]:+.2f}%")
    S = [json.load(open(f)) for f in glob.glob(f"{R}/stats_*_s*.json")]
    S = [x for x in S if "H6" in x and any("upper975_pct" in v for v in x["H6"]["gates"].values())]
    if S:
        g975 = sum(all(v["supported975"] for v in x["H6"]["gates"].values()) for x in S)
        tost7 = sum(sum(v["tost"] for v in x["H6"]["pred"].values()) for x in S)
        tost21 = sum(sum(v.get("tost_b21", False) for v in x["H6"]["pred"].values()) for x in S)
        npred = sum(len(x["H6"]["pred"]) for x in S)
        p(f"  {name:11} gate rule with 97.5% bounds: {g975}/{len(S)}; TOST equivalence: blocks of 7 {tost7}/{npred}, blocks of 21 {tost21}/{npred}")
    H7 = [json.load(open(f)).get("H7") for f in glob.glob(f"{R}/stats_*_s*.json") if "WeatherClean" not in f]
    H7 = [h for h in H7 if isinstance(h, dict) and "winkler_upper975" in h]
    if H7:
        p(f"  {name:11} H3: coverage rule {sum(h['coverage_ok'] for h in H7)}/{len(H7)} (95%) and {sum(h['coverage_ok975'] for h in H7)}/{len(H7)} (97.5%); "
          f"width rule {sum(h['width_ok'] for h in H7)}/{len(H7)} (95%) and {sum(h['width_ok975'] for h in H7)}/{len(H7)} (97.5%)")

p("\n== Best gate vs best static choice in hindsight (utility_*.txt; MSE gain in % of PROCEED; >0 means a gate wins)")
from hgm.analysis.tables import _utility_rows
for tag, d in (("pre", os.path.join(HP.DIAG, "precorrection")), ("corrected", HP.DIAG)):
    G = []
    for f in sorted(glob.glob(f"{d}/utility_*_s*.txt")):
        v = _utility_rows(f)
        st = [x for k, x in v.items() if k.startswith(("standalone", "avg("))]
        ga = [x for k, x in v.items() if k.startswith(("hedge", "bayes", "lcb"))]
        if ga: G.append((min([0.0] + st) - min(ga), os.path.basename(f)[8:-4]))
    if G:
        g = [x for x, _ in G]; w = max(G)
        p(f"  {tag:9}: {len(G)} runs; median {np.median(g):+.2f}, max {w[0]:+.2f} ({w[1]}), min {min(g):+.2f}; "
          f">0.5 in {sum(x > 0.5 for x in g)}, >1 in {sum(x > 1 for x in g)}")
        p("             per run: " + ", ".join(f"{n} {x:+.2f}" for x, n in G))

p("\n== Deployable static choice (h2review) before / after")
for tag, d in (("pre", os.path.join(HP.DIAG, "precorrection")), ("corrected", HP.DIAG)):
    X = [json.load(open(f)) for f in glob.glob(f"{d}/h2review_*_s*.json")]
    if not X: continue
    rows = [(v["mse_rel_to_P_pct"]["deployable_static"] - v["mse_rel_to_P_pct"]["best_static_hindsight"], x.get("data", "?"), x.get("seed", "?"), k,
             v["mse_rel_to_P_pct"]["deployable_static"]) for x in X for k, v in x["pairs"].items()]
    gap = [r[0] for r in rows]; w = max(rows)
    p(f"  {tag:9}: {len(gap)} pairs; deployable - hindsight median {np.median(gap):.2f} points, max {max(gap):.2f} "
      f"({w[1]} s{w[2]} {w[3]}; deployable choice {w[4]:+.2f} points vs PROCEED); >5 points in {sum(g > 5 for g in gap)} pairs")

p("\n== Width-matched ACI control (h3review, harm periods by MSE) before / after")
for tag, d in (("pre", os.path.join(HP.DIAG, "precorrection")), ("corrected", HP.DIAG)):
    F = sorted(f for f in glob.glob(f"{d}/h3review_*_s*_H24.json") if "WeatherClean" not in f)
    X = [json.load(open(f)) for f in F]
    if not X: continue
    try:
        g = [x["rows"]["risk_aci"]["cover_h3_168"] - x["rows"]["aci_width_matched"]["cover_h3_168"] for x in X]
        p(f"  {tag:9}: {len(g)} runs; risk-ACI covers more during harm than width-matched ACI in {sum(v > 0 for v in g)}; median gain {100*np.median(g):.2f} points")
        p("             per run (points): " + ", ".join(f"{os.path.basename(f)[9:-9]} {100*v:+.2f}" for f, v in zip(F, g)))
    except KeyError as e:
        p(f"  {tag}: key {e} not found in h3review rows: {list(X[0]['rows'].keys())}")

p("\n== E1: what the benchmark memory's surprise registers (AUROC top-quartile vs bottom-half; Spearman)")
E = [json.load(open(f)) for f in sorted(glob.glob(f"{HP.DIAG}/e1change_*_s*.json")) if "WeatherClean" not in f]; p(f"  {len(E)} main runs")
if E:
    for sig in ("bench_surprise", "raw_surprise", "novelty"):
        for tgt in ("recent_change", "future_change", "recent_level", "recent_volatility"):
            a = [e["change"][sig][tgt][0] for e in E]; r = [e["change"][sig][tgt][1] for e in E]
            p(f"  {sig:15} vs {tgt:17}: AUROC {min(a):.2f}-{max(a):.2f} (median {np.median(a):.2f}); Spearman {min(r):+.2f} to {max(r):+.2f}")
    for ref in ("mem_frozen", "A"):
        R_ = [e for e in E if ref in e["harm"]]
        if not R_: continue
        p(f"  -- harm on {ref} ({len(R_)} runs)")
        for k in ("bench_surprise", "raw_surprise", "volatility16"):
            au = [e["harm"][ref][k]["auroc"]["surprise"][0] for e in R_]; sp = [e["harm"][ref][k]["spearman"]["surprise"][0] for e in R_]
            win = [e["harm"][ref][k]["diff_vs_risk"]["surprise"][1] > 0 for e in R_]
            swin = [e["harm"][ref][k]["spearman"]["risk_minus_surprise"][1] > 0 for e in R_]
            ds = sorted({e["data"] for e in R_ if e["data"] != "ETTh1"}); wins = 0
            for dn in ds:
                rr = [e for e in R_ if e["data"] == dn]
                wins += np.mean([e["harm"][ref][k]["auroc"]["risk"][0] for e in rr]) > np.mean([e["harm"][ref][k]["auroc"]["surprise"][0] for e in rr])
            pv = binomtest(wins, len(ds), 0.5, "greater").pvalue if ds else float("nan")
            p(f"     {k:15} AUROC {min(au):.2f}-{max(au):.2f} (median {np.median(au):.2f}); Spearman {min(sp):+.2f} to {max(sp):+.2f}; "
              f"risk > it: AUROC {sum(win)}/{len(win)}, Spearman {sum(swin)}/{len(swin)}, datasets {wins}/{len(ds)} (p={pv:.3f})")
txt = "\n".join(lines); print(txt); open(os.path.join(ROOT, "fix_report.txt"), "w").write(txt + "\n")
