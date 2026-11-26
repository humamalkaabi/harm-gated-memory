import glob, json, os, statistics as st, sys
import numpy as np
from scipy.stats import binomtest
from hgm import paths as HP

DATA = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Traffic"]
TAB = HP.TABLES


def load(prefix, dirs):
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, prefix + "*.json")): out.setdefault(os.path.basename(f), json.load(open(f)))
    return list(out.values())


def h1(dirs):
    X = load("e5v3_", dirs); by = {d: [x for x in X if x["data"] == d] for d in DATA}; by = {d: v for d, v in by.items() if v}
    sig = ["surprise", "surprise_week", "surprise_max", "surprise_mom", "deep_surprise", "deep_mom", "novelty", "target_mem", "err_day"]
    name = dict(surprise="Surprise (1 day)", surprise_week="Surprise (7 days)", surprise_max="Surprise (daily max)",
                surprise_mom="Momentum $\\lVert S_t\\rVert$", deep_surprise="Deep memory: surprise", deep_mom="Deep memory: momentum",
                novelty="Input novelty", target_mem="Target-associative memory", err_day="Persistence (1 day)")
    summ = {"runs": len(X)}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Further post hoc checks of H1 (Titans, $H=24$; pre-specified labels; blocks of 21 decisions). For each signal: the range over datasets of the seed-mean AUROC; the range of the seed-mean within-variate Spearman correlation with future log-risk; the runs whose lower 95\% bound of risk minus the signal exceeds zero; and the confirmatory datasets (ETTh1 excluded) whose seed-mean AUROC is below that of the risk monitor, with an exact one-sided sign test. Onset: the AUROC with positives restricted to the first decision point of each harm episode.}",
         r"\label{tab:review-h1}", r"\small", r"\resizebox{\textwidth}{!}{\begin{tabular}{lccccc}", r"\toprule",
         r"Signal & AUROC & Spearman & Runs: risk $>$ signal & Datasets: risk $>$ signal & Onset AUROC \\", r"\midrule"]
    conf = [d for d in by if d != "ETTh1"]
    rows = ["risk"] + [k for k in sig if all(k in x["auroc"] for x in X)]
    for k in rows:
        a = [st.mean(x["auroc"][k][0] for x in by[d]) for d in by]; r = [st.mean(x["spearman"][k][0] for x in by[d]) for d in by]
        on = [st.mean(x["auroc"]["onset_" + k][0] for x in by[d]) for d in by]
        if k == "risk":
            L.append(f"Risk monitor & {min(a):.2f}--{max(a):.2f} & {min(r):.2f}--{max(r):.2f} & -- & -- & {min(on):.2f}--{max(on):.2f} \\\\"); continue
        runs = sum(x["diff_vs_risk"][k][1] > 0 for x in X)
        dw = sum(st.mean(x["auroc"]["risk"][0] for x in by[d]) > st.mean(x["auroc"][k][0] for x in by[d]) for d in conf)
        p = binomtest(dw, len(conf), 0.5, alternative="greater").pvalue
        summ[k] = dict(auroc=[round(min(a), 3), round(max(a), 3)], spearman=[round(min(r), 3), round(max(r), 3)], runs=runs, datasets=dw, p=round(p, 4),
                       onset=[round(min(on), 3), round(max(on), 3)])
        L.append(f"{name[k]} & {min(a):.2f}--{max(a):.2f} & {min(r):.2f}--{max(r):.2f} & {runs}/{len(X)} & {dw}/{len(conf)} ($p={p:.3f}$) & {min(on):.2f}--{max(on):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    holm = {k: sum(x["holm"].get(k, 1) < 0.05 for x in X) for k in ["surprise", "err_day", "novelty"]}
    onset = {k: sum(x["diff_vs_risk"]["onset_" + k][1] > 0 for x in X) for k in ["surprise", "err_day", "novelty"]}
    summ.update(holm=holm, onset_runs=onset, n_onsets=[x["n_onsets"] for x in X])
    open(os.path.join(TAB, "tab_review_h1.tex"), "w").write("\n".join(L) + "\n")
    return summ


def h2(dirs):
    X = load("h2review_", dirs); P = [(x, k, v) for x in X for k, v in x["pairs"].items()]
    rho = [v["spearman_risk_utility_median"] for _, _, v in P]
    g = lambda key: [v["mse_rel_to_P_pct"][key] - v["mse_rel_to_P_pct"]["best_static_hindsight"] for _, _, v in P]
    trig = g("harm_trigger"); vg = g("v_gate"); dep = g("deployable_static")
    return {"pairs": len(P), "runs": len(X), "rho_median": round(st.median(rho), 3), "rho_range": [min(rho), max(rho)],
            "rho_nonpositive": sum(r <= 0.0 for r in rho),
            "trigger_minus_best_median_pp": round(st.median(trig), 2), "trigger_worse_than_P": sum(v["mse_rel_to_P_pct"]["harm_trigger"] > 0 for _, _, v in P),
            "vgate_minus_best_median_pp": round(st.median(vg), 2), "vgate_beats_best_by_gt_0.5pp": sum(x < -0.5 for x in vg), "vgate_best_gain_pp": round(-min(vg), 2),
            "deployable_minus_best_median_pp": round(st.median(dep), 2), "deployable_max_loss_pp": round(max(dep), 2)}


def h3(dirs):
    X = [x for x in load("h3review_", dirs) if x["data"] != "WeatherClean"]; r = lambda k, m: [x["rows"][k][m] for x in X]
    gain_eq = [a - b for a, b in zip(r("risk_aci", "cover_h3_168"), r("aci_width_matched", "cover_h3_168"))]
    gain_wk = [a - b for a, b in zip(r("risk_aci", "cover_h1_week"), r("aci_width_matched", "cover_h1_week"))]
    wink = [a / b for a, b in zip(r("risk_aci", "winkler"), r("aci_width_matched", "winkler"))]
    sc = [a - b for a, b in zip(r("risk_pid", "cover_h3_168"), r("pid_scorecaster", "cover_h3_168"))]
    out = {"runs": len(X), "equal_width_gain_median": round(st.median(gain_eq), 4), "equal_width_gain_positive": sum(g > 0 for g in gain_eq),
           "equal_width_gain_week_positive": sum(g > 0 for g in gain_wk), "winkler_vs_width_matched_median": round(st.median(wink), 3),
           "winkler_vs_width_matched_below1": sum(w < 1 for w in wink),
           "risk_pid_minus_pid_scorecaster_median": round(st.median(sc), 4), "risk_pid_gt_pid_scorecaster": sum(s > 0 for s in sc)}
    for k in ("split", "risk_only", "bayes", "aci", "risk_aci", "aci_width_matched", "pid", "pid_scorecaster", "risk_pid"):
        out["median_" + k] = {m: round(st.median(r(k, m)), 4) for m in ("cover", "cover_h3_168", "cover_h1_week", "winkler")}
    return out


if __name__ == "__main__":
    dirs = sys.argv[1:] or [HP.DIAG]
    res = {"h1": h1(dirs), "h2": h2(dirs), "h3": h3(dirs)}
    json.dump(res, open(os.path.join(HP.ROOT, "outputs", "review_summary.json"), "w"), indent=1); print(json.dumps(res, indent=1))
