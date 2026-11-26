from hgm import paths as HP
import glob, json, os, numpy as np

TAB = HP.TABLES
SETTINGS = [("main", "Titans, $H{=}24$"), ("phase4/titans_H48", "Titans, $H{=}48$"), ("phase4/titans_H96", "Titans, $H{=}96$"),
            ("robust/patchtst", "PatchTST, $H{=}24$"), ("robust/itransformer", "iTransformer, $H{=}24$")]
R = [r for r in json.load(open(os.path.join(HP.ROOT, "outputs", "label_variants_all.json"))) if r["data"] != "WeatherClean"]
med = lambda v: float(np.median(v))


def labels():
    L = [r"""\begin{table}[t]
\centering
\caption{Sensitivity of H1 to the definition of harm (post hoc). \emph{Spearman}: label-free, within-variate rank correlation between each signal and the future log-risk $R$. \emph{AUROC}: positives are the top 10\% of $R$ within each variate; negatives are the pre-specified set (low harm, high input change), low harm with low input change, or every non-positive point. Entries are medians over runs (surprise: range); the last two columns count runs whose 95\% block-bootstrap lower bound of the difference exceeds zero (500 resamples, so counts can differ by up to two runs from Table~\ref{tab:h1}, which uses 2000).}
\label{tab:labels}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrr}
\toprule
Setting & Measure & Runs & Risk & Persistence & Surprise & Risk$>$surprise & Risk$>$persistence \\"""]
    for src, name in SETTINGS:
        L.append(r"\midrule")
        for i, (var, meas) in enumerate((("spearman", "Spearman"), ("q90_registered", "AUROC, pre-specified"),
                                         ("q90_lowchange", "AUROC, low change"), ("q90_all", "AUROC, all"))):
            X = [r for r in R if r["src"] == src and r["var"] == var]; n = len(X)
            rs, ps, ss = [x["risk"] for x in X], [x["err_day"] for x in X], [x["surprise"] for x in X]
            c1 = sum(x["lo(risk-surprise)"] > 0 for x in X); c2 = sum(x["lo(risk-err_day)"] > 0 for x in X)
            L.append(f"{name if i == 0 else ''} & {meas} & {n} & {med(rs):.2f} & {med(ps):.2f} & {min(ss):.2f} to {max(ss):.2f} & {c1}/{n} & {c2}/{n} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L).replace("-0.", "-0.").replace(" -", " $-$").replace("$-$0", "$-$0") if False else "\n".join(L)


def change_vs_harm():
    L = [r"""\begin{table}[t]
\centering
\caption{Change is not harm: median AUROC (top 10\% harm) when the negatives are low-harm points with high or with low input change. An input-change detector (novelty) separates harm from quiet, low-change periods but falls to chance when low-harm periods contain large change; the risk monitor and surprise are not affected in this way.}
\label{tab:change_vs_harm}
\small
\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{2}{c}{Novelty} & \multicolumn{2}{c}{Risk} & \multicolumn{2}{c}{Surprise} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Setting & high change & low change & high change & low change & high change & low change \\
\midrule"""]
    for src, name in SETTINGS:
        hi = [r for r in R if r["src"] == src and r["var"] == "q90_registered"]; lo = [r for r in R if r["src"] == src and r["var"] == "q90_lowchange"]
        L.append(f"{name} & " + " & ".join(f"{med([x[k] for x in A]):.2f}" for k in ("novelty", "risk", "surprise") for A in (hi, lo)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def review_h3():
    dirs = [HP.DIAG, os.path.join(HP.ROOT, "outputs", "results_review")]
    F = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "h3review_*_s*_H24.json")):
            if "WeatherClean" not in f: F.setdefault(os.path.basename(f), f)
    X = [json.load(open(f)) for f in F.values()]
    rows = [("split", "Split conformal"), ("risk_only", "Risk scaling, no calibrator"), ("bayes", "Bayesian Student-$t$"),
            ("aci", "ACI"), ("aci_width_matched", "ACI, width-matched to risk-scaled ACI"), ("risk_aci", "Risk-scaled ACI"),
            ("pid", "Conformal PID"), ("pid_scorecaster", "Conformal PID with scorecaster"), ("risk_pid", "Risk-scaled conformal PID")]
    L = [r"""\begin{table}[t]
\centering
\caption{All interval methods of Section~\ref{sec:acting} on the main runs (Titans, $H=24$; post hoc except ACI and risk-scaled ACI). Medians over the %d runs of overall coverage, coverage during harm with the 168-origin window of H3 and with the seven-day window of H1, and Winkler score (lower is better). The width-matched ACI lowers the target miscoverage of ACI until its mean test width equals that of risk-scaled ACI in the same run. On ECL and Traffic a fixed random subset of 64 variates is used.}
\label{tab:review-h3}
\small
\resizebox{\textwidth}{!}{\begin{tabular}{lcccc}
\toprule
Method & Coverage & Coverage in harm (168 origins) & Coverage in harm (7 days) & Winkler \\
\midrule""" % len(X)]
    for k, name in rows:
        v = lambda q: med([x["rows"][k][q] for x in X])
        L.append(f"{name} & {v('cover'):.3f} & {v('cover_h3_168'):.3f} & {v('cover_h1_week'):.3f} & {v('winkler'):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    for name, fn in (("tab_labels", labels), ("tab_change_vs_harm", change_vs_harm), ("tab_review_h3", review_h3)):
        txt = fn(); open(os.path.join(TAB, f"{name}.tex"), "w").write(txt + "\n"); print(txt)
