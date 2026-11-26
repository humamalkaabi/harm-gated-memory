import glob, json, os, re, statistics as st
from hgm import paths as HP

OUT = os.path.join(HP.ROOT, "outputs"); TAB = HP.TABLES
DATA = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Traffic"]
MAIN = [os.path.join(OUT, "results"), os.path.join(OUT, "results_main_extra")]


def files(prefix, dirs=MAIN):
    fs = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, prefix + "*"))): fs.setdefault(os.path.basename(f), f)
    return fs


def rng(v): return f"{min(v):.2f}" if min(v) == max(v) or len(v) == 1 else f"{st.mean(v):.2f}"


def h1():
    E = [json.load(open(f)) for f in files("e5_").values()]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{H1 on the main runs (Titans memory, $H=24$). AUROC of harm against change without harm, mean over seeds. The last three columns count the seeds that meet each pre-specified rule: the lower 95\% bound of risk minus surprise, and of risk minus the persistence baseline, above zero; and the upper 95\% bound of the AUROC of surprise below 0.55. Best other detector: the strongest of the seven drift detectors of Section~\ref{sec:setup-harm}. Variates with fewer than five positives or five negatives are excluded: one of 321 on ECL and 17 of 862 on Traffic, none elsewhere.}",
         r"\label{tab:h1}", r"\small", r"\setlength{\tabcolsep}{4.5pt}", r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrccc}", r"\toprule",
         r" & & \multicolumn{4}{c}{AUROC} & \multicolumn{3}{c}{Seeds meeting the rule} \\ \cmidrule(lr){3-6}\cmidrule(lr){7-9}",
         r"Dataset & Seeds & Risk & Persist. & Surprise & Best other & $>$surprise & $>$persist. & surprise $<$0.55 \\", r"\midrule"]
    tot = [0, 0, 0, 0]
    for d in DATA:
        X = [x for x in E if x["data"] == d]; n = len(X)
        a = lambda k: [x["auroc"][k][0] for x in X]
        others = ["d3a", "cusum", "ph", "adwin", "kswin", "ddm", "hddm"]
        bo = max(others, key=lambda k: st.mean(a(k)))
        c1 = sum(x["diff_vs_risk"]["surprise"][1] > 0 for x in X); c2 = sum(x["diff_vs_risk"]["err_day"][1] > 0 for x in X)
        c3 = sum(x["auroc"]["surprise"][2] < 0.55 for x in X); tot = [tot[0] + n, tot[1] + c1, tot[2] + c2, tot[3] + c3]
        nm = dict(ph="PH", adwin="ADWIN", cusum="CUSUM", d3a="D3A", ddm="DDM", hddm="HDDM", kswin="KSWIN")[bo]
        L.append(f"{d} & {n} & {st.mean(a('risk')):.2f} & {st.mean(a('err_day')):.2f} & {st.mean(a('surprise')):.2f} & {st.mean(a(bo)):.2f} ({nm}) & {c1}/{n} & {c2}/{n} & {c3}/{n} \\\\")
    L += [r"\midrule", f"All & {tot[0]} & & & & & {tot[1]}/{tot[0]} & {tot[2]}/{tot[0]} & {tot[3]}/{tot[0]} \\\\",
          r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(TAB, "tab_h1.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L[10:]))


def util_gain(path):
    t = open(path).read(); vals = {}
    for m in re.finditer(r"^\s+(standalone [ABC]|avg\([^)]*\)|hedge\([^)]*\)|bayes_sw\([^)]*\)|bayes_mix\([^)]*\)|lcb\([^)]*\)):?\s+([+-][\d.]+)%", t, re.M):
        vals[m.group(1)] = float(m.group(2))
    static = min([0.0] + [v for n, v in vals.items() if n.startswith(("standalone", "avg"))])
    gates = [v for n, v in vals.items() if n.split("(")[0] in ("hedge", "bayes_sw", "bayes_mix", "lcb")]
    return static - min(gates)


def h23():
    S = [json.load(open(f)) for f in files("stats_").values()]
    U = files("utility_", MAIN)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{H2 and H3 on the main runs (Titans memory, $H=24$). Predictability: comparisons (three policies $\times$ two windows $\times$ seeds) whose sign-AUROC and correlation are both equivalent to chance by TOST, and the largest sign-AUROC. Best gate: improvement in MSE of the best gate over the best static choice in hindsight, in percentage points of the MSE of PROCEED (positive favours the gate), median over seeds. H3: median gain in coverage during harm of risk-scaled ACI over ACI, median Winkler ratio, and the seeds that meet the pre-specified coverage rule and width rule.}",
         r"\label{tab:h23}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{\begin{tabular}{lcrrrrcc}", r"\toprule",
         r" & \multicolumn{3}{c}{H2} & \multicolumn{4}{c}{H3} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-8}",
         r"Dataset & TOST & Max AUROC & Best gate (pp) & Cov.\ gain & Winkler & Cov.\ rule & Width rule \\", r"\midrule"]
    T = [0, 0, 0, 0, 0]
    for d in DATA:
        X = [x for x in S if x["data"] == d]; n = len(X)
        P = [v for x in X for v in x["H6"]["pred"].values()]
        tost = sum(v["tost"] for v in P); mx = max(v["auroc"] for v in P)
        g = [util_gain(U[f"utility_{d}_s{x['seed']}.txt"]) for x in X]
        cg = st.median(x["H7"]["cov_gain"] for x in X); wr = st.median(x["H7"]["winkler_ratio"] for x in X)
        c = sum(x["H7"]["coverage_ok"] for x in X); w = sum(x["H7"]["width_ok"] for x in X)
        T = [T[0] + tost, T[1] + len(P), T[2] + c, T[3] + w, T[4] + n]
        L.append(f"{d} & {tost}/{len(P)} & {mx:.2f} & {st.median(g):+.2f} & {cg:.3f} & {wr:.2f} & {c}/{n} & {w}/{n} \\\\")
    L += [r"\midrule", f"All & {T[0]}/{T[1]} & & & & & {T[2]}/{T[4]} & {T[3]}/{T[4]} \\\\", r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(TAB, "tab_h23.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L[10:]))


if __name__ == "__main__":
    h1(); h23()
