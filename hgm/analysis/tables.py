import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from hgm import paths as HP

DS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Traffic"]
OUT = HP.TABLES


def _seeds(pattern):
    return sorted(glob.glob(os.path.join(HP.DIAG, pattern)))


def harm():
    cols = [("risk", "Risk monitor"), ("err_day", "Daily error"), ("ph", "Page--Hinkley"), ("adwin", "ADWIN"),
            ("cusum", "CUSUM"), ("d3a", "D3A"), ("surprise", "Surprise")]
    lines = []
    for d in DS:
        R = [json.load(open(f)) for f in _seeds(f"e5_{d}_s*.json")]
        if not R: continue
        m = {k: np.mean([r["auroc"][k][0] for r in R]) for k, _ in cols}
        tie = {k for k, _ in cols[1:] if min(float(r["diff_vs_risk"][k][1]) for r in R) <= 0}
        best = max(m.values())
        cell = lambda k: (r"\textbf{%.3f}" if abs(m[k] - best) < 5e-4 else "%.3f") % m[k] + (r"$^{\dagger}$" if k in tie else "")
        lines.append(f"{d} & {len(R)} & " + " & ".join(cell(k) for k, _ in cols) + r" \\")
    head = " & ".join(n for _, n in cols)
    return (r"""\begin{table}[t]
\centering
\caption{Anticipating harm (H1). Within-variate AUROC for harm weeks under the fixed reference forecaster, averaged over seeds.
Bold marks the best signal; $^{\dagger}$ marks signals that the risk monitor does not beat significantly in every seed
(lower 95\% block-bootstrap bound of the difference $\le 0$).}
\label{tab:harm}
\small
\begin{tabular}{lc""" + "c" * len(cols) + r"""}
\toprule
Dataset & Seeds & """ + head + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def _utility_rows(f):
    t = open(f).read()
    return {m.group(1).strip(): float(m.group(2)) for m in re.finditer(r"^\s+(\S.*?)\s+([+-]\d+\.\d+)%", t, re.M)}


def utility():
    lines = []
    for d in DS:
        S = [json.load(open(f)) for f in _seeds(f"stats_{d}_s*.json")]
        S = [s for s in S if "H6" in s]
        if not S: continue
        au = [v["auroc"] for s in S for v in s["H6"]["pred"].values()]
        rho = [v["rho"] for s in S for v in s["H6"]["pred"].values()]
        tost = sum(v["tost"] for s in S for v in s["H6"]["pred"].values()); n = sum(len(s["H6"]["pred"]) for s in S)
        gain = []
        for f in _seeds(f"utility_{d}_s*.txt"):
            v = _utility_rows(f)
            static = min([0.0] + [x for k, x in v.items() if k.startswith(("standalone", "avg("))])
            gate = min(x for k, x in v.items() if k.startswith(("hedge", "bayes", "lcb")))
            gain.append(static - gate)
        lines.append(f"{d} & {len(S)} & {min(au):.3f}--{max(au):.3f} & {min(rho):+.2f}--{max(rho):+.2f} & {tost}/{n} & {np.mean(gain):+.2f} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{Adaptation utility (H2). Sign-AUROC and Spearman $\rho$ of past versus future paired advantage of each policy over
PROCEED (ranges over policy pairs, windows and seeds); TOST: comparisons equivalent to chance (90\% CI within $[0.45,0.55]$ and
$[-0.10,0.10]$); last column: MSE gain (\%) of the best utility gate over the best static choice in hindsight (PROCEED, any single policy, or a fixed average); negative values mean every gate is worse.}
\label{tab:utility}
\small
\begin{tabular}{lccccc}
\toprule
Dataset & Seeds & Sign-AUROC & Spearman $\rho$ & TOST & Gate gain (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _uq(f):
    r = {}
    for line in open(f):
        p = line.split()
        if len(p) == 5 and p[0] in ("split", "risk", "split_aci", "risk_aci", "bayes"):
            r[p[0]] = tuple(map(float, p[1:]))
    return r


def intervals():
    meth = [("split_aci", "ACI"), ("risk_aci", "Risk-ACI$^{\\ast}$"), ("risk", "Risk"), ("bayes", "Bayes")]
    lines = []
    for d in DS:
        U = [_uq(f) for f in _seeds(f"uq_{d}_s*.txt")]
        S = [json.load(open(f))["H7"] for f in _seeds(f"stats_{d}_s*.json")]
        if not U: continue
        ch = {k: np.mean([u[k][1] for u in U]) for k, _ in meth}
        wk = {k: np.mean([u[k][3] for u in U]) for k, _ in meth}
        bc, bw = max(ch.values()), min(wk.values())
        c = lambda k: (r"\textbf{%.3f}" if abs(ch[k] - bc) < 5e-4 else "%.3f") % ch[k]
        w = lambda k: (r"\textbf{%.2f}" if abs(wk[k] - bw) < 5e-3 else "%.2f") % wk[k]
        cov = sum(s["coverage_ok"] for s in S); wid = sum(s["width_ok"] for s in S)
        lines.append(f"{d} & " + " & ".join(c(k) for k, _ in meth) + " & " + " & ".join(w(k) for k, _ in meth)
                     + f" & {cov}/{len(S)} & {wid}/{len(S)} \\\\")
    names = " & ".join(n for _, n in meth)
    return r"""\begin{table*}[t]
\centering
\caption{Acting on anticipated harm (H3). Coverage during harm weeks (target 0.90) and Winkler score (lower is better), averaged
over seeds; bold marks the best method. $^{\ast}$Pre-registered primary method; the last two columns give the number of runs in
which its coverage gain over ACI is significant (lower 95\% bound $>0$) and its Winkler ratio to ACI stays within the 5\% margin
(upper 95\% bound $<1.05$).}
\label{tab:intervals}
\small
\begin{tabular}{lcccccccccc}
\toprule
& \multicolumn{4}{c}{Coverage during harm} & \multicolumn{4}{c}{Winkler score} & \multicolumn{2}{c}{Risk-ACI vs ACI} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}
Dataset & """ + names + " & " + names + r""" & Coverage & Width \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def generality():
    variants = [("gdn_H24", "Gated DeltaNet", "24"), ("ttt_H24", "TTT-Linear", "24"),
                ("titans_H48", "Titans", "48"), ("titans_H96", "Titans", "96")]
    lines = []
    for key, name, H in variants:
        R = os.path.join(HP.ROOT, "outputs", "phase4", key, "results")
        if not os.path.isdir(R): continue
        risk, sur, tost, ntost, gain, cov = [], [], 0, 0, [], []
        for d in DS:
            E = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/e5_{d}_s*.json"))]
            if E:
                risk.append(np.mean([e["auroc"]["risk"][0] for e in E])); sur.append(np.mean([e["auroc"]["surprise"][0] for e in E]))
            for f in sorted(glob.glob(f"{R}/stats_{d}_s*.json")):
                st = json.load(open(f))
                if "H6" in st:
                    tost += sum(v["tost"] for v in st["H6"]["pred"].values()); ntost += len(st["H6"]["pred"])
                cov.append(st["H7"]["coverage_ok"])
            for f in sorted(glob.glob(f"{R}/utility_{d}_s*.txt")):
                v = _utility_rows(f)
                static = min([0.0] + [x for k, x in v.items() if k.startswith(("standalone", "avg("))])
                gain.append(static - min(x for k, x in v.items() if k.startswith(("hedge", "bayes", "lcb"))))
        if not risk: continue
        lines.append(f"{name} & {H} & {len(risk)} & {min(risk):.2f}--{max(risk):.2f} & {min(sur):.2f}--{max(sur):.2f} & "
                     f"{tost}/{ntost} & {max(gain):+.2f} & {sum(cov)}/{len(cov)} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{Generality across memory families and horizons. Ranges over datasets of the mean AUROC for harm weeks (risk monitor
and memory surprise); H2: comparisons equivalent to chance (TOST) and the largest gain (\%) of a utility gate over the best static
choice in hindsight; H3: runs with a significant coverage gain of risk-scaled ACI over ACI during harm.}
\label{tab:generality}
\small
\begin{tabular}{lccccccc}
\toprule
Memory & $H$ & Datasets & Risk AUROC & Surprise AUROC & H2 TOST & Max gate gain & H3 coverage \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.isdir(os.path.join(HP.ROOT, "outputs", "phase4")):
        open(os.path.join(OUT, "generality.tex"), "w").write(generality()); print("wrote", os.path.join(OUT, "generality.tex"))
    for name, fn in (("harm", harm), ("utility", utility), ("intervals", intervals)):
        open(os.path.join(OUT, f"{name}.tex"), "w").write(fn()); print("wrote", os.path.join(OUT, f"{name}.tex"))


if __name__ == "__main__":
    main()
