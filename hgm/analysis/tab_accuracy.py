import glob, os, re, statistics as st
from hgm import paths as HP

OUT = os.path.join(HP.ROOT, "outputs"); DATA = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Traffic"]
SRC = {24: [os.path.join(OUT, "results"), os.path.join(OUT, "results_main_extra")],
       48: [os.path.join(OUT, "phase4", "titans_H48", "results")], 96: [os.path.join(OUT, "phase4", "titans_H96", "results")]}


def read(H):
    fs = {}
    for d in SRC[H]:
        for f in glob.glob(os.path.join(d, "utility_*_s*.txt")): fs.setdefault(os.path.basename(f), f)
    rows = {}
    for f in fs.values():
        t = open(f).read(); m = re.search(r"(\w+) H(\d+) s(\d) +reference PROCEED MSE ([\d.]+)", t)
        P = float(m.group(4)); v = {x: float(y) for x, y in re.findall(r"standalone ([ABC]): ([+-][\d.]+)%", t)}
        rows.setdefault(m.group(1), []).append([P] + [P * (1 + v[k] / 100) for k in "ABC"])
    return rows


def main():
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Test MSE on the normalised scale, mean over three seeds (one for ETTh1). PROCEED: the online baseline of Section~\ref{sec:models} on its own RLinear backbone; A: the plain forecaster with one Adam step per revealed origin (the reference); B: the memory forecaster with FR and the updates of A; C: the memory forecaster with FR and frozen parameters. Bold: lowest mean MSE for each dataset and horizon.}",
         r"\label{tab:accuracy}", r"\small", r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "rrrr" * 3 + "}", r"\toprule",
         r" & \multicolumn{4}{c}{$H=24$} & \multicolumn{4}{c}{$H=48$} & \multicolumn{4}{c}{$H=96$} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
         r"Dataset" + r" & PROCEED & A & B & C" * 3 + r" \\", r"\midrule"]
    R = {H: read(H) for H in SRC}
    for d in DATA:
        cells = []
        for H in SRC:
            if d not in R[H]: cells += ["--"] * 4; continue
            m = [st.mean(x[i] for x in R[H][d]) for i in range(4)]; b = min(m)
            cells += [(r"\textbf{%.3f}" if v == b else "%.3f") % v for v in m]
        L.append(d + " & " + " & ".join(cells) + r" \\")
    if any("--" in x for x in L[9:]): L[2] = L[2][:-1] + " Datasets without a run at a horizon are marked --.}"
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(HP.TABLES, "tab_accuracy.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L[9:]))


if __name__ == "__main__":
    main()
