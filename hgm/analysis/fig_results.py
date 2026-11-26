import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from hgm import paths as HP

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
OUT = os.path.join(HP.ROOT, "outputs"); FIGS = os.path.join(OUT, "figures"); os.makedirs(FIGS, exist_ok=True)
DATA = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Traffic"]
SIG = ["risk", "err_day", "d3a", "cusum", "ph", "adwin", "kswin", "ddm", "hddm", "novelty", "surprise"]
NAME = dict(risk="Risk monitor (ours)", err_day="Revealed error, 1 day", d3a="D3A test", cusum="CUSUM",
            ph="Page-Hinkley", adwin="ADWIN", kswin="KSWIN", ddm="DDM", hddm="HDDM-A",
            novelty="Input novelty", surprise="Memory surprise")
SETTINGS = [("Titans\n$H$=24", [os.path.join(OUT, "results"), os.path.join(OUT, "results_main_extra")]),
            ("GDN\n$H$=24", [os.path.join(OUT, "phase4", "gdn_H24", "results")]),
            ("TTT\n$H$=24", [os.path.join(OUT, "phase4", "ttt_H24", "results")]),
            ("Titans\n$H$=48", [os.path.join(OUT, "phase4", "titans_H48", "results")]),
            ("Titans\n$H$=96", [os.path.join(OUT, "phase4", "titans_H96", "results")]),
            ("Patch-\nTST", [os.path.join(OUT, "robust", "patchtst", "results")]),
            ("iTrans-\nformer", [os.path.join(OUT, "robust", "itransformer", "results")])]


def load(dirs):
    out = {}
    for d in DATA:
        for dr in dirs:
            fs = sorted(glob.glob(os.path.join(dr, f"e5_{d}_s*.json")))
            if fs:
                A = [json.load(open(f))["auroc"] for f in fs]
                out[d] = {k: float(np.mean([a[k][0] for a in A])) for k in SIG if k in A[0]}; break
    return out


def cell_text(ax, M):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}"[1:] if M[i, j] < 1 else "1.0", ha="center", va="center", fontsize=6.6,
                        color="white" if abs(M[i, j] - 0.5) > 0.2 else "#222")


def heatmap():
    main = load(SETTINGS[0][1])
    M1 = np.array([[main.get(d, {}).get(k, np.nan) for d in DATA] for k in SIG])
    per = [load(dirs) for _, dirs in SETTINGS]
    M2 = np.array([[np.nanmean([v[k] for v in p.values() if k in v]) if p else np.nan for p in per] for k in SIG])
    norm = TwoSlopeNorm(0.5, 0.3, 0.85); cmap = "RdBu_r"
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[8, 7], wspace=0.06))
    for a_, M, cols, title in ((ax[0], M1, DATA, "(a) Titans, $H$ = 24, by dataset (mean of seeds)"),
                               (ax[1], M2, [s for s, _ in SETTINGS], "(b) By setting (mean of datasets)")):
        im = a_.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        cell_text(a_, M)
        a_.set_xticks(range(len(cols))); a_.set_xticklabels(cols, fontsize=7 if a_ is ax[0] else 6.5, rotation=0 if a_ is ax[1] else 35,
                                                             ha="center" if a_ is ax[1] else "right")
        a_.set_yticks(range(len(SIG))); a_.set_yticklabels([NAME[k] for k in SIG] if a_ is ax[0] else [])
        a_.axhline(8.5, c="black", lw=1.2)
        a_.set_title(title, loc="left", fontsize=8.5)
        for s in a_.spines.values(): s.set_visible(False)
        a_.tick_params(length=0)
    ax[0].text(-0.62, 4, "realised error", rotation=90, ha="center", va="center", fontsize=7, color="#444",
               transform=ax[0].get_yaxis_transform())
    ax[0].text(-0.62, 9.5, "inputs /\nmemory", rotation=90, ha="center", va="center", fontsize=7, color="#444",
               transform=ax[0].get_yaxis_transform())
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015); cb.set_label("AUROC (0.5 = chance)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(FIGS, "F2_auroc_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "F2_auroc_heatmap.png"), dpi=240, bbox_inches="tight"); plt.close(fig)
    return M1, M2


C = dict(risk="#2E7D32", err_day="#1F4E79", novelty="#7F8C8D", surprise="#C0392B")
LAB = dict(risk="Risk monitor", err_day="Revealed error (persistence)", novelty="Input novelty", surprise="Memory surprise")
SRC = [("main", "Titans\n$H$=24"), ("phase4/titans_H48", "Titans\n$H$=48"), ("phase4/titans_H96", "Titans\n$H$=96"),
       ("robust/patchtst", "PatchTST\n$H$=24"), ("robust/itransformer", "iTransf.\n$H$=24")]


def label_robust():
    R = [r for r in json.load(open(os.path.join(OUT, "label_variants_all.json"))) if r["data"] != "WeatherClean"]
    g = lambda s, v: [r for r in R if r["src"] == s and r["var"] == v]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
    a_ = ax[0]; off = dict(risk=-0.27, err_day=-0.09, novelty=0.09, surprise=0.27)
    rng = np.random.default_rng(0)
    for i, (s, _) in enumerate(SRC):
        G = g(s, "spearman")
        for k in off:
            v = np.array([r[k] for r in G]); x = i + off[k]
            a_.scatter(x + rng.uniform(-0.035, 0.035, len(v)), v, s=6, color=C[k], alpha=0.45, lw=0)
            a_.plot([x - 0.07, x + 0.07], [np.median(v)] * 2, color=C[k], lw=2.2, solid_capstyle="butt",
                    label=LAB[k] if i == 0 else None)
    a_.axhline(0, c="#888", lw=0.7)
    a_.set_xticks(range(len(SRC))); a_.set_xticklabels([n for _, n in SRC], fontsize=7)
    a_.set_ylabel("Spearman correlation with future harm $R$")
    a_.set_title("(a) Label-free: rank correlation with future harm", loc="left", fontsize=8.5)
    a_.legend(frameon=False, fontsize=6.6, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8)
    a_.set_ylim(-0.25, 0.68)
    a_ = ax[1]
    for k in ("novelty", "risk", "surprise"):
        for s, _ in SRC:
            hi = np.median([r[k] for r in g(s, "q90_registered")]); lo = np.median([r[k] for r in g(s, "q90_lowchange")])
            a_.plot([0, 1], [lo, hi], color=C[k], lw=0.8, alpha=0.45, marker="o", ms=2.5)
        hi = np.median([r[k] for s, _ in SRC for r in g(s, "q90_registered")])
        lo = np.median([r[k] for s, _ in SRC for r in g(s, "q90_lowchange")])
        a_.plot([0, 1], [lo, hi], color=C[k], lw=2.4, marker="o", ms=5)
        a_.text(1.07, {"risk": hi, "novelty": hi + 0.022, "surprise": hi - 0.02}[k], LAB[k], color="#222", fontsize=7, va="center")
    a_.axhline(0.5, c="#888", lw=0.7, ls="--"); a_.text(-0.05, 0.505, "chance", fontsize=6.5, color="#666", va="bottom")
    a_.set_xticks([0, 1]); a_.set_xticklabels(["low-harm,\nlow change", "low-harm,\nhigh change"], fontsize=7)
    a_.set_xlim(-0.15, 1.6); a_.set_ylim(0.38, 0.85)
    a_.set_ylabel("AUROC (negatives as labelled)")
    a_.set_title("(b) Change is not harm (median AUROC)", loc="left", fontsize=8.5)
    fig.savefig(os.path.join(FIGS, "F3_label_robust.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "F3_label_robust.png"), dpi=240, bbox_inches="tight"); plt.close(fig)


def intervals_fig():
    fs = sorted(glob.glob(os.path.join(OUT, "sensitivity", "sens_*.json")))
    P = [json.load(open(f))["pareto"] for f in fs]
    pairs = [("ACI", "risk^0+aci0.005", "risk^1.0+aci0.005"), ("DtACI", "dtaci", "risk+dtaci"),
             ("PID", "pid0.05", "risk+pid0.05")]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.42))
    cb, cr = "#1F4E79", "#2E7D32"; rng = np.random.default_rng(1)
    for j, (lab, b, r) in enumerate(pairs):
        for a_, key, yl in ((ax[0], "cover_harm", "Coverage in harm weeks"), (ax[1], "cover", "Overall coverage")):
            yb = np.array([p[b][key] for p in P]); yr = np.array([p[r][key] for p in P]); x0 = 3 * j
            for u, v in zip(yb, yr): a_.plot([x0, x0 + 1], [u, v], c="#bbb", lw=0.4, zorder=1)
            a_.scatter(x0 + rng.uniform(-.08, .08, len(yb)), yb, s=5, c=cb, lw=0, zorder=2)
            a_.scatter(x0 + 1 + rng.uniform(-.08, .08, len(yr)), yr, s=5, c=cr, lw=0, zorder=2)
            a_.plot([x0 - .25, x0 + .25], [np.median(yb)] * 2, c=cb, lw=2.2); a_.plot([x0 + .75, x0 + 1.25], [np.median(yr)] * 2, c=cr, lw=2.2)
            a_.set_ylabel(yl)
        w = np.array([p[r]["winkler"] / p[b]["winkler"] for p in P])
        ax[2].scatter(j + rng.uniform(-.12, .12, len(w)), w, s=6, c="#555", lw=0); ax[2].plot([j - .25, j + .25], [np.median(w)] * 2, c="black", lw=2.2)
        ax[2].text(j + 0.3, np.median(w), f"{np.median(w):.2f}", va="center", fontsize=7)
        print(lab, "harm cover gain in", int(np.sum([p[r]["cover_harm"] > p[b]["cover_harm"] for p in P])), "/", len(P),
              "median", round(float(np.median([p[r]["cover_harm"] - p[b]["cover_harm"] for p in P])), 3), "winkler median", round(float(np.median(w)), 3))
    for a_ in ax[:2]:
        a_.set_xticks([0.5, 3.5, 6.5]); a_.set_xticklabels([p[0] for p in pairs], fontsize=7)
        a_.axhline(0.9, c="#888", lw=0.7, ls="--")
    ax[0].set_title("(a) Coverage when harm occurs", loc="left", fontsize=8.5)
    ax[1].set_title("(b) Overall coverage (target 0.9)", loc="left", fontsize=8.5)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels([p[0] for p in pairs], fontsize=7); ax[2].axhline(1, c="#888", lw=0.7, ls="--")
    ax[2].set_yscale("log"); ax[2].set_yticks([0.9, 1, 1.5, 2, 3]); ax[2].set_yticklabels(["0.9", "1", "1.5", "2", "3"]); ax[2].set_ylabel("Winkler score, risk-scaled / base"); ax[2].set_title("(c) Interval score (lower is better)", loc="left", fontsize=8.5)
    from matplotlib.lines import Line2D
    ax[0].legend([Line2D([], [], c=cb, lw=2.2), Line2D([], [], c=cr, lw=2.2)], ["Base controller", "Risk-scaled"],
                 frameon=False, fontsize=6.8, loc="lower right", bbox_to_anchor=(1.06, -0.02), handlelength=1.2)
    fig.savefig(os.path.join(FIGS, "F4_intervals.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "F4_intervals.png"), dpi=240, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    M1, M2 = heatmap(); label_robust(); intervals_fig()
    np.set_printoptions(precision=2, suppress=True, linewidth=160); print(M1); print(M2)
