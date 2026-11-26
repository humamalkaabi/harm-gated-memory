import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from hgm import paths as HP

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
OUT = os.path.join(HP.ROOT, "outputs"); FIGS = os.path.join(OUT, "figures")
D = np.load(os.path.join(OUT, "results", "e5series_ETTh2_s1.npz"))
R, dX, pos, neg = D["R"], D["dX"], D["pos"], D["neg"]; nt, B = R.shape
NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def auc(k, b):
    S = D["S_" + k][:, b]; y = np.r_[np.ones(pos[:, b].sum()), np.zeros(neg[:, b].sum())]
    return roc_auc_score(y, np.r_[S[pos[:, b]], S[neg[:, b]]])


a_risk = np.array([auc("risk", b) for b in range(B)])
b = int(np.argsort(a_risk)[B // 2])
z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
days = np.arange(nt)


def spans(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]: j += 1
            out.append((i - 0.5, j + 0.5)); i = j + 1
        else: i += 1
    return out


fig, ax = plt.subplots(3, 1, figsize=(7.2, 3.9), sharex=True, gridspec_kw=dict(hspace=0.28))
rows = [(np.log10(dX[:, b] + 1e-6), "Input change\n(log$_{10}$, next week)", "#7F8C8D", None),
        (z(D["S_surprise"][:, b]), "Memory surprise\n(z-score)", "#C0392B", auc("surprise", b)),
        (z(D["S_risk"][:, b]), "Risk monitor\n(z-score)", "#2E7D32", a_risk[b])]
hw = spans(pos[:, b]); cw = spans(neg[:, b])
for a_, (y, lab, c, A) in zip(ax, rows):
    for s, e in hw: a_.axvspan(s, e, color="#E8B4B0", alpha=0.55, lw=0)
    a_.plot(days, y, c=c, lw=0.9)
    a_.set_ylabel(lab, fontsize=7.5)
    if A is not None:
        a_.text(0.995, 0.92, f"AUROC {A:.2f}", transform=a_.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(fc="white", ec="none", pad=1))
nd = np.flatnonzero(neg[:, b]); ytop = np.log10(dX[:, b] + 1e-6).max() + 0.6
ax[0].scatter(nd, np.full(len(nd), ytop), marker="v", s=7, c="#555", lw=0)
ax[2].set_xlabel("Test day (decision point)")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax[0].legend([Patch(color="#E8B4B0"), Line2D([], [], marker="v", c="#555", lw=0, ms=4)],
             ["harm week (top 10% of future risk)", "large change, low harm"], frameon=False, fontsize=7,
             loc="upper left", ncol=2, bbox_to_anchor=(0, 1.32))
fig.savefig(os.path.join(FIGS, "F1_teaser.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIGS, "F1_teaser.png"), dpi=240, bbox_inches="tight")
print("variate", b, NAMES[b], "risk AUROC", round(a_risk[b], 3), "surprise", round(auc("surprise", b), 3),
      "novelty", round(auc("novelty", b), 3), "all risk", np.round(a_risk, 2))
