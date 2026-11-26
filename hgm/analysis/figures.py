from hgm import paths as HP
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
os.environ["UQ_RETURN"] = "1"
from hgm.uncertainty import intervals as dev_uq
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})
D = np.load(HP.DIAG + "/e5series_ETTh2_s1.npz"); ts, R, dX, pos, neg = D["ts"], D["R"], D["dX"], D["pos"], D["neg"]
S = {k[2:]: D[k] for k in D.files if k.startswith("S_")}; nt, B = R.shape; d = int(D["d"]); H = int(D["H"])
z = lambda x: (x - np.nanmean(x, 0)) / (np.nanstd(x, 0) + 1e-9)
C = dict(risk="#2E7D32", surprise="#C0392B", novelty="#7F8C8D", err_day="#1F4E79", ph="#8E44AD")
NAME = dict(risk="Risk monitor", surprise="Memory surprise", novelty="Input novelty", err_day="Revealed error (1 day)", ph="Page-Hinkley alarms")

fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 1.1]))
days = np.arange(nt)
for a_, k, cm in ((ax[0], "surprise", "RdBu_r"), (ax[1], "risk", "RdBu_r")):
    im = a_.imshow(z(S[k]).T, aspect="auto", cmap=cm, norm=TwoSlopeNorm(0, -2.5, 2.5), interpolation="nearest")
    a_.set_yticks(range(B)); a_.set_yticklabels([f"v{b + 1}" for b in range(B)], fontsize=7)
    a_.set_title(f"{NAME[k]} (z-score per variate)", loc="left")
    fig.colorbar(im, ax=a_, fraction=0.02, pad=0.01)
ax[2].imshow(pos.T, aspect="auto", cmap="Greys", interpolation="nearest")
ax[2].set_yticks(range(B)); ax[2].set_yticklabels([f"v{b + 1}" for b in range(B)], fontsize=6)
ax[2].set_title("Harm weeks of the reference forecaster (top 10 % of future risk)", loc="left")
ax[2].set_xlabel("Test day (decision point)")
fig.colorbar(ax[2].images[0], ax=ax[2], fraction=0.02, pad=0.01).remove()
fig.tight_layout(); fig.savefig(HP.FIGS + "/V1_heatmaps.png", dpi=220); fig.savefig(HP.FIGS + "/V1_heatmaps.pdf"); plt.close(fig)

fig, a_ = plt.subplots(figsize=(4.8, 4.0))
from hgm.data import load
zz, tr0, va0, end0 = load("ETTh2"); W7 = 7 * d
dXr = np.array([np.abs(zz[tr0 + t + d:tr0 + t + d + W7].mean(0) - zz[tr0 + t - W7:tr0 + t].mean(0)) / np.maximum(zz[tr0 + t - W7:tr0 + t].std(0), 0.1) for t in ts])
x = dXr.ravel(); y = R.ravel(); xt = np.quantile(x, 0.75); yt = np.quantile(y, 0.9)
quad = np.where(x >= xt, np.where(y >= yt, 2, 0), np.where(y >= yt, 1, 3))
cols = ["#7F8C8D", "#C0392B", "#8E44AD", "#BDC3C7"]; labs = ["Change, no harm", "Harm, no change", "Harmful change", "Stable"]
for q in (3, 0, 1, 2):
    m = quad == q; a_.scatter(x[m], y[m], s=6, c=cols[q], alpha=0.55 if q != 3 else 0.3, lw=0, label=f"{labs[q]} ({m.mean() * 100:.0f}%)")
a_.axvline(xt, c="#555", lw=0.8, ls="--"); a_.axhline(yt, c="#555", lw=0.8, ls="--")
a_.set_xscale("log"); a_.set_ylim(-6, 6); a_.set_xlabel("Input change over the next week (dX, log scale)"); a_.set_ylabel("Future harm R (log risk ratio)")
a_.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=2); a_.set_title("All four combinations of change and harm occur", loc="left")
fig.tight_layout(); fig.savefig(HP.FIGS + "/V2_quadrants.png", dpi=220); fig.savefig(HP.FIGS + "/V2_quadrants.pdf"); plt.close(fig)

lags = np.arange(-7, 8); ev = []
for b in range(B):
    p = pos[:, b]; on = np.flatnonzero(p & ~np.r_[False, p[:-1]])
    ev += [(j, b) for j in on if 7 <= j < nt - 7]
fig, a_ = plt.subplots(figsize=(4.8, 3.2))
for k in ("risk", "err_day", "ph", "novelty", "surprise"):
    Z = z(S[k]); m = np.array([[Z[j + l, b] for l in lags] for j, b in ev])
    mu = np.nanmean(m, 0); se = np.nanstd(m, 0) / np.sqrt(len(m))
    a_.plot(lags, mu, c=C[k], lw=2 if k == "risk" else 1.3, label=NAME[k]); a_.fill_between(lags, mu - 1.96 * se, mu + 1.96 * se, color=C[k], alpha=0.12, lw=0)
a_.axvline(0, c="#333", lw=0.8, ls="--"); a_.axhline(0, c="#999", lw=0.6)
a_.set_xlabel("Days relative to the start of a harm week"); a_.set_ylabel("Signal (z-score)")
a_.set_title(f"Signals around the start of a harm week ({len(ev)} onsets)", loc="left", fontsize=8.5); a_.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(HP.FIGS + "/V3_event_aligned.png", dpi=220); fig.savefig(HP.FIGS + "/V3_event_aligned.pdf"); plt.close(fig)

aR, Wd, nv, tr = dev_uq.main("ETTh2", 24, 1, path=HP.POLICY + "/ETTh2_H24_s1.npz")
cov = {k: (aR <= Wd[k]).mean(2) for k in ("split", "split_aci", "risk_aci")}
CL = dict(split="#7F8C8D", split_aci="#1F4E79", risk_aci="#2E7D32"); NM = dict(split="Split conformal", split_aci="ACI", risk_aci="Risk-scaled ACI (ours)")
lagd = np.arange(-7, 15); fig, a_ = plt.subplots(figsize=(4.8, 3.2))
for k, cv in cov.items():
    rows = []
    for j, b in ev:
        o = int(ts[j]) + d
        if o - 7 * d >= 0 and o + 15 * d <= cv.shape[0]:
            rows.append([cv[o + l * d:o + (l + 1) * d, b].mean() for l in lagd])
    rows = np.array(rows); a_.plot(lagd, rows.mean(0), c=CL[k], lw=2 if k == "risk_aci" else 1.3, label=NM[k])
a_.axhline(0.9, c="#333", lw=0.8, ls=":"); a_.axvspan(0, 7, color="#C0392B", alpha=0.07, lw=0)
a_.set_xlabel("Days relative to the start of a harm week"); a_.set_ylabel("Coverage (target 0.90)")
a_.set_title("Coverage around the start of a harm week (ETTh2, seed 1)", loc="left", fontsize=9); a_.legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(HP.FIGS + "/V4_coverage.png", dpi=220); fig.savefig(HP.FIGS + "/V4_coverage.pdf"); plt.close(fig)
print("events", len(ev), "quadrant shares", [round((quad == q).mean(), 3) for q in range(4)])
