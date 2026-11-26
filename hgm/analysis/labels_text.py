from hgm import paths as HP
import json, os, numpy as np

R = [r for r in json.load(open(os.path.join(HP.ROOT, "outputs", "label_variants_all.json"))) if r["data"] != "WeatherClean"]
SR = {"main": 24, "phase4/titans_H48": 48, "phase4/titans_H96": 96, "robust/patchtst": "PatchTST", "robust/itransformer": "iTransformer"}
sp = [r for r in R if r["var"] == "spearman"]
print(f"runs in the label analysis: {len(sp)}")
print(f"surprise Spearman range: {min(r['surprise'] for r in sp):.2f} to {max(r['surprise'] for r in sp):.2f}")
print(f"risk Spearman positive in {sum(r['risk'] > 0 for r in sp)}/{len(sp)}; medians by setting: " +
      ", ".join(f"{SR[s]}: {np.median([r['risk'] for r in sp if r['src'] == s]):.2f}" for s in SR))
print(f"risk minus surprise lower bound > 0 in {sum(r['lo(risk-surprise)'] > 0 for r in sp)}/{len(sp)}")
tr = [r for r in sp if r["data"] == "Traffic" and r["src"] == "main"]
if tr: print("Traffic (main) Spearman means: risk %.2f, persistence %.2f, surprise %.2f" % tuple(np.mean([r[k] for r in tr]) for k in ("risk", "err_day", "surprise")))
for s in ("main", "phase4/titans_H48", "phase4/titans_H96"):
    X = [r for r in sp if r["src"] == s]
    print(f"{SR[s]}: risk Spearman above persistence in {sum(r['risk'] > r['err_day'] for r in X)}/{len(X)}, lower bound > 0 in {sum(r['lo(risk-err_day)'] > 0 for r in X)}")
med = lambda s, v, k: np.median([r[k] for r in R if r["src"] == s and r["var"] == v])
rng = lambda v, k: (min(med(s, v, k) for s in SR), max(med(s, v, k) for s in SR))
print("novelty median AUROC, low-change negatives: %.2f to %.2f; high-change (pre-specified): %.2f to %.2f" % (rng("q90_lowchange", "novelty") + rng("q90_registered", "novelty")))
print("risk median AUROC, low change and high change, over settings: %.2f to %.2f" % (min(rng("q90_lowchange", "risk")[0], rng("q90_registered", "risk")[0]), max(rng("q90_lowchange", "risk")[1], rng("q90_registered", "risk")[1])))
print("risk median AUROC, pre-specified: %.2f to %.2f; all non-positives as negatives: %.2f to %.2f" % (rng("q90_registered", "risk") + rng("q90_all", "risk")))
for s in ("main", "phase4/titans_H48", "phase4/titans_H96"):
    X = [r for r in R if r["src"] == s and r["var"] == "q90_all"]
    print(f"{SR[s]}, all non-positives: risk > persistence (lower bound > 0) in {sum(r['lo(risk-err_day)'] > 0 for r in X)}/{len(X)}")
