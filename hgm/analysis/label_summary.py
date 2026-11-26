import glob, json, os
from hgm import paths as HP

VARS = ["q90_registered", "q90_all", "q90_lowchange", "q80_registered", "q80_all", "q80_lowchange",
        "q95_registered", "q95_all", "q95_lowchange", "spearman"]


def main():
    base = os.path.join(HP.ROOT, "outputs"); lines = []
    for f in sorted(glob.glob(os.path.join(base, "**", "results", "e5v2_*.json"), recursive=True)):
        x = json.load(open(f)); tag = os.path.relpath(os.path.dirname(os.path.dirname(f)), base); tag = "main" if tag == "." else tag
        for v in VARS:
            r = x["spearman"] if v == "spearman" else x["variants"][v]
            lines.append(f"{tag}\t{x['data']}\ts{x['seed']}\tH{x['H']}\t{v}\trisk={r['risk'][0]}[{r['risk'][1]},{r['risk'][2]}]"
                         f"\terr_day={r['err_day'][0]}\tsurprise={r['surprise'][0]}[{r['surprise'][1]},{r['surprise'][2]}]"
                         f"\tnovelty={r['novelty'][0]}\tlo(risk-err_day)={r['risk_minus_err_day'][1]}\tlo(risk-surprise)={r['risk_minus_surprise'][1]}")
    out = os.path.join(base, "label_variants.txt"); open(out, "w").write("\n".join(lines) + "\n"); print("wrote", out, len(lines), "rows")


if __name__ == "__main__":
    main()
