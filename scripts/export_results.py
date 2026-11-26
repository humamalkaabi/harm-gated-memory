import json, os, shutil, sys

DROP = ()
SKIP_DIRS = ("policies", "logs", "tables", "figures")
SKIP_FILES = ("label_variants_mac.txt", "rebuild_report.txt", "rebuild_bundle.zip")


def unwanted(name):
    return any(d in name for d in DROP)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if not unwanted(str(k)) and not (k == "data" and unwanted(str(v)))}
    if isinstance(obj, list):
        return [clean(v) for v in obj if not (isinstance(v, dict) and unwanted(str(v.get("data", ""))))]
    return obj


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.endswith(".json"):
        json.dump(clean(json.load(open(src))), open(dst, "w"), indent=1)
    elif src.endswith((".jsonl", ".txt", ".csv")):
        open(dst, "w").writelines(l for l in open(src) if not unwanted(l))
    else:
        shutil.copy2(src, dst)


def main(src_root):
    dst_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n = 0
    for sub in ("outputs", "checkpoints"):
        base = os.path.join(src_root, sub)
        for d, dirs, files in os.walk(base):
            rel = os.path.relpath(d, base)
            dirs[:] = [x for x in dirs if not unwanted(x) and not (sub == "outputs" and rel == "." and x in SKIP_DIRS)
                       and x not in ("gdn", "ttt")]
            for f in files:
                if unwanted(f) or f in SKIP_FILES or f.endswith((".tmp", ".part")): continue
                if sub == "outputs" and f.endswith(".npz") and not f.startswith("e5series_"): continue
                if sub == "checkpoints" and "_mem_mlp_" in f: continue
                copy_file(os.path.join(d, f), os.path.join(dst_root, sub, rel, f)); n += 1
    bad = []
    for sub in ("outputs", "checkpoints"):
        for d, _, files in os.walk(os.path.join(dst_root, sub)):
            for f in files:
                p = os.path.join(d, f)
                if unwanted(p): bad.append(p); continue
                if f.endswith((".json", ".jsonl", ".txt", ".csv")) and any(x in open(p, errors="ignore").read() for x in DROP):
                    bad.append(p)
    print(f"copied {n} files into {dst_root}")
    if bad:
        print("files that still mention an excluded dataset:"); print("\n".join(bad)); sys.exit(1)
    print("check passed")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--exclude" in args:
        i = args.index("--exclude"); names = [x for x in args[i + 1].split(",") if x]; del args[i:i + 2]
        DROP = tuple(names) + tuple(x.lower() for x in names)
    if len(args) != 1: sys.exit("usage: python scripts/export_results.py SOURCE [--exclude NAME[,NAME...]]")
    main(os.path.abspath(args[0]))
