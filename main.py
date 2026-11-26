import argparse, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

SPYDER_COMMAND = "reproduce"

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from hgm import paths as HP

PHASE3 = ["ETTm1", "Weather", "ECL", "Traffic"]
BIG = threading.Semaphore(1)
SEEDS = [1, 2, 3]


def sh(args, log=None, threads=1, extra=None):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1", PYTHONPATH=ROOT)
    env.update(extra or {})
    cmd = [sys.executable, *args]
    if log is None: return subprocess.run(cmd, cwd=ROOT, env=env).returncode
    with open(log, "a") as f: rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        tail = open(log, errors="replace").read().strip().splitlines()[-6:]
        step(f"FAILED ({os.path.basename(log)}):\n    " + "\n    ".join(tail))
    return rc


def step(msg): print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def train(d, s, kind, H=24, iters=200):
    if os.path.exists(f"{HP.CKPT}/{d}_H{H}_{kind}_linear_s{s}.pt"): return 0
    if kind == "plain" and HP.MEMORY != "titans":
        base = f"{HP.CKPT_BASE}/{d}_H{H}_plain_linear_s{s}.pt"
        if not os.path.exists(base):
            rc = sh(["-m", "hgm.train", "--data", d, "--H", str(H), "--seed", str(s), "--kind", "plain", "--iters", str(iters)],
                    f"{HP.LOGS}/train_{d}_H{H}_plain_s{s}.txt", threads=2, extra={"HGM_MEMORY": "titans"})
            if rc: return rc
        os.makedirs(HP.CKPT, exist_ok=True); shutil.copy(base, f"{HP.CKPT}/{d}_H{H}_plain_linear_s{s}.pt"); return 0
    return sh(["-m", "hgm.train", "--data", d, "--H", str(H), "--seed", str(s), "--kind", kind, "--iters", str(iters)],
              f"{HP.LOGS}/train_{d}_H{H}_{kind}_s{s}.txt", threads=2)


def policies(d, s, H=24):
    if os.path.exists(f"{HP.POLICY}/{d}_H{H}_s{s}.json") and os.path.exists(f"{HP.POLICY}/{d}_H{H}_s{s}.npz"): return 0
    return sh(["-m", "hgm.experiments.policies", d, str(H), str(s)], f"{HP.LOGS}/policies_{d}_s{s}.txt")


def harm(d, s, H=24):
    if os.path.exists(f"{HP.DIAG}/e5_{d}_s{s}.json"): return 0
    return sh(["-c", f"from hgm.experiments.harm_detection import main; main('{d}', {s}, H={H})"], f"{HP.LOGS}/harm_{d}_s{s}.txt")


def _to_file(code, out, log):
    tmp = out + ".tmp"; rc = sh(["-c", code], tmp)
    if rc == 0: os.replace(tmp, out)
    else: os.replace(tmp, log)
    return rc


def utility(d, s, H=24):
    out = f"{HP.DIAG}/utility_{d}_s{s}.txt"
    if os.path.exists(out): return 0
    if not os.path.exists(f"{HP.ONLINETSF}/preds/{d}_{H}_RL_Proceed.npz"): step(f"utility {d} H{H}: no PROCEED predictions, skipped"); return 0
    return _to_file(f"from hgm.experiments.utility import main; main('{d}', H={H}, seed={s})", out, f"{HP.LOGS}/utility_{d}_s{s}.err")


def intervals(d, s, H=24):
    out = f"{HP.DIAG}/uq_{d}_s{s}.txt"
    if os.path.exists(out): return 0
    return _to_file(f"from hgm.uncertainty.intervals import main; main('{d}', H={H}, seed={s}, path='{HP.POLICY}/{d}_H{H}_s{s}.npz')",
                    out, f"{HP.LOGS}/intervals_{d}_s{s}.err")


def stats(d, s, H=24):
    if os.path.exists(f"{HP.DIAG}/stats_{d}_s{s}.json"): return 0
    return sh(["-m", "hgm.analysis.stats", d, str(s), str(H)], f"{HP.LOGS}/stats_{d}_s{s}.txt")


def evaluate(d, s, H=24, harm_all=False):
    if d in ("ECL", "Traffic"):
        with BIG: return _evaluate(d, s, H, harm_all)
    return _evaluate(d, s, H, harm_all)


def _evaluate(d, s, H, harm_all):
    need_harm = harm_all or d in ("ECL", "Traffic")
    if os.path.exists(f"{HP.DIAG}/stats_{d}_s{s}.json") and (not need_harm or os.path.exists(f"{HP.DIAG}/e5_{d}_s{s}.json")):
        return 0
    step(f"start {d} s{s}")
    for k in ("mem", "plain"):
        if train(d, s, k, H): step(f"FAILED training {d} {k} s{s}"); return 1
    rc = policies(d, s, H)
    if rc: step(f"FAILED policies {d} s{s} (see {HP.LOGS})"); return rc
    if need_harm: harm(d, s, H)
    utility(d, s, H); intervals(d, s, H); stats(d, s, H)
    if d in ("ECL", "Traffic"):
        for f in (f"{HP.POLICY}/{d}_H{H}_s{s}.npz",):
            if os.path.exists(f"{HP.DIAG}/stats_{d}_s{s}.json") and os.path.exists(f): os.remove(f)
    step(f"done {d} s{s}"); return 0


def _setup_if_needed():
    from hgm import prepare
    if not os.path.exists(f"{HP.ONLINETSF}/run.py") or not os.path.isdir(f"{HP.TCPD}/.git"):
        step("first run: setting up datasets, OnlineTSF (PROCEED) and TCPD"); prepare.all_()


def run_grid(datasets, H=24, jobs=None, harm_all=False):
    from hgm import baselines
    jobs = jobs or max(1, min(6, (os.cpu_count() or 4) - 4))
    step(f"memory {HP.MEMORY}, H = {H}, datasets {datasets}, {jobs} parallel jobs, outputs in {HP.OUT}")
    with ThreadPoolExecutor(1) as gpu, ThreadPoolExecutor(jobs) as pool:
        base = {d: gpu.submit(baselines.run, d, H) for d in datasets}
        work = []
        for d in datasets:
            base[d].result()
            work += [pool.submit(evaluate, d, s, H, harm_all) for s in seeds_of(d)]
        rcs = [f.result() for f in work]
    step(("DONE" if not any(rcs) else f"DONE with {sum(1 for r in rcs if r)} failed runs (see above)") + f" -> {HP.DIAG}")


def seeds_of(d): return [1] if d == "ETTh1" else SEEDS


def phase3(jobs=None):
    _setup_if_needed(); run_grid(PHASE3, 24, jobs)


ALL = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Traffic"]
PHASE4 = [("gdn", 24), ("ttt", 24), ("titans", 96), ("titans", 48)]


def phase4(jobs=None):
    _setup_if_needed()
    for rule, H in PHASE4:
        out = os.path.join(ROOT, "outputs", "phase4", f"{rule}_H{H}")
        args = ["main.py", "grid", "--H", str(H)] + (["--jobs", str(jobs)] if jobs else [])
        step(f"===== phase 4: memory {rule}, H = {H}")
        sh(args, extra={"HGM_MEMORY": rule, "HGM_OUT": out}, threads=os.cpu_count() or 4)
    step("PHASE4_DONE -> outputs/phase4/")


def sensitivity():
    for d in ("ECL", "Traffic"):
        npz = f"{HP.POLICY}/{d}_H24_s1.npz"
        if os.path.exists(f"{HP.DIAG}/sens_{d}_H24_s1.json"): continue
        step(f"sensitivity: regenerating the {d} stream (seed 1)")
        if policies(d, 1, 24) == 0:
            sh(["-m", "hgm.analysis.sensitivity", d, "1"], f"{HP.LOGS}/sens_{d}_s1.txt")
            if os.path.exists(npz): os.remove(npz)
    step("sensitivity: main runs, H = 24"); sh(["-m", "hgm.analysis.sensitivity"], f"{HP.LOGS}/sens_main.txt")
    for v in ("titans_H48", "titans_H96"):
        step(f"sensitivity: {v}")
        sh(["-m", "hgm.analysis.sensitivity"], f"{HP.LOGS}/sens_{v}.txt",
           extra={"HGM_OUT": os.path.join(ROOT, "outputs", "phase4", v)})
    step("SENSITIVITY_DONE")


ROBUST = ["ETTh2", "ETTm1", "Weather", "ECL"]


def robust(jobs=None):
    _setup_if_needed()
    jobs = jobs or 3
    def one(bb, d, s):
        out = os.path.join(ROOT, "outputs", "robust", bb); env = {"HGM_OUT": out, "HGM_SKIP_H6": "1"}
        res = os.path.join(out, "results")
        if os.path.exists(f"{res}/stats_{d}_s{s}.json") and os.path.exists(f"{res}/e5_{d}_s{s}.json"): return 0
        step(f"start {bb} {d} s{s}")
        os.makedirs(os.path.join(out, "logs"), exist_ok=True); log = os.path.join(out, "logs", f"{d}_s{s}.txt")
        rc = sh(["-m", "hgm.experiments.robust", d, str(s), bb], log, threads=4, extra=env)
        if rc: return rc
        rc = sh(["-c", f"from hgm.experiments.harm_detection import main; main('{d}', {s}, H=24)"], log, extra=env)
        rc = rc or sh(["-m", "hgm.analysis.stats", d, str(s), "24"], log, extra=env)
        npz = os.path.join(out, "policies", f"{d}_H24_s{s}.npz")
        if d in ("ECL",) and rc == 0 and os.path.exists(npz): os.remove(npz)
        step(f"done {bb} {d} s{s}" if rc == 0 else f"FAILED {bb} {d} s{s} (see {log})"); return rc
    work = [(bb, d, s) for bb in ("patchtst", "itransformer") for d in ROBUST for s in SEEDS]
    def guarded(a):
        if a[1] == "ECL":
            with BIG: return one(*a)
        return one(*a)
    with ThreadPoolExecutor(jobs) as pool: rcs = list(pool.map(guarded, work))
    step(("ROBUST_DONE" if not any(rcs) else f"ROBUST_DONE with {sum(1 for r in rcs if r)} failed runs") + " -> outputs/robust/")


def labels(jobs=None):
    import glob, re
    jobs = jobs or 3
    srcs = [(HP.OUT, None)] + [(os.path.join(ROOT, "outputs", "phase4", v), None) for v in ("titans_H48", "titans_H96")] \
         + [(os.path.join(ROOT, "outputs", "robust", bb), None) for bb in ("patchtst", "itransformer")]
    def one(out, d, s, H):
        res = os.path.join(out, "results", f"e5v2_{d}_s{s}.json")
        if os.path.exists(res): return 0
        step(f"labels {os.path.relpath(out, ROOT)} {d} s{s} H{H}")
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        return sh(["-c", f"from hgm.experiments.harm_detection import main; main('{d}', {s}, H={H})"],
                  os.path.join(out, "logs", f"labels_{d}_s{s}.txt"), extra={"HGM_OUT": out, "HGM_LABEL_VARIANTS": "1"})
    for d in ("ECL", "Traffic"):
        npz = f"{HP.POLICY}/{d}_H24_s1.npz"
        if os.path.exists(f"{HP.DIAG}/e5v2_{d}_s1.json"): continue
        made = not os.path.exists(npz)
        if made: step(f"labels: regenerating the {d} stream (seed 1)")
        if policies(d, 1, 24) == 0:
            one(HP.OUT, d, 1, 24)
            if made and os.path.exists(npz): os.remove(npz)
    work = []
    for out, _ in srcs:
        for f in sorted(glob.glob(os.path.join(out, "policies", "*_H*_s*.npz"))):
            m = re.match(r"(.+)_H(\d+)_s(\d+)\.npz$", os.path.basename(f))
            if m: work.append((out, m.group(1), int(m.group(3)), int(m.group(2))))
    def guarded(a):
        if a[1] in ("ECL", "Traffic"):
            with BIG: return one(*a)
        return one(*a)
    with ThreadPoolExecutor(jobs) as pool: rcs = list(pool.map(guarded, work))
    sh(["-m", "hgm.analysis.label_summary"], os.path.join(HP.LOGS, "labels_summary.txt"))
    step(("LABELS_DONE" if not any(rcs) else f"LABELS_DONE with {sum(1 for r in rcs if r)} failed runs") + " -> outputs/label_variants.txt")


def review(jobs=None):
    _setup_if_needed()
    jobs = jobs or 3
    work = [(d, s) for d in ALL for s in seeds_of(d)]
    def one(d, s):
        res = f"{HP.DIAG}"
        if all(os.path.exists(f"{res}/{p}") for p in (f"e5v3_{d}_s{s}.json", f"h2review_{d}_s{s}.json", f"h3review_{d}_s{s}_H24.json")):
            return 0
        step(f"review {d} s{s}")
        log = f"{HP.LOGS}/review_{d}_s{s}.txt"
        rc = sh(["-m", "hgm.train", "--data", d, "--H", "24", "--seed", str(s), "--kind", "mem", "--mem", "mlp", "--iters", "200"], log, threads=2)
        if rc: return rc
        npz = f"{HP.POLICY}/{d}_H24_s{s}.npz"; made = not os.path.exists(npz)
        if policies(d, s, 24): return 1
        if not os.path.exists(f"{res}/e5v3_{d}_s{s}.json"):
            rc = sh(["-c", f"from hgm.experiments.harm_detection import main; main('{d}', {s}, H=24)"], log, extra={"HGM_V3": "1"}) or rc
        if not os.path.exists(f"{res}/h2review_{d}_s{s}.json"):
            rc = sh(["-m", "hgm.analysis.review_h2", d, str(s)], log) or rc
        if not os.path.exists(f"{res}/h3review_{d}_s{s}_H24.json"):
            rc = sh(["-m", "hgm.analysis.review_intervals", d, str(s)], log, extra={"HGM_UQ_SUBSET": "1"}) or rc
        if d in ("ECL", "Traffic") and made and os.path.exists(npz): os.remove(npz)
        step(f"done review {d} s{s}" if rc == 0 else f"FAILED review {d} s{s} (see {log})"); return rc
    def guarded(a):
        if a[0] in ("ECL", "Traffic"):
            with BIG: return one(*a)
        return one(*a)
    with ThreadPoolExecutor(jobs) as pool: rcs = list(pool.map(guarded, work))
    step(("REVIEW_DONE" if not any(rcs) else f"REVIEW_DONE with {sum(1 for r in rcs if r)} failed runs") + " -> outputs/results/")


def same(jobs=None):
    _setup_if_needed()
    jobs = jobs or 3
    work = [(d, s) for d in ALL for s in seeds_of(d)]
    def one(d, s):
        if os.path.exists(f"{HP.DIAG}/e5same_{d}_s{s}.json"): return 0
        if not os.path.exists(f"{HP.CKPT}/{d}_H24_mem_linear_s{s}.pt"):
            if train(d, s, "mem"): return 1
        if not os.path.exists(f"{HP.CKPT}/{d}_H24_plain_linear_s{s}.pt") and not os.path.exists(f"{HP.POLICY}/{d}_H24_s{s}.npz"):
            if train(d, s, "plain"): return 1
        step(f"same-model {d} s{s}")
        rc = sh(["-m", "hgm.experiments.same_model", d, str(s), "24"], f"{HP.LOGS}/same_{d}_s{s}.txt")
        step(f"done same-model {d} s{s}" if rc == 0 else f"FAILED same-model {d} s{s} (see outputs/logs/same_{d}_s{s}.txt)")
        return rc
    def guarded(a):
        if a[0] in ("ECL", "Traffic"):
            with BIG: return one(*a)
        return one(*a)
    with ThreadPoolExecutor(jobs) as pool: rcs = list(pool.map(guarded, work))
    sh(["-m", "hgm.analysis.same_model_summary"])
    step(("SAME_DONE" if not any(rcs) else f"SAME_DONE with {sum(1 for r in rcs if r)} failed runs") + " -> outputs/same_model.txt")


def fix(jobs=None):
    import glob
    _setup_if_needed()
    if sh(["-m", "tests.causality"], f"{HP.LOGS}/causality.txt"):
        step("FIX STOPPED: a decision rule failed the causal-availability test (see outputs/logs/causality.txt)"); return
    step("causal-availability test passed for every H2 decision rule")
    jobs = jobs or 3
    variants = [("titans", 24, HP.OUT)] + [(r, h, os.path.join(ROOT, "outputs", "phase4", f"{r}_H{h}")) for r, h in PHASE4]
    for _, _, out in variants:
        res = os.path.join(out, "results"); pre = os.path.join(res, "precorrection"); os.makedirs(pre, exist_ok=True)
        pats = ["utility_*_s*.txt", "stats_*_s*.json"] + (["h2review_*_s*.json", "h3review_*_s*_H24.json", "e20.json"] if out == HP.OUT else [])
        for pat in pats:
            for f in glob.glob(os.path.join(res, pat)):
                if not os.path.exists(os.path.join(pre, os.path.basename(f))): shutil.copy(f, pre)
    sh(["-m", "hgm.experiments.delayed_feedback"], f"{HP.LOGS}/delayed_fix.txt"); step("controlled experiment recomputed")

    def one(rule, H, out, d, s):
        res = os.path.join(out, "results"); env = {"HGM_MEMORY": rule, "HGM_OUT": out}
        flag = os.path.join(res, f"fixed_{d}_s{s}.flag")
        main_run = out == HP.OUT
        st = os.path.join(res, f"stats_{d}_s{s}.json")
        if os.path.exists(flag):
            if not os.path.exists(st) or "winkler_upper975" in open(st).read(): return 0
            stats_only = True
        else:
            stats_only = False
        os.makedirs(os.path.join(out, "logs"), exist_ok=True); log = os.path.join(out, "logs", f"fix_{d}_s{s}.txt")
        pol = os.path.join(out, "policies", f"{d}_H{H}_s{s}.npz"); made = not os.path.exists(pol)
        if made and not main_run and d in ("ECL", "Traffic") and not os.environ.get("HGM_FIX_FULL"):
            step(f"fix {rule} H{H} {d} s{s}: stream not kept, skipped (set HGM_FIX_FULL=1 to regenerate, ~2 h each)")
            return 0
        if made and not main_run:
            step(f"fix {rule} H{H} {d} s{s}: no saved stream, regenerating")
        step(f"fix {rule} H{H} {d} s{s}")
        for f in (f"utility_{d}_s{s}.txt", f"stats_{d}_s{s}.json", f"h2review_{d}_s{s}.json", f"h3review_{d}_s{s}_H24.json",
                  f"e1change_{d}_s{s}.json"):
            if stats_only and not f.startswith("stats_"): continue
            if os.path.exists(os.path.join(res, f)) and (main_run or not f.startswith(("h2", "h3", "e1"))): os.remove(os.path.join(res, f))
        rc = 0
        if made:
            rc = sh(["-m", "hgm.experiments.policies", d, str(H), str(s)], log, extra=env)
            if rc: step(f"FAILED fix {rule} H{H} {d} s{s} (policies)"); return rc
        if not stats_only:
            u_out = os.path.join(res, f"utility_{d}_s{s}.txt"); tmp = u_out + ".tmp"
            if os.path.exists(tmp): os.remove(tmp)
            rcu = sh(["-c", f"from hgm.experiments.utility import main; main('{d}', H={H}, seed={s})"], tmp, extra=env)
            if rcu == 0: os.replace(tmp, u_out)
            rc = rcu or rc
        rc = sh(["-m", "hgm.analysis.stats", d, str(s), str(H)], log, extra=env) or rc
        if main_run and not stats_only:
            rc = sh(["-m", "hgm.analysis.review_h2", d, str(s)], log) or rc
            rc = sh(["-m", "hgm.analysis.review_intervals", d, str(s)], log, extra={"HGM_UQ_SUBSET": "1"}) or rc
            rc = sh(["-m", "hgm.experiments.surprise_change", d, str(s), "24"], log) or rc
        if made and d in ("ECL", "Traffic") and os.path.exists(pol): os.remove(pol)
        if rc == 0: open(flag, "w").write("ok")
        step(f"done fix {rule} H{H} {d} s{s}" if rc == 0 else f"FAILED fix {rule} H{H} {d} s{s} (see {log})"); return rc

    rcs = []
    for rule, H, out in variants:
        if not os.path.exists(f"{HP.ONLINETSF}/preds/ETTh1_{H}_RL_Proceed.npz"):
            step(f"fix {rule} H{H}: no PROCEED predictions for H={H}, skipped"); continue
        work = [(rule, H, out, d, s) for d in ALL for s in seeds_of(d)]
        def guarded(a):
            if a[3] in ("ECL", "Traffic"):
                with BIG: return one(*a)
            return one(*a)
        with ThreadPoolExecutor(jobs) as pool: rcs += list(pool.map(guarded, work))
    sh(["-m", "hgm.analysis.fix_summary"], f"{HP.LOGS}/fix_summary.txt")
    step(("FIX_DONE" if not any(rcs) else f"FIX_DONE with {sum(1 for r in rcs if r)} failed runs") + " -> outputs/fix_report.txt")


def e5(jobs=None):
    _setup_if_needed()
    jobs = jobs or 3; d = "WeatherClean"
    def one(s):
        log = f"{HP.LOGS}/e5_{d}_s{s}.txt"
        if os.path.exists(f"{HP.DIAG}/h3review_{d}_s{s}_H24.json"): return 0
        step(f"E5 {d} s{s}")
        for k in ("mem", "plain"):
            if train(d, s, k): step(f"FAILED E5 training {k} s{s}"); return 1
        if policies(d, s): step(f"FAILED E5 policies s{s}"); return 1
        rc = harm(d, s)
        rc = sh(["-c", f"from hgm.experiments.harm_detection import main; main('{d}', {s}, H=24)"], log, extra={"HGM_LABEL_VARIANTS": "1"}) or rc
        rc = intervals(d, s) or rc
        rc = stats(d, s) or rc
        rc = sh(["-m", "hgm.analysis.review_intervals", d, str(s)], log) or rc
        step(f"done E5 {d} s{s}" if rc == 0 else f"FAILED E5 {d} s{s} (see {log})"); return rc
    with ThreadPoolExecutor(jobs) as pool: rcs = list(pool.map(one, SEEDS))
    sh(["-m", "hgm.analysis.e5_summary"], f"{HP.LOGS}/e5_summary.txt")
    step(("E5_DONE" if not any(rcs) else f"E5_DONE with {sum(1 for r in rcs if r)} failed runs") + " -> outputs/e5_report.txt")


def e6(jobs=None):
    jobs = jobs or 3
    def one(d, s):
        out = f"{HP.DIAG}/h3vol_{d}_s{s}_H24.json"
        npz = f"{HP.POLICY}/{d}_H24_s{s}.npz"; made = not os.path.exists(npz)
        if os.path.exists(out) and ("vol_aci_rob" in json.load(open(out))["rows"] or made): return 0
        if made and d in ("ECL", "Traffic") and not os.environ.get("HGM_FIX_FULL"):
            step(f"E6 {d} s{s}: stream not kept, skipped (set HGM_FIX_FULL=1 to regenerate)"); return 0
        step(f"E6 {d} s{s}")
        if made and policies(d, s, 24): return 1
        rc = sh(["-m", "hgm.analysis.vol_intervals", d, str(s)], f"{HP.LOGS}/e6_{d}_s{s}.txt")
        if made and d in ("ECL", "Traffic") and os.path.exists(npz): os.remove(npz)
        step(f"done E6 {d} s{s}" if rc == 0 else f"FAILED E6 {d} s{s} (see outputs/logs/e6_{d}_s{s}.txt)"); return rc
    def guarded(a):
        if a[0] in ("ECL", "Traffic"):
            with BIG: return one(*a)
        return one(*a)
    with ThreadPoolExecutor(jobs) as pool: rcs = list(pool.map(guarded, [(d, s) for d in ALL for s in seeds_of(d)]))
    sh(["-m", "hgm.analysis.e6_summary"], f"{HP.LOGS}/e6_summary.txt")
    step(("E6_DONE" if not any(rcs) else f"E6_DONE with {sum(1 for r in rcs if r)} failed runs") + " -> outputs/e6_report.txt")


def cli(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("command"); ap.add_argument("--data", default="ETTh1"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--H", type=int, default=24); ap.add_argument("--kind", default="mem"); ap.add_argument("--jobs", type=int)
    a = ap.parse_args(argv); c = a.command
    if c == "setup":
        from hgm import prepare; prepare.all_()
    elif c == "train": sys.exit(train(a.data, a.seed, a.kind, a.H))
    elif c == "proceed":
        from hgm import baselines; baselines.run(a.data, a.H)
    elif c == "policies": sys.exit(policies(a.data, a.seed, a.H))
    elif c == "harm": sys.exit(harm(a.data, a.seed, a.H))
    elif c == "utility": sys.exit(utility(a.data, a.seed, a.H))
    elif c == "intervals": sys.exit(intervals(a.data, a.seed, a.H))
    elif c == "stats": sys.exit(stats(a.data, a.seed, a.H))
    elif c == "evaluate": sys.exit(evaluate(a.data, a.seed, a.H, harm_all=True))
    elif c == "grid": run_grid(ALL, a.H, a.jobs, harm_all=True)
    elif c == "phase4": phase4(a.jobs)
    elif c == "delayed": sh(["-m", "hgm.experiments.delayed_feedback"])
    elif c == "tcpd": sh(["-m", "hgm.experiments.tcpd"])
    elif c == "summary": sh(["-m", "hgm.analysis.summarize"])
    elif c == "figures": sh(["-m", "hgm.analysis.figures"]); sh(["-m", "hgm.analysis.fig_framework"])
    elif c == "tables": sh(["-m", "hgm.analysis.tables"])
    elif c == "sensitivity": sensitivity()
    elif c == "robust": robust(a.jobs)
    elif c == "labels": labels(a.jobs)
    elif c == "review": review(a.jobs)
    elif c == "same": same(a.jobs)
    elif c == "fix": fix(a.jobs)
    elif c == "e5": e5(a.jobs)
    elif c == "e6": e6(a.jobs)
    elif c == "phase3": phase3(a.jobs)
    elif c == "main": run_grid(ALL, 24, a.jobs, harm_all=True)
    elif c == "generality": phase4(a.jobs)
    elif c == "robustness": robust(a.jobs)
    elif c == "posthoc":
        for f in (labels, review, same, fix, e5, e6): f(a.jobs)
        sensitivity()
    elif c == "reproduce": sys.exit(subprocess.call(["bash", os.path.join(ROOT, "scripts", "reproduce_paper.sh")]))
    else: ap.error(f"unknown command {c}")


if __name__ == "__main__":
    cli(sys.argv[1:] if len(sys.argv) > 1 else SPYDER_COMMAND.split())
