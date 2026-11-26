import os, subprocess, sys
from hgm import paths as HP


def run(data, H=24, threads=4):
    T = HP.ONLINETSF
    if not os.path.exists(f"{T}/run.py"): print("OnlineTSF is not set up: run  python main.py setup"); return False
    os.makedirs(f"{T}/logs/online", exist_ok=True); os.makedirs(f"{T}/preds", exist_ok=True)
    import torch
    pin = "True" if torch.cuda.is_available() else "False"
    base = [sys.executable, "-u", "run.py", "--dataset", data, "--border_type", "online", "--model", "RLinear",
            "--seq_len", "336", "--pred_len", str(H), "--itr", "1"]
    jobs = [("pre", None, ["--only_test", "--pin_gpu", "False", "--reduce_bs", "False", "--save_opt", "--learning_rate", "0.003"]),
            ("Online", f"preds/{data}_{H}_RL_Online.npz",
             ["--online_method", "Online", "--pretrain", "--pin_gpu", "False", "--reduce_bs", "False", "--save_opt", "--only_test",
              "--online_learning_rate", "0.00001"]),
            ("Proceed", f"preds/{data}_{H}_RL_Proceed.npz",
             ["--batch_size", "16", "--tune_mode", "down_up", "--val_online_lr", "--diff_online_lr", "--pretrain", "--save_opt",
              "--online_method", "Proceed", "--bottleneck_dim", "32", "--concept_dim", "50", "--online_learning_rate", "0.00003",
              "--learning_rate", "0.003", "--lradj", "type3", "--pin_gpu", pin])]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    for tag, pred, extra in jobs:
        if pred and os.path.exists(f"{T}/{pred}"): continue
        log = f"{T}/logs/online/RLinear_{data}_{H}_{tag}.log"
        e = dict(env, SAVE_PRED=pred) if pred else env
        with open(log, "w") as f:
            subprocess.run(base + extra, cwd=T, env=e, stdout=f, stderr=subprocess.STDOUT)
    ok = os.path.exists(f"{T}/preds/{data}_{H}_RL_Proceed.npz")
    print(f"{data}: PROCEED predictions {'saved' if ok else 'MISSING (see ' + T + '/logs/online)'}")
    return ok


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24)
