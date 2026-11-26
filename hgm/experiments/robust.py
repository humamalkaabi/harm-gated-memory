import copy, json, os, sys, time
import numpy as np, torch
from hgm import paths as HP
from hgm.data import load
from hgm.models.backbones import build, CHANNEL_INDEPENDENT

L = 336


def windows(zt, ts):
    return torch.stack([zt[:, t - L:t] for t in ts])


def pretrain(model, zt, tr, va, H, seed, steps=None, bs=32):
    steps = steps or int(os.environ.get("HGM_PRETRAIN_STEPS", "4000"))
    g = torch.Generator().manual_seed(seed); ci = CHANNEL_INDEPENDENT[model.name]; N = zt.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lo, hi = L, tr - H + 1
    vts = np.linspace(tr, va - H, num=min(64, va - H - tr), dtype=int)
    Xv, Yv = windows(zt, vts), torch.stack([zt[:, t:t + H] for t in vts])
    best, best_state, bad = np.inf, None, 0
    for it in range(steps):
        model.train()
        if ci:
            ts = torch.randint(lo, hi, (128,), generator=g); ns = torch.randint(0, N, (128,), generator=g)
            X = torch.stack([zt[n, t - L:t] for n, t in zip(ns.tolist(), ts.tolist())])
            Y = torch.stack([zt[n, t:t + H] for n, t in zip(ns.tolist(), ts.tolist())]); P = model(X)
        else:
            ts = torch.randint(lo, hi, (bs,), generator=g).tolist()
            X, Y = windows(zt, ts), torch.stack([zt[:, t:t + H] for t in ts]); P = model(X)
        loss = ((P - Y) ** 2).mean(); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (it + 1) % min(100, max(1, steps // 4)) == 0 or it + 1 == steps:
            model.eval()
            with torch.no_grad():
                if ci:
                    xf = Xv.reshape(-1, L); Pv = torch.cat([model(xf[k:k + 2048]) for k in range(0, len(xf), 2048)]).reshape(Yv.shape)
                else:
                    Pv = torch.cat([model(Xv[k:k + 4]) for k in range(0, len(Xv), 4)])
                v = ((Pv - Yv) ** 2).mean().item()
            if v < best: best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= 10: break
    model.load_state_dict(best_state); model.eval()
    return best


def online(model, zt, t0, t1, H, lr):
    model = copy.deepcopy(model); model.eval(); ci = CHANNEL_INDEPENDENT[model.name]
    opt = torch.optim.Adam(model.parameters(), lr=lr) if lr > 0 else None
    N = zt.shape[0]; Y = np.zeros((t1 - t0, N, H), np.float32); mse = np.full(t1 - t0, np.nan)
    for i, t in enumerate(range(t0, t1)):
        x = zt[:, t - L:t]
        with torch.no_grad():
            Y[i] = model(x).numpy()
        if t + H <= zt.shape[1]:
            mse[i] = float(((Y[i] - zt[:, t:t + H].numpy()) ** 2).mean())
        o = t - H
        if opt is not None and o >= L:
            p = model(zt[:, o - L:o]); loss = ((p - zt[:, o:o + H]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return Y, mse


def stream(data, seed, name, H=24):
    out = f"{HP.POLICY}/{data}_H{H}_s{seed}.npz"
    if os.path.exists(out): return out
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    z, tr, va, end = load(data); zt = torch.tensor(z.T, dtype=torch.float32); T1 = end - H + 1
    model = build(name, L, H); model.name = name; ti = time.time()
    vbest = pretrain(model, zt, tr, va, H, seed)
    vm = {lr: float(np.nanmean(online(model, zt, tr, va, H, lr)[1])) for lr in (1e-5, 1e-4, 1e-3)}
    lr = min(vm, key=vm.get)
    Y, mse = online(model, zt, tr, T1, H, lr)
    np.savez_compressed(out, A=Y)
    info = dict(data=data, seed=seed, backbone=name, H=H, pretrain_val=vbest, gd_val=vm, lr=lr,
                test_mse=float(np.nanmean(mse[va - tr:])), sec=time.time() - ti)
    json.dump(info, open(out.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(info), flush=True)
    return out


if __name__ == "__main__":
    stream(sys.argv[1], int(sys.argv[2]), sys.argv[3])
