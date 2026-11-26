import numpy as np
from hgm.experiments import utility as U
from hgm.policies.gates import gate
from hgm.analysis.review_h2 import ewma_revealed, deployable_choice

RNG = np.random.default_rng(0)


def gates_pair_precorrection(lP, lX, H, warm):
    n = len(lP); d = lP - lX; nb = n // H; blk = d[:nb * H].reshape(nb, H).mean(1)
    r = np.var(blk[:warm // H]) + 1e-12
    M, V = U.kalman_utility(blk, 0.05 * r, r); p = np.zeros(n)
    for i in range(n):
        k = (i + 1) // H - 2
        if k >= 0: p[i] = 1 - U.norm.cdf(0, M[k], np.sqrt(V[k]))
    return {"bayes_mix": p}


def perturb_after(x, i, H):
    y = x.copy(); y[i - H + 1:] = RNG.normal(size=y[i - H + 1:].shape) ** 2 + 0.1
    return y


def check(name, fn, H, warm, n, checks=200):
    lP = RNG.normal(size=n) ** 2 + 0.1; lX = RNG.normal(size=n) ** 2 + 0.1
    base = fn(lP, lX)
    bad = 0
    for i in RNG.integers(warm, n, checks):
        alt = fn(perturb_after(lP, i, H), perturb_after(lX, i, H))
        bad += int(any(not np.isclose(base[k][i], alt[k][i]) for k in base))
    print(f"  {name:38} H={H:3}: {'CAUSAL' if bad == 0 else f'LOOK-AHEAD at {bad}/{checks} decisions'}")
    return bad


def main():
    fails = 0
    for H in (24, 48, 96):
        n, warm = 3000, 720
        fails += check("Bayesian and LCB gates (corrected)", lambda a, b: U.gates_pair(a, b, H, warm), H, warm, n) > 0
        ctrl = check("Bayesian gate, pre-correction (control)", lambda a, b: gates_pair_precorrection(a, b, H, warm), H, warm, n, 400)
        B, K = 1, 2
        Y = {k: RNG.normal(size=(n, B, H)) for k in ("P", "X")}; tgt = RNG.normal(size=(n, B, H))
        for kind, par in (("hedge", 0.1), ("switch", 168)):
            _, w0 = gate(Y, tgt, H, ["P", "X"], kind, par, nval=warm); bad = 0
            for i in RNG.integers(warm, n, 50):
                t2 = tgt.copy(); t2[i - H + 1:] = RNG.normal(size=t2[i - H + 1:].shape)
                _, w1 = gate(Y, t2, H, ["P", "X"], kind, par, nval=warm); bad += int(not np.isclose(w0[i], w1[i]).all())
            print(f"  {kind + ' gate':38} H={H:3}: {'CAUSAL' if bad == 0 else f'LOOK-AHEAD at {bad}/50 decisions'}"); fails += bad > 0
        fails += check("disagreement-aware gate (m-hat)", lambda a, b: {"m": ewma_revealed((a - b)[:, None], H)[:, 0]}, H, warm, n) > 0
        d0 = deployable_choice({"P": np.ones(n), "X": np.ones(n) * 2}, H, warm); bad = 0
        for _ in range(50):
            lP, lX = RNG.normal(size=n) ** 2, RNG.normal(size=n) ** 2
            c0 = deployable_choice({"P": lP, "X": lX}, H, warm)
            c1 = deployable_choice({"P": perturb_after(lP, warm, H), "X": perturb_after(lX, warm, H)}, H, warm)
            bad += int(c0 != c1)
        print(f"  {'deployable static choice':38} H={H:3}: {'CAUSAL' if bad == 0 else f'LOOK-AHEAD in {bad}/50 draws'}"); fails += bad > 0
        if ctrl == 0: print("  WARNING: the control was not flagged; the test has no power at this H")
    print("ALL CAUSAL" if fails == 0 else f"{fails} rule(s) FAILED")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
