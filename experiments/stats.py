import itertools

import numpy as np


def bootstrap_ci(diffs, n=10000, alpha=0.05, seed=0):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_permutation_p(diffs):
    # exact sign-flip test: under "no difference" each seed's sign is a coin flip
    diffs = np.asarray(diffs, dtype=float)
    observed = abs(diffs.mean())
    count = 0
    signs = list(itertools.product([1, -1], repeat=len(diffs)))
    for s in signs:
        if abs((np.array(s) * diffs).mean()) >= observed - 1e-12:
            count += 1
    return count / len(signs)


def paired_summary(a, b):
    # a, b: per-seed scores for the two methods, same seed order
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(d)
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    return {
        "mean_diff": float(d.mean()), "std_diff": float(sd), "ci_low": lo, "ci_high": hi,
        "p_perm": paired_permutation_p(d), "effect_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
        "wins": int((d > 0).sum()), "n": len(d),
    }
