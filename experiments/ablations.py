import sys

import numpy as np

from experiments.common import out, write_csv
from experiments.parallel import run_jobs

VARIANTS = {
    "magnitude": dict(criterion="magnitude"),
    "taylor1": dict(criterion="taylor1"),
    "taylor1_nonorm": dict(criterion="taylor1", normalize=False),
    "taylor2": dict(criterion="taylor2"),
    "magnitude_nonorm": dict(criterion="magnitude", normalize=False),
    "taylor1_regrow": dict(criterion="taylor1", regrow=0.3),
    "magnitude_regrow": dict(criterion="magnitude", regrow=0.3),
    "taylor1_oneshot": dict(criterion="taylor1", schedule="oneshot"),
    "magnitude_oneshot": dict(criterion="magnitude", schedule="oneshot"),
    "random": dict(criterion="random"),
}
ROUND2 = {
    "taylor1_ema99": dict(criterion="taylor1", ema=0.99),
    "taylor2_ema99": dict(criterion="taylor2", ema=0.99),
    "taylor2_regrow": dict(criterion="taylor2", regrow=0.3),
    "taylor2_ema99_regrow": dict(criterion="taylor2", ema=0.99, regrow=0.3),
}
# different seeds from the final sweep so nothing is tuned on the reported runs
SEEDS = [100, 101, 102]
LEVELS = [0.9, 0.95]


def main():
    round2 = len(sys.argv) > 1 and sys.argv[1] == "2"
    variants = ROUND2 if round2 else VARIANTS
    jobs = [dict(variant=name, sparsity=s, seed=seed, **kw)
            for s in LEVELS for name, kw in variants.items() for seed in SEEDS]
    flat = run_jobs(jobs)
    write_csv(out("ablations2.csv" if round2 else "ablations.csv"), flat)

    print("\nvariant              sparsity   val_acc (mean±std)   test_acc (mean)")
    for s in LEVELS:
        for name in variants:
            sel = [r for r in flat if r["variant"] == name and r["target_sparsity"] == s]
            v = np.array([r["val_acc"] for r in sel])
            t = np.array([r["test_acc"] for r in sel])
            print(f"{name:20s} {s:5.2f}      {v.mean():.4f} ± {v.std():.4f}      {t.mean():.4f}")


if __name__ == "__main__":
    main()
