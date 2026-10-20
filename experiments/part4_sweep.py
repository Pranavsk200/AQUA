import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import out, write_csv
from experiments.parallel import run_jobs
from experiments.stats import paired_summary

LEVELS = [0.5, 0.75, 0.9, 0.95, 0.98]
SEEDS = [0, 1, 2, 3, 4]
METHODS = {
    "magnitude": dict(criterion="magnitude"),
    "ours": dict(criterion="taylor1"),
}
MODEL_DIR = os.path.join(os.path.dirname(out("x")), "..", "data", "models")


def model_path(variant, sparsity, seed):
    return os.path.join(MODEL_DIR, f"{variant}_s{int(sparsity * 100)}_seed{seed}.npz")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    jobs = [dict(variant="dense", sparsity=0.0, seed=s, save_path=model_path("dense", 0, s)) for s in SEEDS]
    for level in LEVELS:
        for name, kw in METHODS.items():
            for s in SEEDS:
                jobs.append(dict(variant=name, sparsity=level, seed=s,
                                 save_path=model_path(name, level, s), **kw))
    rows = run_jobs(jobs)
    write_csv(out("pareto.csv"), sorted(rows, key=lambda r: (r["target_sparsity"], r["variant"], r["seed"])))

    def scores(variant, level, key="test_acc"):
        sel = sorted((r for r in rows if r["variant"] == variant and r["target_sparsity"] == level),
                     key=lambda r: r["seed"])
        return np.array([r[key] for r in sel])

    stats = []
    for level in LEVELS:
        for key in ("test_acc", "val_acc"):
            s = paired_summary(scores("ours", level, key), scores("magnitude", level, key))
            stats.append({"sparsity": level, "metric": key,
                          "ours_mean": scores("ours", level, key).mean(),
                          "magnitude_mean": scores("magnitude", level, key).mean(),
                          "ours_std": scores("ours", level, key).std(ddof=1),
                          "magnitude_std": scores("magnitude", level, key).std(ddof=1), **s})
    write_csv(out("baseline_stats.csv"), stats)

    dense = scores("dense", 0.0)
    xs = [0.0] + LEVELS
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, color in (("magnitude", "tab:blue"), ("ours", "tab:red")):
        mean = [dense.mean()] + [scores(name, l).mean() for l in LEVELS]
        std = [dense.std(ddof=1)] + [scores(name, l).std(ddof=1) for l in LEVELS]
        ax[0].errorbar(xs, mean, yerr=std, marker="o", capsize=3, color=color, label=name)
    ax[0].set_xlabel("sparsity (fraction of weights removed)")
    ax[0].set_ylabel("test accuracy (mean ± std over 5 seeds)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Sparsity-accuracy Pareto curve")

    st = [s for s in stats if s["metric"] == "test_acc"]
    mid = [s["mean_diff"] * 100 for s in st]
    err = [[s["mean_diff"] * 100 - s["ci_low"] * 100 for s in st],
           [s["ci_high"] * 100 - s["mean_diff"] * 100 for s in st]]
    ax[1].errorbar(LEVELS, mid, yerr=err, marker="o", capsize=3, color="k")
    ax[1].axhline(0, color="gray", lw=1)
    ax[1].set_xlabel("sparsity"); ax[1].set_ylabel("ours − magnitude (accuracy points)")
    ax[1].set_title("Paired difference, bootstrap 95% CI"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("pareto.png"), dpi=130)

    print("\nsparsity  magnitude          ours               diff [95% CI]            p(perm)")
    for s in st:
        print(f"{s['sparsity']:.2f}      {s['magnitude_mean']:.4f}±{s['magnitude_std']:.4f}   "
              f"{s['ours_mean']:.4f}±{s['ours_std']:.4f}   "
              f"{s['mean_diff'] * 100:+.2f} [{s['ci_low'] * 100:+.2f}, {s['ci_high'] * 100:+.2f}]   {s['p_perm']:.3f}")


if __name__ == "__main__":
    main()
