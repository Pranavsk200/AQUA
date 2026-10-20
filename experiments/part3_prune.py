import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import out, write_csv
from experiments.runner import run_one
from train.data import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--criterion", default="taylor2", choices=["magnitude", "taylor1", "taylor2", "random"])
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--schedule", default="cubic", choices=["cubic", "oneshot"])
    ap.add_argument("--regrow", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data = load_dataset(args.dataset, args.seed)
    res = run_one(data, args.seed, args.criterion, args.sparsity, args.epochs,
                  schedule=args.schedule, regrow=args.regrow, ema=args.ema, log=print)
    row = res["row"]
    print(f"\nachieved sparsity {row['achieved_sparsity']:.4f} "
          f"({row['active_weights']} weights left)  test acc {row['test_acc']:.4f}")

    name = f"part3_{args.criterion}_s{int(args.sparsity * 100)}{args.tag}"
    hist = res["history"]
    write_csv(out(name + "_curve.csv"), hist)
    write_csv(out(name + "_events.csv"), res["pruner"].log)

    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ep, [h["val_acc"] for h in hist], label="val acc")
    ax.plot(ep, [h["train_acc"] for h in hist], label="train acc")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_ylim(0.9, 1.0)
    ax2 = ax.twinx()
    ax2.plot(ep, [h["sparsity"] for h in hist], "k--", label="sparsity")
    ax2.set_ylabel("sparsity")
    fig.legend(loc="lower center", ncol=3, fontsize=8)
    ax.set_title(f"{args.criterion}, target {args.sparsity:.0%}, {args.schedule}")
    fig.tight_layout()
    fig.savefig(out(name + ".png"), dpi=130)


if __name__ == "__main__":
    main()
