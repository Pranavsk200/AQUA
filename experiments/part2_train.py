import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import out, write_csv
from nn.layers import make_mlp
from train.data import load_dataset
from train.loop import evaluate, fit
from train.optim import Adam


def activation_variances(model, X):
    from engine import Tensor, no_grad
    stats = []
    with no_grad():
        h = Tensor(X)
        for layer in model.layers:
            h = layer(h)
            if layer.__class__.__name__ == "Linear":
                stats.append(float(h.data.var()))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_dataset(args.dataset, args.seed)
    rng = np.random.default_rng(args.seed)
    sizes = [data["n_features"], *args.hidden, data["n_classes"]]
    model = make_mlp(sizes, rng)
    print("pre-activation variance per layer at init:",
          [round(v, 3) for v in activation_variances(model, data["X_train"][:1000])])

    opt = Adam(model.parameters(), lr=args.lr)
    history = fit(model, opt, data, args.epochs, args.batch_size, args.seed)
    test_loss, test_acc = evaluate(model, data["X_test"], data["y_test"])
    print(f"test acc {test_acc:.4f}  test loss {test_loss:.4f}")

    write_csv(out(f"part2_curve_{args.dataset}.csv"), history)
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend()
    fig.suptitle(f"Dense MLP {sizes} on {args.dataset} (test acc {test_acc:.4f})")
    fig.tight_layout()
    fig.savefig(out(f"part2_curve_{args.dataset}.png"), dpi=130)


if __name__ == "__main__":
    main()
