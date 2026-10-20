import numpy as np

from experiments.common import out, write_csv
from experiments.part4_sweep import LEVELS, model_path
from nn.layers import load_model
from prune.sparse_infer import dense_net, sparse_net, time_forward
from train.data import load_dataset

BATCHES = [1, 64, 512]


def main(method="ours", seed=0):
    data = load_dataset("mnist", seed)
    X, y = data["X_test"], data["y_test"]
    rows = []
    for level in [0.0] + LEVELS:
        variant, lvl = ("dense", 0) if level == 0 else (method, level)
        model = load_model(model_path(variant, lvl, seed))
        dense, sparse = dense_net(model), sparse_net(model)

        acc_dense = float((dense(X).argmax(1) == y).mean())
        acc_sparse = float((sparse(X).argmax(1) == y).mean())
        row = {
            "sparsity": level,
            "active_weights": sparse.weights_stored(),
            "dense_flops_per_sample": dense.flops_per_sample(),
            "sparse_flops_per_sample": sparse.flops_per_sample(),
            "flop_reduction_x": dense.flops_per_sample() / sparse.flops_per_sample(),
            "test_acc_dense_path": acc_dense,
            "test_acc_csr_path": acc_sparse,
        }
        for b in BATCHES:
            xb = X[:b]
            td, ts = time_forward(dense, xb) * 1e6, time_forward(sparse, xb) * 1e6
            row[f"dense_us_b{b}"] = td
            row[f"csr_us_b{b}"] = ts
            row[f"csr_speedup_b{b}"] = td / ts
        rows.append(row)
        print(f"sparsity {level:.2f}  weights {row['active_weights']:6d}  "
              f"FLOPs x{row['flop_reduction_x']:.1f}  acc {acc_dense:.4f}/{acc_sparse:.4f}  "
              + "  ".join(f"b{b}: dense {row[f'dense_us_b{b}']:.0f}us csr {row[f'csr_us_b{b}']:.0f}us" for b in BATCHES))
    write_csv(out("cost.csv"), rows)


if __name__ == "__main__":
    main()
