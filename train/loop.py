import numpy as np

from engine import Tensor, no_grad, ops
from train.data import batches


def evaluate(model, X, y, batch_size=1000):
    loss, correct = 0.0, 0
    with no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(Tensor(X[i:i + batch_size]))
            loss += ops.cross_entropy(logits, y[i:i + batch_size]).item() * len(logits.data)
            correct += int((logits.data.argmax(axis=1) == y[i:i + batch_size]).sum())
    return loss / len(X), correct / len(X)


def fit(model, opt, data, epochs, batch_size, seed, pruner=None, log=print):
    rng = np.random.default_rng(seed)
    history = []
    step = 0
    for epoch in range(1, epochs + 1):
        run_loss, n = 0.0, 0
        for xb, yb in batches(data["X_train"], data["y_train"], batch_size, rng):
            opt.zero_grad()
            loss = ops.cross_entropy(model(Tensor(xb)), yb)
            if not np.isfinite(loss.data):
                raise FloatingPointError(f"loss went non-finite at step {step}")
            loss.backward()
            if pruner is not None:
                pruner.observe()
            opt.step()
            if pruner is not None:
                pruner.maybe_update(step)
            run_loss += loss.item() * len(xb)
            n += len(xb)
            step += 1

        tr_loss, tr_acc = evaluate(model, data["X_train"], data["y_train"])
        va_loss, va_acc = evaluate(model, data["X_val"], data["y_val"])
        row = {"epoch": epoch, "step": step, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": va_loss, "val_acc": va_acc}
        if pruner is not None:
            row["sparsity"] = pruner.current_sparsity()
        history.append(row)
        if log:
            extra = f" sparsity {row['sparsity']:.3f}" if pruner is not None else ""
            log(f"epoch {epoch:3d}  loss {tr_loss:.4f}  acc {tr_acc:.4f}  "
                f"val_loss {va_loss:.4f}  val_acc {va_acc:.4f}{extra}")
    return history
