import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_mnist():
    cache = os.path.join(DATA_DIR, "mnist.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X"], z["y"]
    # sklearn is only used here, to download/parse the dataset. Nothing else from it.
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                        data_home=os.path.join(DATA_DIR, "openml"), return_X_y=True)
    X, y = X.astype(np.uint8), y.astype(np.int64)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def _load_digits():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.data.astype(np.float64), d.target.astype(np.int64)


def load_dataset(name="mnist", seed=0):
    if name == "mnist":
        X, y = _load_mnist()
        X = X.astype(np.float32) / 255.0
        # the standard split: first 60k train, last 10k test; hold 5k of train for validation
        order = np.random.default_rng(seed).permutation(60000)
        tr, va = order[:55000], order[55000:]
        parts = (X[tr], y[tr], X[va], y[va], X[60000:], y[60000:])
    elif name == "digits":
        X, y = _load_digits()
        X = (X / 16.0).astype(np.float32)
        order = np.random.default_rng(seed).permutation(len(X))
        a, b = int(0.7 * len(X)), int(0.85 * len(X))
        tr, va, te = order[:a], order[a:b], order[b:]
        parts = (X[tr], y[tr], X[va], y[va], X[te], y[te])
    else:
        raise ValueError(f"unknown dataset {name}")

    Xtr, ytr, Xva, yva, Xte, yte = parts
    mu, sd = Xtr.mean(), Xtr.std()
    norm = lambda a: ((a - mu) / sd).astype(np.float32)
    return {"X_train": norm(Xtr), "y_train": ytr, "X_val": norm(Xva), "y_val": yva,
            "X_test": norm(Xte), "y_test": yte, "n_features": Xtr.shape[1], "n_classes": 10}


def batches(X, y, batch_size, rng):
    order = rng.permutation(len(X))
    for i in range(0, len(X), batch_size):
        idx = order[i:i + batch_size]
        yield X[idx], y[idx]
