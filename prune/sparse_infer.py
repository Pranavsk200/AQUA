import time

import numpy as np

from engine import Tensor, no_grad
from nn.layers import Activation


class CSR:
    # stores W^T (out, in): row r lists the inputs feeding output unit r
    def __init__(self, weight, mask=None):
        w = weight if mask is None else weight * mask
        wt = np.ascontiguousarray(w.T)
        rows, cols = np.nonzero(wt)
        self.shape = wt.shape
        self.data = wt[rows, cols]
        self.indices = cols
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=wt.shape[0]))])

    @property
    def nnz(self):
        return len(self.data)

    def matmul(self, x):
        # x: (batch, in) -> (batch, out)
        n_out = self.shape[0]
        out = np.zeros((n_out, x.shape[0]), dtype=x.dtype)
        if self.nnz == 0:
            return out.T
        prod = x.T[self.indices] * self.data[:, None]
        # reduceat misbehaves on empty segments, so only feed it the non-empty rows
        nonempty = np.flatnonzero(np.diff(self.indptr))
        out[nonempty] = np.add.reduceat(prod, self.indptr[nonempty], axis=0)
        return out.T


class _Net:
    def __init__(self, model, sparse):
        self.steps = []
        for layer in model.layers:
            if isinstance(layer, Activation):
                self.steps.append(("act", layer.name))
            else:
                w = layer.weight.data
                mask = getattr(layer, "mask", None)
                w = w if mask is None else w * mask
                self.steps.append(("lin", CSR(w) if sparse else w, layer.bias.data))

    def __call__(self, x):
        from engine import ops
        for step in self.steps:
            if step[0] == "act":
                with no_grad():
                    x = getattr(ops, step[1])(Tensor(x)).data
            else:
                _, w, b = step
                x = (w.matmul(x) if isinstance(w, CSR) else x @ w) + b
        return x

    def flops_per_sample(self):
        total = 0
        for step in self.steps:
            if step[0] == "lin":
                w = step[1]
                total += 2 * (w.nnz if isinstance(w, CSR) else w.size)
        return total

    def weights_stored(self):
        return sum(s[1].nnz if isinstance(s[1], CSR) else s[1].size for s in self.steps if s[0] == "lin")


def sparse_net(model):
    return _Net(model, sparse=True)


def dense_net(model):
    # same weights with the zeros left in: the "multiply by zero" shortcut
    return _Net(model, sparse=False)


def time_forward(net, x, repeats=30):
    net(x)
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        net(x)
        best = min(best, time.perf_counter() - t)
    return best

