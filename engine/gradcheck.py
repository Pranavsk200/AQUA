import numpy as np

from engine.tensor import Tensor


def numerical_grad(fn, arrays, index, eps=1e-6):
    base = [a.copy() for a in arrays]
    grad = np.zeros_like(base[index])
    it = np.nditer(base[index], flags=["multi_index"], op_flags=["readwrite"])
    for _ in it:
        i = it.multi_index
        orig = base[index][i]
        base[index][i] = orig + eps
        plus = fn(*[Tensor(a) for a in base]).data
        base[index][i] = orig - eps
        minus = fn(*[Tensor(a) for a in base]).data
        base[index][i] = orig
        grad[i] = ((plus - minus) * _probe(plus.shape)).sum() / (2 * eps)
    return grad


def _probe(shape, seed=1234):
    # fixed random weights so the scalar loss doesn't have a uniform upstream gradient
    return np.random.default_rng(seed).normal(size=shape)


def gradcheck(fn, *arrays, eps=1e-6, tol=1e-5):
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*tensors)
    out.backward(_probe(out.shape))

    worst = 0.0
    for i, t in enumerate(tensors):
        analytic = t.grad if t.grad is not None else np.zeros_like(arrays[i])
        numeric = numerical_grad(fn, arrays, i, eps)
        denom = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-8)
        worst = max(worst, float((np.abs(analytic - numeric) / denom).max()))
    if worst > tol:
        raise AssertionError(f"gradcheck failed: max relative error {worst:.3e} > {tol:.0e}")
    return worst
