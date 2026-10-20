import numpy as np

from engine.tensor import Tensor, grad_enabled, unbroadcast


def _wrap(x, like):
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x, dtype=like.data.dtype))


def _make(data, parents, op, backward):
    needs = grad_enabled() and any(p.requires_grad for p in parents)
    if not needs:
        return Tensor(data, op=op)
    out = Tensor(data, requires_grad=True, parents=tuple(parents), op=op)
    out._backward = backward
    return out


def add(a, b):
    b = _wrap(b, a)
    return _make(a.data + b.data, (a, b), "add",
                 lambda g: (unbroadcast(g, a.shape), unbroadcast(g, b.shape)))


def sub(a, b):
    b = _wrap(b, a)
    return _make(a.data - b.data, (a, b), "sub",
                 lambda g: (unbroadcast(g, a.shape), unbroadcast(-g, b.shape)))


def mul(a, b):
    b = _wrap(b, a)
    return _make(a.data * b.data, (a, b), "mul",
                 lambda g: (unbroadcast(g * b.data, a.shape), unbroadcast(g * a.data, b.shape)))


def div(a, b):
    b = _wrap(b, a)
    return _make(a.data / b.data, (a, b), "div",
                 lambda g: (unbroadcast(g / b.data, a.shape),
                            unbroadcast(-g * a.data / (b.data ** 2), b.shape)))


def neg(a):
    return _make(-a.data, (a,), "neg", lambda g: (-g,))


def _rsub(a, b):
    return sub(_wrap(b, a), a)


def _rdiv(a, b):
    return div(_wrap(b, a), a)


Tensor.__add__ = Tensor.__radd__ = add
Tensor.__sub__ = sub
Tensor.__rsub__ = _rsub
Tensor.__mul__ = Tensor.__rmul__ = mul
Tensor.__truediv__ = div
Tensor.__rtruediv__ = _rdiv
Tensor.__neg__ = neg


def matmul(a, b):
    b = _wrap(b, a)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("matmul expects at least 2-D operands")

    def backward(g):
        ga = g @ np.swapaxes(b.data, -1, -2)
        gb = np.swapaxes(a.data, -1, -2) @ g
        return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)

    return _make(a.data @ b.data, (a, b), "matmul", backward)


def _expand(g, shape, axis, keepdims):
    if axis is not None and not keepdims:
        g = np.expand_dims(g, axis)
    return np.broadcast_to(g, shape).copy()


def sum(a, axis=None, keepdims=False):
    return _make(a.data.sum(axis=axis, keepdims=keepdims), (a,), "sum",
                 lambda g: (_expand(g, a.shape, axis, keepdims),))


def mean(a, axis=None, keepdims=False):
    if axis is None:
        count = a.data.size
    else:
        axes = (axis,) if isinstance(axis, int) else axis
        count = int(np.prod([a.shape[i] for i in axes]))
    return _make(a.data.mean(axis=axis, keepdims=keepdims), (a,), "mean",
                 lambda g: (_expand(g, a.shape, axis, keepdims) / count,))


def relu(a):
    mask = a.data > 0
    return _make(a.data * mask, (a,), "relu", lambda g: (g * mask,))


def tanh(a):
    y = np.tanh(a.data)
    return _make(y, (a,), "tanh", lambda g: (g * (1 - y * y),))


def sigmoid(a):
    # exp(-|x|) keeps the exponent <= 0 so it never overflows
    e = np.exp(-np.abs(a.data))
    y = np.where(a.data >= 0, 1 / (1 + e), e / (1 + e))
    return _make(y, (a,), "sigmoid", lambda g: (g * y * (1 - y),))


_GELU_C = np.sqrt(2 / np.pi)


def gelu(a):
    x = a.data
    t = np.tanh(_GELU_C * (x + 0.044715 * x ** 3))

    def backward(g):
        dt = (1 - t * t) * _GELU_C * (1 + 3 * 0.044715 * x * x)
        return (g * (0.5 * (1 + t) + 0.5 * x * dt),)

    return _make(0.5 * x * (1 + t), (a,), "gelu", backward)


def _log_softmax(x, axis):
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def log_softmax(a, axis=-1):
    out = _log_softmax(a.data, axis)
    p = np.exp(out)
    return _make(out, (a,), "log_softmax",
                 lambda g: (g - p * g.sum(axis=axis, keepdims=True),))


def softmax(a, axis=-1):
    p = np.exp(_log_softmax(a.data, axis))
    return _make(p, (a,), "softmax",
                 lambda g: (p * (g - (g * p).sum(axis=axis, keepdims=True)),))


def cross_entropy(logits, targets):
    # fused from raw logits, the gradient collapses to (softmax - onehot) / N
    targets = np.asarray(targets)
    n = logits.shape[0]
    logp = _log_softmax(logits.data, axis=1)
    loss = -logp[np.arange(n), targets].mean()

    def backward(g):
        grad = np.exp(logp)
        grad[np.arange(n), targets] -= 1
        return (g * grad / n,)

    return _make(np.asarray(loss, dtype=logits.data.dtype), (logits,), "cross_entropy", backward)


def masked_mul(w, mask):
    # mask is a constant 0/1 array, not part of the graph
    return _make(w.data * mask, (w,), "masked_mul", lambda g: (g * mask,))


Tensor.__matmul__ = matmul
Tensor.sum = sum
Tensor.mean = mean
Tensor.relu = relu
Tensor.tanh = tanh
Tensor.sigmoid = sigmoid
