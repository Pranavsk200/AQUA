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
