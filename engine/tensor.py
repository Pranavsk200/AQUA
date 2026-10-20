from contextlib import contextmanager

import numpy as np

_grad_enabled = True


@contextmanager
def no_grad():
    global _grad_enabled
    prev, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = prev


def grad_enabled():
    return _grad_enabled


def unbroadcast(grad, shape):
    # undo numpy broadcasting: sum out the axes that were added or stretched
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and grad.shape[ax] != 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, parents=(), op=""):
        data = np.asarray(data)
        if data.dtype.kind != "f":
            data = data.astype(np.float64)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = parents
        self._backward = None
        self.op = op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self.op!r}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        self.grad = None

    def item(self):
        return float(self.data)

    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be given for non-scalar output")
            grad = np.ones_like(self.data)
        self.grad = np.asarray(grad, dtype=self.data.dtype)

        for node in reversed(_toposort(self)):
            if node._backward is None or node.grad is None:
                continue
            grads = node._backward(node.grad)
            for parent, g in zip(node._parents, grads):
                if g is None or not parent.requires_grad:
                    continue
                parent.grad = g if parent.grad is None else parent.grad + g


def _toposort(root):
    # iterative on purpose, deep graphs would blow the recursion limit
    order, seen = [], set()
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for p in node._parents:
            if id(p) not in seen:
                stack.append((p, False))
    return order
