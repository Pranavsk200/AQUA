import numpy as np


class Optimizer:
    def __init__(self, params, masks=None):
        self.params = list(params)
        # masks[i] is the live 0/1 array of params[i] (or None); it is shared with
        # the layer so prune/regrow events are seen here immediately
        self.masks = masks if masks is not None else [None] * len(self.params)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def apply_masks(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, masks=None):
        super().__init__(params, masks)
        self.lr, self.momentum = lr, momentum
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v, mask in zip(self.params, self.v, self.masks):
            if p.grad is None:
                continue
            g = p.grad if mask is None else p.grad * mask
            v *= self.momentum
            v += g
            if mask is not None:
                v *= mask
            p.data -= self.lr * v

    def apply_masks(self):
        for v, mask in zip(self.v, self.masks):
            if mask is not None:
                v *= mask


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, masks=None):
        super().__init__(params, masks)
        self.lr, self.eps = lr, eps
        self.b1, self.b2 = betas
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for p, m, v, mask in zip(self.params, self.m, self.v, self.masks):
            if p.grad is None:
                continue
            g = p.grad if mask is None else p.grad * mask
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            if mask is not None:
                # a zero gradient alone doesn't freeze a weight, stale m and v
                # would keep pushing it. Clearing the state makes the update 0/(0+eps) = 0.
                m *= mask
                v *= mask
            p.data -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

    def apply_masks(self):
        for m, v, mask in zip(self.m, self.v, self.masks):
            if mask is not None:
                m *= mask
                v *= mask
