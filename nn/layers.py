import numpy as np

from engine import Tensor, ops
from nn.init import INITS


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    def __init__(self, fan_in, fan_out, rng, init="relu", dtype=np.float32):
        self.weight = Tensor(INITS[init](fan_in, fan_out, rng, dtype), requires_grad=True)
        self.bias = Tensor(np.zeros(fan_out, dtype=dtype), requires_grad=True)

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        return x @ self.weight + self.bias


class MaskedLinear(Linear):
    def __init__(self, fan_in, fan_out, rng, init="relu", dtype=np.float32):
        super().__init__(fan_in, fan_out, rng, init, dtype)
        self.mask = np.ones_like(self.weight.data)
        self.w_eff = None

    def forward(self, x):
        self.w_eff = ops.masked_mul(self.weight, self.mask)
        return x @ self.w_eff + self.bias

    def dense_grad(self):
        # dL/dw_eff: what the loss would do if each connection were switched on.
        # Never fed to the optimizer, only used to score regrowth.
        return None if self.w_eff is None else self.w_eff.grad

    def apply_mask(self):
        self.weight.data *= self.mask


class Activation(Module):
    def __init__(self, name):
        self.name = name
        self.fn = getattr(ops, name)

    def forward(self, x):
        return self.fn(x)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_mlp(sizes, rng, activation="relu", masked=False, dtype=np.float32):
    cls = MaskedLinear if masked else Linear
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        # logits layer feeds a softmax, no gain from the relu correction there
        layers.append(cls(a, b, rng, init="tanh" if last else activation, dtype=dtype))
        if not last:
            layers.append(Activation(activation))
    return Sequential(*layers)


def masked_layers(model):
    return [l for l in model.layers if isinstance(l, MaskedLinear)]
