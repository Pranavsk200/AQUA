import numpy as np


def he_normal(fan_in, fan_out, rng, dtype=np.float32):
    # Var[y] = fan_in * Var[w] * E[x^2], and E[x^2] = Var[z]/2 after a relu,
    # so Var[w] = 2/fan_in keeps the pre-activation variance constant
    return (rng.normal(size=(fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(dtype)


def xavier_normal(fan_in, fan_out, rng, dtype=np.float32):
    return (rng.normal(size=(fan_in, fan_out)) * np.sqrt(2.0 / (fan_in + fan_out))).astype(dtype)


INITS = {"relu": he_normal, "gelu": he_normal, "tanh": xavier_normal, "sigmoid": xavier_normal}
