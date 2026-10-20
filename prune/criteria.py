import numpy as np

# Removing weight w sets it to 0. With g = dL/dw and H_ii the curvature along w:
#   dL ~ -g*w + 0.5*H_ii*w^2
# magnitude only looks at |w|; the taylor criteria bring in what the loss actually says.


def magnitude(w, gbar, g2bar, rng=None):
    return np.abs(w)


def taylor1(w, gbar, g2bar, rng=None):
    # first-order term. gbar is an EMA over batches so it estimates the mean gradient
    return np.abs(w * gbar)


def taylor2(w, gbar, g2bar, rng=None):
    # adds the second-order term, with E[g^2] (diagonal Fisher) standing in for H_ii
    return np.abs(w * gbar) + 0.5 * g2bar * w * w


def random(w, gbar, g2bar, rng):
    return rng.uniform(size=w.shape)


CRITERIA = {"magnitude": magnitude, "taylor1": taylor1, "taylor2": taylor2, "random": random}
