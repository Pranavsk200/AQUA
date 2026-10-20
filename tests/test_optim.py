import numpy as np

from engine import Tensor
from train.optim import SGD, Adam


def test_adam_matches_hand_computed_steps():
    p = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    grads = [np.array([0.5, -1.0]), np.array([0.25, 0.5])]

    w = p.data.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    for t, g in enumerate(grads, start=1):
        p.grad = g.copy()
        opt.step()
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - 0.1 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    np.testing.assert_allclose(p.data, w, rtol=1e-12)


def test_first_adam_step_is_lr_sized():
    p = Tensor(np.array([3.0]), requires_grad=True)
    opt = Adam([p], lr=0.01)
    p.grad = np.array([1234.5])
    opt.step()
    np.testing.assert_allclose(p.data, [3.0 - 0.01], atol=1e-6)


def test_sgd_momentum_hand_computed():
    p = Tensor(np.array([1.0]), requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = np.array([1.0])
    opt.step()
    p.grad = np.array([1.0])
    opt.step()
    # v1 = 1, v2 = 1.9
    np.testing.assert_allclose(p.data, [1.0 - 0.1 - 0.19])


def test_adam_minimizes_quadratic():
    p = Tensor(np.array([5.0, -3.0]), requires_grad=True)
    opt = Adam([p], lr=0.1)
    for _ in range(500):
        p.grad = 2 * p.data
        opt.step()
    assert np.abs(p.data).max() < 1e-2
