import numpy as np
import pytest

from engine import ops
from engine.gradcheck import gradcheck

rng = np.random.default_rng(0)


def away_from_zero(shape, gap=0.05):
    x = rng.normal(size=shape)
    return np.where(np.abs(x) < gap, gap * np.sign(x + 1e-12) * 2, x)


A = rng.normal(size=(4, 3))
B = rng.normal(size=(4, 3))
POS = rng.uniform(0.5, 2.0, size=(4, 3))


def test_add():
    gradcheck(ops.add, A, B)


def test_sub():
    gradcheck(ops.sub, A, B)


def test_mul():
    gradcheck(ops.mul, A, B)


def test_div():
    gradcheck(ops.div, A, POS)


def test_neg():
    gradcheck(ops.neg, A)


def test_matmul():
    gradcheck(ops.matmul, rng.normal(size=(4, 3)), rng.normal(size=(3, 5)))


def test_matmul_batched_broadcast():
    gradcheck(ops.matmul, rng.normal(size=(2, 4, 3)), rng.normal(size=(3, 5)))


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, False),
                                           (0, True), (1, True), ((0, 1), False)])
def test_sum(axis, keepdims):
    gradcheck(lambda x: ops.sum(x, axis=axis, keepdims=keepdims), A)


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, False),
                                           (0, True), (1, True), ((0, 1), False)])
def test_mean(axis, keepdims):
    gradcheck(lambda x: ops.mean(x, axis=axis, keepdims=keepdims), A)


def test_relu():
    gradcheck(ops.relu, away_from_zero((5, 4)))


def test_tanh():
    gradcheck(ops.tanh, A)


def test_sigmoid():
    gradcheck(ops.sigmoid, A)


def test_gelu():
    gradcheck(ops.gelu, A)


def test_log_softmax():
    gradcheck(ops.log_softmax, A)


def test_softmax():
    gradcheck(ops.softmax, A)


def test_cross_entropy():
    targets = np.array([0, 2, 1, 2])
    gradcheck(lambda z: ops.cross_entropy(z, targets), A)


def test_cross_entropy_extreme_logits_stay_finite():
    from engine import Tensor
    z = Tensor(np.array([[1000.0, -1000.0, 0.0], [-500.0, 500.0, 3.0]]), requires_grad=True)
    loss = ops.cross_entropy(z, np.array([1, 1]))
    loss.backward()
    assert np.isfinite(loss.data) and np.isfinite(z.grad).all()


def test_sigmoid_extreme_inputs_stay_finite():
    from engine import Tensor
    y = ops.sigmoid(Tensor(np.array([-800.0, 0.0, 800.0])))
    assert np.isfinite(y.data).all()
    assert y.data[0] == 0.0 and y.data[2] == 1.0


def test_masked_mul():
    mask = (rng.uniform(size=(4, 3)) > 0.5).astype(np.float64)
    gradcheck(lambda w: ops.masked_mul(w, mask), A)


def test_composed_graph():
    w1, w2 = rng.normal(size=(3, 5)), rng.normal(size=(5, 2))
    x = rng.normal(size=(6, 3))
    t = np.array([0, 1, 1, 0, 1, 0])

    def net(a, b):
        from engine import Tensor
        h = ops.tanh(Tensor(x) @ a)
        return ops.cross_entropy(h @ b, t)

    gradcheck(net, w1, w2)
