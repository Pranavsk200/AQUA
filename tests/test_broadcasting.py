import numpy as np

from engine import Tensor, ops
from engine.gradcheck import gradcheck

rng = np.random.default_rng(1)


def test_matrix_plus_row_vector():
    gradcheck(lambda x, b: x + b, rng.normal(size=(5, 4)), rng.normal(size=(4,)))


def test_row_vector_plus_matrix():
    gradcheck(lambda b, x: b + x, rng.normal(size=(4,)), rng.normal(size=(5, 4)))


def test_column_times_row():
    gradcheck(lambda a, b: a * b, rng.normal(size=(5, 1)), rng.normal(size=(1, 4)))


def test_scalar_tensor_with_matrix():
    gradcheck(lambda s, x: s * x, rng.normal(size=()), rng.normal(size=(3, 4)))


def test_div_broadcast():
    gradcheck(lambda x, d: x / d, rng.normal(size=(5, 4)), rng.uniform(1, 2, size=(4,)))


def test_sub_broadcast_three_dims():
    gradcheck(lambda x, b: x - b, rng.normal(size=(2, 3, 4)), rng.normal(size=(3, 1)))


def test_python_scalar_operands():
    x = Tensor(rng.normal(size=(3, 3)), requires_grad=True)
    y = ((2.0 * x + 1.0) / 4.0 - 3.0).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, np.full((3, 3), 0.5))


def test_reused_node_accumulates():
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    (x * x + x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)


def test_diamond_graph():
    x = Tensor(np.array([2.0]), requires_grad=True)
    a = x * 3.0
    b = x * x
    (a + b + a * b).sum().backward()
    # d/dx (3x + x^2 + 3x^3) = 3 + 2x + 9x^2
    np.testing.assert_allclose(x.grad, [3 + 4 + 36])


def test_bias_grad_is_sum_over_batch():
    n, d = 7, 3
    x = Tensor(rng.normal(size=(n, d)))
    b = Tensor(np.zeros(d), requires_grad=True)
    (x + b).sum().backward()
    np.testing.assert_allclose(b.grad, np.full(d, float(n)))


def test_deep_chain_does_not_recurse():
    x = Tensor(np.array([1.0]), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [1.0])
