import numpy as np

from engine import Tensor, ops
from engine.gradcheck import gradcheck
from nn.layers import MaskedLinear, make_mlp, masked_layers
from train.optim import SGD, Adam

rng = np.random.default_rng(3)


class NaiveAdam:
    # the usual mistake: zero the gradient of masked weights and call it a day
    def __init__(self, params, masks, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.masks = params, masks
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for p, m, v, mask in zip(self.params, self.m, self.v, self.masks):
            g = p.grad * mask
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            p.data -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


def tiny_problem(dtype=np.float64):
    layer = MaskedLinear(6, 4, np.random.default_rng(0), dtype=dtype)
    x = np.random.default_rng(1).normal(size=(16, 6)).astype(dtype)
    y = np.random.default_rng(2).integers(0, 4, size=16)
    return layer, x, y


def run_steps(layer, opt, x, y, n):
    for _ in range(n):
        layer.zero_grad()
        ops.cross_entropy(layer(Tensor(x)), y).backward()
        opt.step()


def random_mask(shape, keep=0.5, seed=5):
    return (np.random.default_rng(seed).uniform(size=shape) < keep).astype(np.float64)


def test_masked_forward_equals_physically_deleted_weights():
    layer, x, _ = tiny_problem()
    layer.mask[:] = random_mask(layer.weight.data.shape)
    out = layer(Tensor(x)).data
    pruned = layer.weight.data.copy()
    pruned[layer.mask == 0] = 0.0
    np.testing.assert_allclose(out, x @ pruned + layer.bias.data)


def test_masked_forward_ignores_stale_weight_values():
    # the underlying weight may hold garbage under the mask, it must not leak into the output
    layer, x, _ = tiny_problem()
    layer.mask[:] = random_mask(layer.weight.data.shape)
    clean = layer(Tensor(x)).data.copy()
    layer.weight.data[layer.mask == 0] = 1e6
    np.testing.assert_allclose(layer(Tensor(x)).data, clean)


def test_param_grad_is_exactly_zero_where_masked_and_dense_grad_is_not():
    layer, x, y = tiny_problem()
    layer.mask[:] = random_mask(layer.weight.data.shape)
    ops.cross_entropy(layer(Tensor(x)), y).backward()
    off = layer.mask == 0
    assert np.all(layer.weight.grad[off] == 0.0)
    assert np.all(layer.dense_grad()[off] != 0.0)


def test_dense_grad_matches_finite_differences_at_zero_weight():
    # the regrowth signal must be the true derivative wrt the effective weight
    layer, x, y = tiny_problem()
    mask = random_mask(layer.weight.data.shape)
    w_eff0 = layer.weight.data * mask
    bias = layer.bias.data

    def loss(w_eff):
        return ops.cross_entropy(Tensor(x) @ w_eff + Tensor(bias), y)

    layer.mask[:] = mask
    ops.cross_entropy(layer(Tensor(x)), y).backward()
    t = Tensor(w_eff0.copy(), requires_grad=True)
    loss(t).backward()
    np.testing.assert_allclose(layer.dense_grad(), t.grad, atol=1e-12)
    gradcheck(loss, w_eff0)


def test_adam_keeps_masked_weights_at_exact_zero_with_clean_state():
    layer, x, y = tiny_problem()
    opt = Adam(layer.parameters(), lr=1e-2, masks=[layer.mask, None])
    run_steps(layer, opt, x, y, 20)

    layer.mask[:] = random_mask(layer.weight.data.shape)
    layer.apply_mask()
    opt.apply_masks()
    run_steps(layer, opt, x, y, 50)

    off = layer.mask == 0
    assert np.all(layer.weight.data[off] == 0.0)
    assert np.all(opt.m[0][off] == 0.0) and np.all(opt.v[0][off] == 0.0)
    assert np.abs(layer.weight.data[~off]).max() > 0


def test_naive_adam_drifts_masked_weights():
    layer, x, y = tiny_problem()
    opt = NaiveAdam(layer.parameters(), masks=[layer.mask, np.ones_like(layer.bias.data)], lr=1e-2)
    run_steps(layer, opt, x, y, 20)

    layer.mask[:] = random_mask(layer.weight.data.shape)
    layer.apply_mask()
    run_steps(layer, opt, x, y, 50)

    off = layer.mask == 0
    assert np.abs(layer.weight.data[off]).max() > 1e-4


def test_sgd_momentum_keeps_masked_weights_at_zero():
    layer, x, y = tiny_problem()
    opt = SGD(layer.parameters(), lr=0.05, momentum=0.9, masks=[layer.mask, None])
    run_steps(layer, opt, x, y, 20)
    layer.mask[:] = random_mask(layer.weight.data.shape)
    layer.apply_mask()
    opt.apply_masks()
    run_steps(layer, opt, x, y, 30)
    off = layer.mask == 0
    assert np.all(layer.weight.data[off] == 0.0) and np.all(opt.v[0][off] == 0.0)


def test_revived_connection_starts_from_clean_state():
    layer, x, y = tiny_problem()
    opt = Adam(layer.parameters(), lr=1e-2, masks=[layer.mask, None])
    run_steps(layer, opt, x, y, 20)

    layer.mask[:] = random_mask(layer.weight.data.shape)
    layer.apply_mask()
    opt.apply_masks()
    run_steps(layer, opt, x, y, 20)

    revived = (layer.mask == 0)
    layer.mask[:] = 1.0
    opt.t = 5000
    assert np.all(layer.weight.data[revived] == 0.0)
    assert np.all(opt.m[0][revived] == 0.0) and np.all(opt.v[0][revived] == 0.0)

    before = layer.weight.data.copy()
    layer.zero_grad()
    ops.cross_entropy(layer(Tensor(x)), y).backward()
    g = layer.weight.grad
    opt.step()
    moved = layer.weight.data - before
    # fresh moments: the step follows -sign(g). With a late global t the bias correction is ~1,
    # so the size is 0.1/sqrt(0.001) = 3.16 * lr, not lr. Stale momentum would not look like this.
    nz = revived & (g != 0)
    assert np.all(np.sign(moved[nz]) == -np.sign(g[nz]))
    c1, c2 = 1 - 0.9 ** opt.t, 1 - 0.999 ** opt.t
    expected = 1e-2 * (0.1 / c1) / np.sqrt(0.001 / c2)
    np.testing.assert_allclose(np.abs(moved[nz]), expected, rtol=1e-6)
    assert expected > 3 * 1e-2


def test_full_mlp_forward_backward_with_masks_has_no_nans():
    model = make_mlp([10, 8, 3], np.random.default_rng(0), masked=True, dtype=np.float32)
    for layer in masked_layers(model):
        layer.mask[:] = random_mask(layer.weight.data.shape, keep=0.2).astype(np.float32)
        layer.apply_mask()
    x = np.random.default_rng(1).normal(size=(5, 10)).astype(np.float32)
    ops.cross_entropy(model(Tensor(x)), np.array([0, 1, 2, 0, 1])).backward()
    for p in model.parameters():
        assert np.isfinite(p.grad).all()
