import numpy as np
import pytest

from nn.layers import make_mlp, mask_list, masked_layers
from prune.masking import SelfPruner
from prune.schedule import cubic_plan, oneshot_plan
from train.data import load_dataset
from train.loop import fit
from train.optim import Adam


def make_pruner(criterion="taylor1", plan=None, seed=0, sizes=(20, 16, 10), **kw):
    rng = np.random.default_rng(seed)
    model = make_mlp(list(sizes), rng, masked=True)
    layers = masked_layers(model)
    pruner = SelfPruner(layers, criterion, plan or {0: 0.5}, seed=seed, **kw)
    for gb, g2, l in zip(pruner.gbar, pruner.g2bar, layers):
        gb[...] = rng.normal(size=gb.shape).astype(gb.dtype)
        g2[...] = gb ** 2
    return model, layers, pruner


def n_active(layers):
    return sum(int(l.mask.sum()) for l in layers)


def test_cubic_plan_endpoints_and_monotone():
    plan = cubic_plan(0.9, start=100, end=1000, every=50)
    steps = sorted(plan)
    vals = [plan[s] for s in steps]
    assert steps[0] == 100 and steps[-1] == 1000
    assert vals[0] == pytest.approx(0.0) and vals[-1] == 0.9
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # cubic means the first half of the ramp does most of the pruning
    assert plan[min(steps, key=lambda s: abs(s - 550))] > 0.9 * 0.8


def test_oneshot_plan():
    assert oneshot_plan(0.8, 300) == {300: 0.8}


@pytest.mark.parametrize("criterion", ["magnitude", "taylor1", "taylor2", "random"])
@pytest.mark.parametrize("target", [0.5, 0.75, 0.9, 0.95])
def test_exact_sparsity_budget(criterion, target):
    model, layers, pruner = make_pruner(criterion, {0: target})
    pruner.maybe_update(0)
    total = sum(l.weight.data.size for l in layers)
    assert n_active(layers) == round((1 - target) * total)
    for l in layers:
        assert np.all(l.weight.data[l.mask == 0] == 0.0)


def test_ties_do_not_break_the_budget():
    model, layers, pruner = make_pruner("magnitude", {0: 0.7})
    for l in layers:
        l.weight.data[...] = 0.5
    pruner.maybe_update(0)
    total = sum(l.weight.data.size for l in layers)
    assert n_active(layers) == round(0.3 * total)


def test_pruning_is_deterministic():
    _, la, pa = make_pruner("taylor1", {0: 0.8}, seed=3)
    _, lb, pb = make_pruner("taylor1", {0: 0.8}, seed=3)
    pa.maybe_update(0)
    pb.maybe_update(0)
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(x.mask, y.mask)


def test_keeps_the_highest_scoring_weights():
    model, layers, pruner = make_pruner("magnitude", {0: 0.9}, min_density=0.0, normalize=False)
    allw = np.concatenate([np.abs(l.weight.data).ravel() for l in layers])
    pruner.maybe_update(0)
    k = n_active(layers)
    kept = np.concatenate([np.abs(l.weight.data[l.mask > 0]) for l in layers])
    assert kept.min() >= np.sort(allw)[::-1][k]


def test_layer_floor_prevents_collapse():
    model, layers, pruner = make_pruner("magnitude", {0: 0.95}, min_density=0.05, normalize=False)
    layers[0].weight.data *= 1e-3
    pruner.maybe_update(0)
    for l in layers:
        assert l.mask.sum() >= np.ceil(0.05 * l.mask.size)


def test_taylor_differs_from_magnitude():
    _, la, pa = make_pruner("magnitude", {0: 0.8})
    _, lb, pb = make_pruner("taylor1", {0: 0.8})
    pa.maybe_update(0)
    pb.maybe_update(0)
    assert any((x.mask != y.mask).any() for x, y in zip(la, lb))


def test_regrow_picks_largest_dense_gradient_and_keeps_budget():
    model, layers, pruner = make_pruner("magnitude", {0: 0.8, 250: 0.8, 500: 0.8}, regrow=0.5,
                                        cooldown=0, normalize=False, min_density=0.0)
    pruner.maybe_update(0)
    off_before = [l.mask == 0 for l in layers]
    for gb, off in zip(pruner.gbar, off_before):
        gb[off] = 0.0
    big = layers[0]
    spot = np.argwhere(big.mask == 0)[0]
    pruner.gbar[0][tuple(spot)] = 1e3
    pruner.maybe_update(250)
    assert big.mask[tuple(spot)] == 1
    assert big.weight.data[tuple(spot)] == 0.0
    total = sum(l.weight.data.size for l in layers)
    assert n_active(layers) == round(0.2 * total)


def test_cooldown_blocks_immediate_regrowth():
    plan = {0: 0.8, 50: 0.8, 1000: 0.8}
    model, layers, pruner = make_pruner("magnitude", plan, regrow=0.5, cooldown=100,
                                        normalize=False, min_density=0.0)
    pruner.maybe_update(0)
    just_pruned = [l.mask == 0 for l in layers]
    for gb in pruner.gbar:
        gb[...] = 1e3
    pruner.maybe_update(50)
    for l, was_off in zip(layers, just_pruned):
        assert np.all(l.mask[was_off] == 0)


def test_grace_protects_new_connections():
    plan = {0: 0.5, 100: 0.5, 150: 0.5, 5000: 0.5}
    model, layers, pruner = make_pruner("taylor1", plan, regrow=0.3, cooldown=0, grace=1000,
                                        normalize=False, min_density=0.0)
    pruner.maybe_update(0)
    pruner.maybe_update(100)
    grown = [g == 100 for g in pruner.grown_at]
    assert sum(int(g.sum()) for g in grown) > 0
    for l in layers:
        l.weight.data[...] = 0.0
    pruner.maybe_update(150)
    for l, g in zip(layers, grown):
        assert np.all(l.mask[g] == 1)


def test_end_to_end_run_hits_target_with_clean_state():
    data = load_dataset("digits", seed=0)
    rng = np.random.default_rng(0)
    model = make_mlp([64, 32, 10], rng, masked=True)
    params = model.parameters()
    opt = Adam(params, lr=3e-3, masks=mask_list(model, params))
    plan = cubic_plan(0.9, start=20, end=200, every=20)
    pruner = SelfPruner(masked_layers(model), "taylor1", plan, seed=0, regrow=0.2,
                        cooldown=40, grace=40, optimizer=opt)
    hist = fit(model, opt, data, epochs=8, batch_size=32, seed=0, pruner=pruner, log=None)

    layers = masked_layers(model)
    total = sum(l.weight.data.size for l in layers)
    assert n_active(layers) == round(0.1 * total)
    for l, m, v in zip(layers, [opt.m[0], opt.m[2]], [opt.v[0], opt.v[2]]):
        off = l.mask == 0
        assert np.all(l.weight.data[off] == 0.0)
        assert np.all(m[off] == 0.0) and np.all(v[off] == 0.0)
    assert hist[-1]["val_acc"] > 0.8
