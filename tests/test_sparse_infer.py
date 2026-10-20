import numpy as np

from engine import Tensor, no_grad
from nn.layers import make_mlp, masked_layers
from prune.sparse_infer import CSR, dense_net, sparse_net

rng = np.random.default_rng(7)


def test_csr_matches_dense_matmul():
    w = rng.normal(size=(30, 20)).astype(np.float32)
    mask = (rng.uniform(size=w.shape) < 0.1).astype(np.float32)
    x = rng.normal(size=(8, 30)).astype(np.float32)
    np.testing.assert_allclose(CSR(w, mask).matmul(x), x @ (w * mask), rtol=1e-5, atol=1e-5)


def test_csr_handles_empty_output_rows():
    w = rng.normal(size=(12, 6)).astype(np.float32)
    mask = np.zeros_like(w)
    mask[:, 1] = 1
    mask[3, 4] = 1
    x = rng.normal(size=(5, 12)).astype(np.float32)
    got = CSR(w, mask).matmul(x)
    np.testing.assert_allclose(got, x @ (w * mask), rtol=1e-5, atol=1e-6)
    assert np.all(got[:, [0, 2, 3, 5]] == 0)


def test_csr_all_zero_matrix():
    w = np.zeros((4, 3), dtype=np.float32)
    assert np.all(CSR(w).matmul(np.ones((2, 4), dtype=np.float32)) == 0)


def test_csr_single_sample_batch():
    w = rng.normal(size=(9, 7)).astype(np.float32)
    mask = (rng.uniform(size=w.shape) < 0.3).astype(np.float32)
    x = rng.normal(size=(1, 9)).astype(np.float32)
    np.testing.assert_allclose(CSR(w, mask).matmul(x), x @ (w * mask), rtol=1e-5, atol=1e-5)


def test_sparse_network_matches_masked_model_forward():
    model = make_mlp([40, 24, 16, 10], np.random.default_rng(0), masked=True)
    for l in masked_layers(model):
        l.mask[...] = (rng.uniform(size=l.mask.shape) < 0.1).astype(np.float32)
        l.apply_mask()
    x = rng.normal(size=(11, 40)).astype(np.float32)
    with no_grad():
        want = model(Tensor(x)).data
    np.testing.assert_allclose(sparse_net(model)(x), want, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dense_net(model)(x), want, rtol=1e-4, atol=1e-5)


def test_flop_count_scales_with_nonzeros():
    model = make_mlp([40, 24, 10], np.random.default_rng(0), masked=True)
    dense = dense_net(model).flops_per_sample()
    for l in masked_layers(model):
        l.mask[...] = (rng.uniform(size=l.mask.shape) < 0.1).astype(np.float32)
        l.apply_mask()
    nnz = sum(int(l.mask.sum()) for l in masked_layers(model))
    assert sparse_net(model).flops_per_sample() == 2 * nnz
    assert dense == 2 * (40 * 24 + 24 * 10)
