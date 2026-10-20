import numpy as np

from experiments.stats import bootstrap_ci, paired_permutation_p, paired_summary


def test_permutation_p_all_same_sign_small_sample():
    # 5 seeds all positive: only the all-plus and all-minus flips are as extreme -> 2/32
    assert paired_permutation_p([0.1, 0.2, 0.15, 0.3, 0.05]) == 2 / 32


def test_permutation_p_symmetric_noise_is_large():
    assert paired_permutation_p([0.1, -0.1, 0.1, -0.1, 0.0]) == 1.0


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    d = [0.01, 0.03, -0.01, 0.02, 0.0]
    lo, hi = bootstrap_ci(d)
    assert lo <= np.mean(d) <= hi
    assert (lo, hi) == bootstrap_ci(d)


def test_paired_summary_counts_wins():
    s = paired_summary([0.9, 0.8, 0.7], [0.8, 0.8, 0.6])
    assert s["wins"] == 2 and s["n"] == 3
    assert abs(s["mean_diff"] - 0.2 / 3) < 1e-12
