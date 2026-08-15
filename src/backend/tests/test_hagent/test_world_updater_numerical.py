"""
Tests for numerical stability and Bayesian log-space updates in world/updater.py.
"""

from __future__ import annotations

import math
import time

from hagent.world.updater import (
    bayesian_belief_update,
    bayesian_belief_update_linear,
    bayesian_gaussian_update,
    gaussian_log_likelihood,
    log_sum_exp,
    update_discrete_distribution,
)


def test_log_sum_exp_basic() -> None:
    """log_sum_exp phải cho kết quả chính xác cho các giá trị thông thường."""
    vals = [math.log(0.2), math.log(0.3), math.log(0.5)]
    res = log_sum_exp(vals)
    # log(0.2 + 0.3 + 0.5) = log(1.0) = 0.0
    assert abs(res - 0.0) < 1e-10


def test_log_sum_exp_extreme_values() -> None:
    """log_sum_exp phải chống overflow khi có giá trị rất lớn và underflow khi có giá trị rất nhỏ."""
    # Overflow test: exp(1000) sẽ OverflowError nếu tính naive
    large_vals = [1000.0, 1000.0]
    res_large = log_sum_exp(large_vals)
    # log(exp(1000) + exp(1000)) = 1000 + log(2)
    expected_large = 1000.0 + math.log(2.0)
    assert abs(res_large - expected_large) < 1e-8
    assert math.isfinite(res_large)

    # Underflow test: exp(-1000) = 0 naive
    small_vals = [-1000.0, -1000.0]
    res_small = log_sum_exp(small_vals)
    expected_small = -1000.0 + math.log(2.0)
    assert abs(res_small - expected_small) < 1e-8
    assert math.isfinite(res_small)


def test_bayesian_belief_update_underflow_prior() -> None:
    """Priors cực nhỏ (1e-300) không gây NaN hoặc Inf."""
    priors = [1e-300, 1.0 - 1e-300]
    log_lls = [10.0, 0.0]
    post = bayesian_belief_update(priors, log_lls)

    assert len(post) == 2
    assert all(math.isfinite(p) for p in post)
    assert abs(sum(post) - 1.0) < 1e-10
    assert post[0] >= 0.0
    assert post[1] <= 1.0


def test_bayesian_belief_update_overflow_likelihood() -> None:
    """Likelihood rất lớn (+2000.0) không crash và normalize thành công."""
    priors = [0.5, 0.5]
    log_lls = [2000.0, 2005.0]  # Chênh lệch 5.0 in log space
    post = bayesian_belief_update(priors, log_lls)

    assert len(post) == 2
    assert all(math.isfinite(p) for p in post)
    assert abs(sum(post) - 1.0) < 1e-10
    # Ratio P(H2)/P(H1) phải là exp(5) ~ 148.413
    assert abs(post[1] / post[0] - math.exp(5.0)) < 1e-6


def test_numerical_equivalence_with_standard_bayes() -> None:
    """
    So sánh kết quả log-space Bayesian updater với công thức nhân trực tiếp tiêu chuẩn.
    Sai số phải < 1e-8.
    """
    priors = [0.2, 0.3, 0.5]
    likelihoods = [0.8, 0.4, 0.1]

    # Standard Bayes: (p_i * L_i) / sum(p_j * L_j)
    unnorm_standard = [p * l for p, l in zip(priors, likelihoods)]
    total_standard = sum(unnorm_standard)
    expected_post = [u / total_standard for u in unnorm_standard]

    # Log-space updater
    log_lls = [math.log(l) for l in likelihoods]
    actual_post = bayesian_belief_update(priors, log_lls)

    for act, exp in zip(actual_post, expected_post):
        assert abs(act - exp) < 1e-8, (
            f"Difference {abs(act - exp)} exceeds tolerance 1e-8"
        )


def test_bayesian_belief_update_linear() -> None:
    """Hàm wrapper bayesian_belief_update_linear nhận linear likelihoods."""
    priors = [0.5, 0.5]
    likelihoods = [0.9, 0.1]
    post = bayesian_belief_update_linear(priors, likelihoods)
    assert abs(post[0] - 0.9) < 1e-10
    assert abs(post[1] - 0.1) < 1e-10


def test_update_discrete_distribution() -> None:
    """Cập nhật phân phối xác suất dạng dict."""
    prior_dist = {"model_a": 0.5, "model_b": 0.5}
    log_lls = {"model_a": math.log(0.8), "model_b": math.log(0.2)}
    post_dist = update_discrete_distribution(prior_dist, log_lls)

    assert abs(post_dist["model_a"] - 0.8) < 1e-10
    assert abs(post_dist["model_b"] - 0.2) < 1e-10


def test_gaussian_log_likelihood() -> None:
    """Tính log-likelihood chính xác của phân phối chuẩn Gaussian."""
    ll_center = gaussian_log_likelihood(0.0, 0.0, 1.0)
    # log(1 / sqrt(2*pi)) = -0.5 * log(2*pi) ~ -0.9189385332
    assert abs(ll_center - (-0.5 * math.log(2 * math.pi))) < 1e-8

    ll_tail = gaussian_log_likelihood(3.0, 0.0, 1.0)
    assert ll_tail < ll_center
    assert abs(ll_tail - (ll_center - 4.5)) < 1e-8


def test_bayesian_gaussian_update() -> None:
    """Cập nhật Gaussian conjugate (Kalman 1D)."""
    # Prior N(0, 1), Obs N(2, 1) -> Posterior N(1, 0.5)
    post_mean, post_var = bayesian_gaussian_update(0.0, 1.0, 2.0, 1.0)
    assert abs(post_mean - 1.0) < 1e-8
    assert abs(post_var - 0.5) < 1e-8


def test_benchmark_speed() -> None:
    """Benchmark: 10,000 lần Bayesian update phải chạy dưới 0.5 giây."""
    priors = [0.1 * i for i in range(1, 5)]
    s = sum(priors)
    priors = [p / s for p in priors]
    log_lls = [-0.5, -1.2, -0.1, -2.0]

    start = time.perf_counter()
    for _ in range(10_000):
        bayesian_belief_update(priors, log_lls)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, (
        f"10k Bayesian updates took {elapsed:.3f}s (exceeds threshold 0.5s)"
    )
