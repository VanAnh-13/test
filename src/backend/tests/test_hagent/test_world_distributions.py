"""
Unit tests for non-Gaussian distribution support in World Model (REFAC-012).
"""

from __future__ import annotations

import math

from hagent.world.schema import DistributionSpec, DistributionType
from hagent.world.surprise import (
    compute_distribution_surprise,
    kl_divergence_beta,
    kl_divergence_categorical,
    kl_divergence_dirichlet,
    kl_divergence_gaussian,
)
from hagent.world.updater import (
    bayesian_beta_update,
    bayesian_categorical_update,
    bayesian_dirichlet_update,
    infer_distribution_type,
    update_distribution,
)


def test_schema_distribution_spec() -> None:
    """DistributionSpec khởi tạo và serialize chính xác."""
    spec = DistributionSpec(
        dist_type=DistributionType.BETA.value,
        params={"alpha": 2.0, "beta": 5.0},
        meta={"metric": "accuracy"},
    )
    d = spec.to_dict()
    assert d["dist_type"] == "beta"
    assert d["params"]["alpha"] == 2.0
    assert d["schema_version"] == "1.0"


def test_infer_distribution_type() -> None:
    """Tự động suy luận distribution type từ metric_type hoặc dữ liệu mẫu."""
    # Bounded classification metrics -> Beta
    assert infer_distribution_type(metric_type="accuracy") == "beta"
    assert infer_distribution_type(metric_type="f1_score") == "beta"
    assert infer_distribution_type(metric_type="roc_auc") == "beta"
    assert infer_distribution_type(metric_type="precision") == "beta"

    # Class probabilities list summing to 1 -> Dirichlet
    assert infer_distribution_type(sample=[0.7, 0.2, 0.1]) == "dirichlet"

    # Discrete choice dict summing to 1 -> Categorical
    assert infer_distribution_type(sample={"opt_a": 0.6, "opt_b": 0.4}) == "categorical"

    # Latent vectors / continuous -> Gaussian
    assert infer_distribution_type(sample=[1.5, -2.3, 0.4]) == "gaussian"
    assert infer_distribution_type(metric_type="mse") == "gaussian"


def test_bayesian_beta_update() -> None:
    """Cập nhật Beta phân phối nhị thức liên hợp."""
    # Prior Beta(1, 1), 8 successes, 2 failures -> Beta(9, 3)
    post_a, post_b = bayesian_beta_update(1.0, 1.0, 8.0, 2.0)
    assert abs(post_a - 9.0) < 1e-8
    assert abs(post_b - 3.0) < 1e-8

    # Edge cases: 0 counts
    post_a0, post_b0 = bayesian_beta_update(5.0, 5.0, 0.0, 0.0)
    assert abs(post_a0 - 5.0) < 1e-8
    assert abs(post_b0 - 5.0) < 1e-8


def test_bayesian_dirichlet_update() -> None:
    """Cập nhật Dirichlet phân phối đa thức liên hợp."""
    prior = [1.0, 1.0, 1.0]
    counts = [10.0, 5.0, 2.0]
    post = bayesian_dirichlet_update(prior, counts)
    assert post == [11.0, 6.0, 3.0]


def test_bayesian_categorical_update() -> None:
    """Cập nhật Categorical phân phối rời rạc."""
    priors = [0.5, 0.5]
    counts = [10.0, 1.0]
    post = bayesian_categorical_update(priors, counts)
    assert len(post) == 2
    assert abs(sum(post) - 1.0) < 1e-8
    assert post[0] > post[1]


def test_update_distribution_dispatcher() -> None:
    """update_distribution điều phối cập nhật chính xác theo dist_type."""
    # Beta
    beta_spec = {"dist_type": "beta", "params": {"alpha": 2.0, "beta": 2.0}}
    beta_updated = update_distribution(
        beta_spec, observation={"successes": 6.0, "failures": 0.0}
    )
    assert beta_updated["dist_type"] == "beta"
    assert beta_updated["params"]["alpha"] == 8.0
    assert abs(beta_updated["mean"] - 0.8) < 1e-6

    # Gaussian
    gauss_spec = {"dist_type": "gaussian", "params": {"mean": 0.0, "std": 1.0}}
    gauss_updated = update_distribution(gauss_spec, observation=2.0)
    assert gauss_updated["dist_type"] == "gaussian"
    assert "mean" in gauss_updated["params"]


def test_kl_divergence_gaussian() -> None:
    """KL divergence cho Gaussian distributions."""
    # Trùng nhau -> KL = 0
    assert abs(kl_divergence_gaussian(0.0, 1.0, 0.0, 1.0)) < 1e-10

    # Khác nhau -> KL > 0
    kl = kl_divergence_gaussian(1.0, 1.0, 0.0, 1.0)
    # KL = (1^2) / 2 = 0.5
    assert abs(kl - 0.5) < 1e-8


def test_kl_divergence_beta() -> None:
    """KL divergence cho Beta distributions."""
    # Trùng nhau -> KL = 0
    assert abs(kl_divergence_beta(2.0, 2.0, 2.0, 2.0)) < 1e-8

    # Khác nhau -> KL > 0
    kl = kl_divergence_beta(10.0, 2.0, 2.0, 2.0)
    assert kl > 0.0
    assert math.isfinite(kl)


def test_kl_divergence_categorical() -> None:
    """KL divergence cho Categorical distributions."""
    # Trùng nhau -> KL = 0
    assert abs(kl_divergence_categorical([0.5, 0.5], [0.5, 0.5])) < 1e-10

    # Khác nhau -> KL > 0
    kl = kl_divergence_categorical([0.9, 0.1], [0.5, 0.5])
    assert kl > 0.0


def test_kl_divergence_dirichlet() -> None:
    """KL divergence cho Dirichlet distributions."""
    # Trùng nhau -> KL = 0
    assert abs(kl_divergence_dirichlet([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])) < 1e-8

    # Khác nhau -> KL > 0
    kl = kl_divergence_dirichlet([10.0, 2.0, 1.0], [2.0, 2.0, 2.0])
    assert kl > 0.0


def test_compute_distribution_surprise() -> None:
    """compute_distribution_surprise tính surprise và phân loại level theo KL divergence."""
    pred_beta = {"dist_type": "beta", "params": {"alpha": 1.0, "beta": 1.0}}
    # Near identical observation (val ~ 0.5) -> low surprise
    surp_low = compute_distribution_surprise(pred_beta, 0.50)
    assert surp_low.level == "low"

    # Extreme discrepancy (val = 0.99) -> higher surprise
    surp_high = compute_distribution_surprise(pred_beta, 0.99)
    assert surp_high.value > surp_low.value
