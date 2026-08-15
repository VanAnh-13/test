"""
Kiểm thử tích hợp cho Vòng đời và Động lực học của World Model (REFAC-027).
"""

from __future__ import annotations

from hagent.world.schema import LatentState
from hagent.world.surprise import (
    compute_surprise,
    kl_divergence_beta,
    kl_divergence_gaussian,
)
from hagent.world.updater import (
    bayesian_beta_update,
    bayesian_gaussian_update,
)


def test_world_model_bayesian_lifecycle() -> None:
    """Kiểm thử vòng đời cập nhật Bayesian cho cả phân phối Gaussian và Beta."""
    # 1. Cập nhật niềm tin cho phân phối chuẩn (Gaussian update)
    prior_mean = 0.5
    prior_var = 0.1
    obs_val = 0.85
    obs_var = 0.05

    post_mean, post_var = bayesian_gaussian_update(
        prior_mean, prior_var, obs_val, obs_var
    )
    assert prior_mean < post_mean < obs_val
    assert post_var < prior_var

    kl_gauss = kl_divergence_gaussian(prior_mean, prior_var, post_mean, post_var)
    assert kl_gauss >= 0.0

    # 2. Cập nhật niềm tin cho phân phối Beta
    alpha_0 = 2.0
    beta_0 = 2.0
    # Quan sát 8 thành công, 2 thất bại
    alpha_post, beta_post = bayesian_beta_update(
        alpha_0, beta_0, successes=8, failures=2
    )
    assert alpha_post == 10.0
    assert beta_post == 4.0

    kl_beta = kl_divergence_beta(alpha_0, beta_0, alpha_post, beta_post)
    assert kl_beta >= 0.0


def test_world_model_surprise_pipeline() -> None:
    """Kiểm thử pipeline tính toán surprise đa chiều và phân loại ngưỡng độ bất ngờ."""
    pred_latent = LatentState(dim=4, vector=[0.25, 0.25, 0.25, 0.25])
    actual_latent = LatentState(dim=4, vector=[0.90, 0.05, 0.03, 0.02])

    surprise_res = compute_surprise(pred_latent, actual_latent)
    assert surprise_res.value > 0.0
    assert surprise_res.level in {"low", "medium", "high"}
