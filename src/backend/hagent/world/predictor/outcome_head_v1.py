"""
Outcome head v1 — dự đoán phân phối best_score của một training-job variant.

(μ, σ) = g_φ(features(variant params, dataset meta, z))

Numpy-only, checkpoint .npz như neural_jepa_v1. σ là aleatoric (Gaussian NLL);
epistemic uncertainty lấy từ ensemble wrapper (world/predictor/ensemble.py).
Chưa có checkpoint → predict trả None, caller giữ nguyên thứ tự variant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from hagent.world.predictor.base import load_mlp_weights

logger = structlog.get_logger(__name__)

_DEFAULT_SEARCH_ALGOS = ["grid_search", "bayesian_search", "genetic_algorithm"]
_DEFAULT_PROBLEM_TYPES = ["classification", "regression"]
_DEFAULT_METRICS = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "mae",
    "mse",
    "rmse",
    "r2",
]
_SIGMA_MIN = 1e-3


def _one_hot_with_unknown(value: Any, space: Sequence[str]) -> np.ndarray:
    """One-hot trên space + 1 bucket unknown cuối."""
    v = np.zeros(len(space) + 1, dtype=np.float64)
    key = str(value).lower() if value is not None else ""
    try:
        v[list(space).index(key)] = 1.0
    except ValueError:
        v[-1] = 1.0 if key else 0.0
    return v


def _softplus(x: float) -> float:
    return float(np.logaddexp(0.0, x))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))))


def outcome_feature_config(config: dict | None = None) -> dict[str, Any]:
    """Chuẩn hóa config feature — một nguồn duy nhất cho cả train và predict."""
    cfg = dict(config or {})
    return {
        "search_algorithms": list(cfg.get("search_algorithms") or _DEFAULT_SEARCH_ALGOS),
        "problem_types": list(cfg.get("problem_types") or _DEFAULT_PROBLEM_TYPES),
        "metrics": list(cfg.get("metrics") or _DEFAULT_METRICS),
        "time_limit_norm": float(cfg.get("time_limit_norm", 600.0)),
        "use_latent": bool(cfg.get("use_latent", True)),
        "latent_dim": int(cfg.get("latent_dim", 64)),
        # Vocab để multi-hot membership của params["models"]; rỗng = tắt
        # (giữ nguyên chiều feature của checkpoint cũ)
        "model_vocab": list(cfg.get("model_vocab") or []),
        # v1: chỉ n_rows/n_cols (checkpoint cũ). v2: đủ META_KEYS_V2 —
        # điều kiện cần cho transfer xuyên dataset.
        "meta_profile": str(cfg.get("meta_profile") or "v2"),
    }


def outcome_features(
    params: dict[str, Any],
    dataset_meta: dict[str, Any] | None = None,
    z: Sequence[float] | None = None,
    config: dict | None = None,
) -> np.ndarray:
    """
    Vector đặc trưng cố định chiều cho một variant config.

    Thành phần: one-hot search_algorithm / problem_type / metric (+unknown),
    time_limit (log-normalized), số models, meta dataset (log rows/cols), bias,
    và (tùy config) latent z pad/cắt về latent_dim.
    """
    fc = outcome_feature_config(config)
    params = dict(params or {})
    meta = dict(dataset_meta or {})

    algo_oh = _one_hot_with_unknown(
        params.get("search_algorithm"), fc["search_algorithms"]
    )
    parts: list[np.ndarray] = [
        algo_oh,
        _one_hot_with_unknown(params.get("problem_type"), fc["problem_types"]),
        _one_hot_with_unknown(params.get("metric"), fc["metrics"]),
    ]

    norm = max(1.0, fc["time_limit_norm"])
    try:
        t = float(params.get("time_limit") or 0.0)
    except (TypeError, ValueError):
        t = 0.0
    parts.append(
        np.array(
            [
                min(2.0, math.log1p(max(0.0, t)) / math.log1p(norm)),
                1.0 if t > 0 else 0.0,
            ]
        )
    )

    models = params.get("models") or []
    n_models = len(models) if isinstance(models, (list, tuple)) else 0
    parts.append(np.array([min(n_models, 10) / 10.0, 1.0 if n_models else 0.0]))

    vocab = fc["model_vocab"]
    if vocab:
        member = np.zeros(len(vocab), dtype=np.float64)
        if isinstance(models, (list, tuple)):
            model_set = {str(m) for m in models}
            for i, name in enumerate(vocab):
                if name in model_set:
                    member[i] = 1.0
        parts.append(member)

    def _log_scaled(key: str, denom: float) -> float:
        try:
            val = float(meta.get(key) or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        return min(2.0, math.log1p(max(0.0, val)) / math.log(denom))

    def _unit(key: str, scale: float = 1.0) -> float:
        try:
            val = float(meta.get(key) or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        return max(0.0, min(1.0, val / scale))

    if fc["meta_profile"] == "v2":
        log_rows = _log_scaled("n_rows", 1e6)
        frac_cat = _unit("frac_categorical")
        parts.append(
            np.array(
                [
                    log_rows,
                    _log_scaled("n_cols", 1e3),
                    _log_scaled("n_classes", 1e2),
                    _unit("class_imbalance"),
                    frac_cat,
                    _unit("missing_frac"),
                    _unit("mean_abs_skew", scale=3.0),
                    1.0,
                ]
            )
        )
        # Giao chéo tường minh algo × meta — "thuật toán nào tốt phụ thuộc
        # dataset" phải học được TUYẾN TÍNH, không trông chờ SGD tự tìm
        # interaction trong MLP nhỏ (đã kiểm chứng SGD thuần fail).
        parts.append(np.outer(algo_oh, np.array([log_rows, frac_cat])).ravel())
    else:  # "v1" — chiều feature của checkpoint thế hệ đầu
        parts.append(
            np.array([_log_scaled("n_rows", 1e6), _log_scaled("n_cols", 1e3), 1.0])
        )

    if fc["use_latent"]:
        dim = fc["latent_dim"]
        zv = np.zeros(dim, dtype=np.float64)
        if z is not None:
            arr = np.asarray(list(z)[:dim], dtype=np.float64)
            zv[: arr.shape[0]] = arr
        parts.append(zv)

    return np.concatenate(parts)


def outcome_feature_dim(config: dict | None = None) -> int:
    return int(outcome_features({}, None, None, config).shape[0])


@dataclass
class OutcomePrediction:
    """Phân phối best_score dự đoán cho một variant."""

    mean: float
    std: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "meta": dict(self.meta)}


class OutcomeHeadV1:
    """
    MLP một lớp ẩn:
      h = tanh(W1 @ x + b1)
      [μ, s] = W2 @ h + b2 ;  σ = softplus(s) + σ_min
    """

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.feature_cfg = outcome_feature_config(self.config)
        self.hidden_dim = int(self.config.get("hidden_dim", 64))
        self.checkpoint_path = self.config.get("checkpoint_path")
        self._W1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._W2: np.ndarray | None = None
        self._b2: np.ndarray | None = None
        self._loaded = False

        if self.checkpoint_path:
            self._try_load(str(self.checkpoint_path))

    # ── State ────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._W1 is not None

    def init_random(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        in_dim = outcome_feature_dim(self.feature_cfg)
        h = self.hidden_dim
        self._W1 = rng.normal(0, 1.0 / math.sqrt(in_dim), size=(h, in_dim))
        self._b1 = np.zeros(h)
        self._W2 = rng.normal(0, 1.0 / math.sqrt(h), size=(2, h))
        self._b2 = np.zeros(2)
        self._loaded = True

    def save(self, path: str) -> None:
        if self._W1 is None:
            raise RuntimeError("No weights to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self._W1,
            b1=self._b1,
            W2=self._W2,
            b2=self._b2,
            hidden_dim=self.hidden_dim,
            search_algorithms=np.array(
                self.feature_cfg["search_algorithms"], dtype=object
            ),
            problem_types=np.array(self.feature_cfg["problem_types"], dtype=object),
            metrics=np.array(self.feature_cfg["metrics"], dtype=object),
            time_limit_norm=self.feature_cfg["time_limit_norm"],
            use_latent=self.feature_cfg["use_latent"],
            latent_dim=self.feature_cfg["latent_dim"],
            model_vocab=np.array(self.feature_cfg["model_vocab"], dtype=object),
            meta_profile=self.feature_cfg["meta_profile"],
        )

    def _try_load(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            logger.info("Outcome head checkpoint missing: %s", path)
            return
        try:
            data = np.load(str(p), allow_pickle=True)
            self._W1, self._b1, self._W2, self._b2 = load_mlp_weights(data)
            for key in ("search_algorithms", "problem_types", "metrics", "model_vocab"):
                if key in data:
                    self.feature_cfg[key] = [str(x) for x in data[key].tolist()]
            if "time_limit_norm" in data:
                self.feature_cfg["time_limit_norm"] = float(data["time_limit_norm"])
            if "use_latent" in data:
                self.feature_cfg["use_latent"] = bool(data["use_latent"])
            if "latent_dim" in data:
                self.feature_cfg["latent_dim"] = int(data["latent_dim"])
            # Checkpoint thế hệ đầu không có meta_profile → chiều v1
            self.feature_cfg["meta_profile"] = (
                str(data["meta_profile"]) if "meta_profile" in data else "v1"
            )
            self._loaded = True
            logger.info("Loaded outcome head checkpoint from %s", path)
        except Exception as exc:
            logger.warning("Failed to load outcome head checkpoint: %s", exc)
            self._loaded = False

    # ── Inference ────────────────────────────────────────

    def _forward(self, x: np.ndarray) -> tuple[float, float, np.ndarray]:
        assert self._W1 is not None and self._b1 is not None
        assert self._W2 is not None and self._b2 is not None
        h = np.tanh(self._W1 @ x + self._b1)
        out = self._W2 @ h + self._b2
        mu = float(out[0])
        sigma = _softplus(float(out[1])) + _SIGMA_MIN
        return mu, sigma, h

    def predict(
        self,
        params: dict[str, Any],
        dataset_meta: dict[str, Any] | None = None,
        z: Sequence[float] | None = None,
    ) -> OutcomePrediction | None:
        if not self.is_ready:
            return None
        x = outcome_features(params, dataset_meta, z, self.feature_cfg)
        expected = self._W1.shape[1]  # type: ignore[union-attr]
        if x.shape[0] != expected:
            # Feature config lệch checkpoint — không đoán bừa
            logger.debug(
                "Outcome head feature dim mismatch: %d != %d", x.shape[0], expected
            )
            return None
        mu, sigma, _ = self._forward(x)
        return OutcomePrediction(
            mean=mu,
            std=sigma,
            meta={"predictor": "outcome_head_v1"},
        )


# ── Training ─────────────────────────────────────────────


def train_outcome_head(
    samples: list[dict[str, Any]],
    *,
    config: dict | None = None,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 0,
    warmup_epochs: int | None = None,
) -> OutcomeHeadV1:
    """
    SGD từ samples: sample = {params, dataset_meta?, z?, best_score}.

    Warmup MSE (mặc định epochs//3) cho μ trước, rồi mới Gaussian NLL —
    NLL từ đầu để σ phình sớm làm gradient của μ tắt (model sập về mean).

    Trả về head đã train (head.config["train_history"] chứa loss theo epoch).
    """
    head = OutcomeHeadV1(dict(config or {}))
    head.init_random(seed=seed)
    fc = head.feature_cfg

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for s in samples:
        try:
            y = float(s["best_score"])
        except (KeyError, TypeError, ValueError):
            continue
        xs.append(
            outcome_features(
                dict(s.get("params") or {}),
                s.get("dataset_meta"),
                s.get("z"),
                fc,
            )
        )
        ys.append(y)

    if not xs:
        logger.warning("No valid outcome samples for training")
        return head

    rng = np.random.default_rng(seed)
    order = np.arange(len(xs))
    history: list[float] = []
    if warmup_epochs is None:
        warmup_epochs = max(1, epochs // 3)
    assert head._W1 is not None
    for epoch in range(epochs):
        warm = epoch < warmup_epochs
        rng.shuffle(order)
        total = 0.0
        for i in order:
            x, y = xs[i], ys[i]
            h = np.tanh(head._W1 @ x + head._b1)
            out = head._W2 @ h + head._b2
            mu = float(out[0])
            s_raw = float(out[1])
            sigma = _softplus(s_raw) + _SIGMA_MIN
            var = sigma * sigma
            err = mu - y
            total += 0.5 * (err * err / var) + math.log(sigma)

            if warm:
                # MSE thuần cho μ; σ đứng yên
                d_out = np.array([2.0 * err, 0.0])
            else:
                d_mu = err / var
                d_sigma = (1.0 / sigma) - (err * err) / (sigma * var)
                d_out = np.array([d_mu, d_sigma * _sigmoid(s_raw)])
            gnorm = float(np.linalg.norm(d_out))
            if gnorm > 5.0:
                d_out *= 5.0 / gnorm

            d_h = head._W2.T @ d_out
            d_h *= 1.0 - h * h
            head._W2 -= lr * np.outer(d_out, h)
            head._b2 -= lr * d_out
            head._W1 -= lr * np.outer(d_h, x)
            head._b1 -= lr * d_h

        mean_loss = total / len(xs)
        history.append(mean_loss)
        if epoch % max(1, epochs // 5) == 0:
            logger.info("outcome_head epoch %d mean_nll=%.6f", epoch, mean_loss)

    head.config["train_history"] = history
    return head


def extract_outcome_samples(
    trajectory_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Lấy sample (params, dataset_meta, z, best_score) từ trajectory docs:
    quét jobs completed trong next_observation, dedup theo job id (giữ bản ghi
    mới nhất — docs được duyệt theo thứ tự, bản sau ghi đè bản trước).
    """
    by_job: dict[str, dict[str, Any]] = {}
    for doc in trajectory_docs or []:
        obs = doc.get("next_observation") or doc.get("observation") or {}
        jobs = obs.get("jobs") or {}
        datasets = obs.get("datasets") or {}
        z_doc = doc.get("z_next") or doc.get("z") or {}
        z_vec = z_doc.get("vector")
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            if str(job.get("status") or "").lower() != "completed":
                continue
            if job.get("best_score") is None:
                continue
            config = job.get("config")
            if not isinstance(config, dict) or not config:
                continue
            ds_id = job.get("dataset_id")
            meta = datasets.get(ds_id) if isinstance(datasets, dict) else None
            by_job[str(job_id)] = {
                "params": dict(config),
                "dataset_meta": dict(meta) if isinstance(meta, dict) else None,
                "z": list(z_vec) if isinstance(z_vec, (list, tuple)) else None,
                "best_score": job.get("best_score"),
            }
    return list(by_job.values())


def rank_variants_by_outcome(
    variants: list[Any],
    *,
    head: OutcomeHeadV1 | None,
    dataset_meta: dict[str, Any] | None = None,
    z: Sequence[float] | None = None,
    higher_is_better: bool = True,
) -> list[tuple[Any, OutcomePrediction | None]]:
    """
    Xếp hạng CampaignVariant theo mean dự đoán (giảm dần khi higher_is_better).
    Head chưa sẵn sàng → giữ nguyên thứ tự, prediction None.
    """
    if head is None or not head.is_ready:
        return [(v, None) for v in variants]

    scored: list[tuple[Any, OutcomePrediction | None]] = []
    for v in variants:
        params = getattr(v, "params", None)
        if params is None and isinstance(v, dict):
            params = v.get("params")
        pred = head.predict(dict(params or {}), dataset_meta, z)
        scored.append((v, pred))

    def sort_key(item: tuple[Any, OutcomePrediction | None]) -> float:
        pred = item[1]
        if pred is None:
            return float("-inf") if higher_is_better else float("inf")
        return pred.mean

    scored.sort(key=sort_key, reverse=higher_is_better)
    return scored
