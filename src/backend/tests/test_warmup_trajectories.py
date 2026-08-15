"""Guard cho generator warmup trajectories (MATRIX-META-001).

Kiểm 3 hợp đồng liêm chính:
  - không leakage: profile synthetic không trùng dataset eval;
  - doc đúng schema extract_outcome_samples và đủ META_KEYS_V2;
  - featurize được bằng đúng config outcome_head v2 (5 thuật toán).
"""

import json

import pytest

from hagent.world.meta_features import META_KEYS_V2
from hagent.world.predictor.outcome_head_v1 import (
    extract_outcome_samples,
    outcome_features,
)
from scripts.generate_warmup_trajectories import (
    EVAL_DATASETS,
    JOB_CFG,
    build_profiles,
    generate,
    make_dataset,
    make_doc,
)


def _stub_search(algo, X, y, job_cfg, seed):
    return {
        "best_params": {"n_estimators": 50},
        "best_score": 0.9,
        "seconds": 0.01,
        "time_limited": False,
    }


def test_profiles_disjoint_from_eval():
    names = {p["name"] for p in build_profiles()}
    assert not names & EVAL_DATASETS
    assert len(names) == 24


def test_make_dataset_meta_matches_profile():
    profile = {
        "name": "synth_r300_c3_cat40",
        "n_rows": 300,
        "n_classes": 3,
        "frac_cat": 0.4,
        "n_features": 20,
        "imbalanced": True,
    }
    ds = make_dataset(profile, seed=0)
    meta = ds["meta"]
    assert set(meta) == set(META_KEYS_V2)
    assert meta["n_rows"] == 300.0
    assert meta["n_classes"] == 3.0
    assert meta["frac_categorical"] == pytest.approx(0.4)
    # imbalanced: lớp lớn nhất ~0.6, chắc chắn > cân bằng 1/3
    assert meta["class_imbalance"] > 0.45
    assert meta["mean_abs_skew"] > 0.0
    assert ds["X"].shape == (300, 20)


def test_generate_docs_trainable(tmp_path):
    out = tmp_path / "warmup.jsonl"
    report = generate(out, seed=0, limit=10, search_fn=_stub_search)
    assert report["todo"] == 10

    docs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    samples = extract_outcome_samples(docs)
    assert len(samples) == 10
    for s in samples:
        assert s["best_score"] == 0.9
        assert set(s["dataset_meta"]) == set(META_KEYS_V2)
        # featurize bằng vocab v2 — không nổ, đúng chiều với head prod
        x = outcome_features(
            s["params"],
            s["dataset_meta"],
            None,
            {
                "search_algorithms": [
                    "grid_search",
                    "bayesian_search",
                    "genetic_algorithm",
                    "random_search",
                    "successive_halving",
                ]
            },
        )
        assert x.ndim == 1 and x.shape[0] > 0

    # resume: chạy lại không ghi trùng
    report2 = generate(out, seed=0, limit=10, search_fn=_stub_search)
    assert report2["done"] == 10
    lines_after = len(out.read_text(encoding="utf-8").splitlines())
    assert lines_after == 10 + report2["todo"]


def test_doc_params_match_serve_variant_shape():
    doc = make_doc(
        "synth_x",
        "grid_search",
        {k: 0.0 for k in META_KEYS_V2},
        {"best_score": 0.8, "seconds": 1.0, "time_limited": False, "best_params": {}},
        JOB_CFG,
    )
    cfg = doc["next_observation"]["jobs"]["warmup:synth_x:grid_search"]["config"]
    # đúng shape variant serve của builder — KHÔNG có 'models' khi user
    # không ràng buộc (train/serve consistency)
    assert set(cfg) == {"search_algorithm", "problem_type", "metric", "time_limit"}
