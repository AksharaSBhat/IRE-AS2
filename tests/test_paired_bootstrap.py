"""
Unit tests for Paired Bootstrap CI (Q3).
Asserts that when model B is strictly better than model A, delta > 0 with CI excluding zero.
"""
import numpy as np
from src.evaluation.paired_bootstrap import compute_paired_bootstrap_ci

def test_paired_bootstrap_strictly_positive():
    np.random.seed(42)
    n = 200
    base_scores = np.random.uniform(0.5, 0.6, size=n).tolist()
    # Improved scores are systematically higher by ~0.05
    imp_scores = (np.array(base_scores) + np.random.uniform(0.02, 0.08, size=n)).tolist()

    res = compute_paired_bootstrap_ci(base_scores, imp_scores, n_rounds=500)
    assert res["mean_delta"] > 0
    assert res["ci_lower"] > 0
    assert res["significant"] is True
    assert res["p_value"] < 0.05

def test_paired_bootstrap_neutral():
    np.random.seed(42)
    n = 200
    base_scores = np.random.uniform(0.5, 0.6, size=n).tolist()
    # No systematic difference
    imp_scores = (np.array(base_scores) + np.random.normal(0, 0.02, size=n)).tolist()

    res = compute_paired_bootstrap_ci(base_scores, imp_scores, n_rounds=500)
    # CI should span across zero
    assert res["ci_lower"] <= 0.0 <= res["ci_upper"]
