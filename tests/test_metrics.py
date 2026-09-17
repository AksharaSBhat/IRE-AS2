"""Unit tests for offline evaluation metrics."""
import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_impression_auc,
    compute_impression_mrr,
    compute_impression_ndcg,
    compute_intra_list_diversity,
    compute_novelty,
    compute_catalog_coverage
)

def test_auc_metric():
    labels = [1, 0, 1, 0]
    scores_perfect = [0.9, 0.1, 0.8, 0.2]
    auc_perfect = compute_impression_auc(labels, scores_perfect)
    assert auc_perfect == 1.0

    scores_worst = [0.1, 0.9, 0.2, 0.8]
    auc_worst = compute_impression_auc(labels, scores_worst)
    assert auc_worst == 0.0

    # Single class returns None
    assert compute_impression_auc([1, 1, 1], [0.5, 0.6, 0.7]) is None

def test_mrr_metric():
    labels = [0, 0, 1, 0]
    scores = [0.1, 0.2, 0.9, 0.3]  # Ranked 1st
    assert compute_impression_mrr(labels, scores) == 1.0

    scores_second = [0.1, 0.95, 0.9, 0.3]  # Ranked 2nd
    assert compute_impression_mrr(labels, scores_second) == 0.5

def test_ndcg_metric():
    labels = [1, 0, 0]
    scores = [0.9, 0.5, 0.1]
    assert compute_impression_ndcg(labels, scores, k=5) == 1.0

def test_beyond_accuracy():
    # Intra-List Diversity
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    ild = compute_intra_list_diversity(np.array([v1, v2]))
    assert abs(ild - 1.0) < 1e-5

    # Novelty
    item_probs = {"A1": 0.5, "A2": 0.25}
    nov = compute_novelty(["A1", "A2"], item_probs)
    # -log2(0.5) = 1.0, -log2(0.25) = 2.0 -> mean = 1.5
    assert abs(nov - 1.5) < 1e-5

    # Coverage
    cov = compute_catalog_coverage({"A1", "A2", "A3"}, total_catalog_size=10)
    assert cov == 0.3
