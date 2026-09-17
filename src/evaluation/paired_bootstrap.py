"""
Paired Bootstrap Confidence Interval Engine (Assignment 2 - Q3).
Computes paired non-parametric bootstrap distributions for metric deltas
between Baseline and Improved models to prove statistical significance (Δ > 0).
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

def compute_paired_bootstrap_ci(
    metrics_baseline: List[float],
    metrics_improved: List[float],
    n_rounds: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Computes the paired bootstrap confidence interval for delta = improved - baseline.
    Returns:
        mean_baseline: float
        mean_improved: float
        mean_delta: float
        ci_lower: float
        ci_upper: float
        p_value: float (empirical fraction where delta <= 0)
        significant: bool (True if ci_lower > 0)
    """
    arr_base = np.array(metrics_baseline, dtype=np.float64)
    arr_imp = np.array(metrics_improved, dtype=np.float64)

    assert len(arr_base) == len(arr_imp), "Baseline and Improved metric arrays must have identical length."
    n = len(arr_base)
    if n == 0:
        return {
            "mean_baseline": 0.0,
            "mean_improved": 0.0,
            "mean_delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "significant": False
        }

    # Paired differences per sample
    paired_diffs = arr_imp - arr_base
    obs_delta = float(np.mean(paired_diffs))
    obs_base = float(np.mean(arr_base))
    obs_imp = float(np.mean(arr_imp))

    rng = np.random.default_rng(random_seed)
    
    # Vectorized bootstrap resampling
    # Draw (n_rounds, n) random indices
    sample_indices = rng.integers(0, n, size=(n_rounds, n))
    
    # Resampled mean deltas
    boot_deltas = np.mean(paired_diffs[sample_indices], axis=1)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_deltas, lower_pct))
    ci_upper = float(np.percentile(boot_deltas, upper_pct))

    # Empirical one-sided p-value: fraction of bootstrap samples where delta <= 0
    p_val = float(np.mean(boot_deltas <= 0.0))

    significant = bool(ci_lower > 0.0)

    return {
        "mean_baseline": obs_base,
        "mean_improved": obs_imp,
        "mean_delta": obs_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "significant": significant,
        "n_samples": n,
        "n_rounds": n_rounds
    }


def evaluate_paired_models(
    scores_baseline: List[np.ndarray],
    scores_improved: List[np.ndarray],
    labels_list: List[np.ndarray],
    n_rounds: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    Computes paired bootstrap CI across all standard ranking metrics:
    AUC, MRR, nDCG@5, and nDCG@10.
    """
    from src.evaluation.metrics import (
        compute_impression_auc,
        compute_mrr,
        compute_ndcg_at_k
    )

    base_auc, imp_auc = [], []
    base_mrr, imp_mrr = [], []
    base_ndcg5, imp_ndcg5 = [], []
    base_ndcg10, imp_ndcg10 = [], []

    for s_base, s_imp, y in zip(scores_baseline, scores_improved, labels_list):
        if len(y) < 2 or sum(y) == 0 or sum(y) == len(y):
            continue  # Skip impressions with all positive or all negative

        # AUC
        base_auc.append(compute_impression_auc(y, s_base))
        imp_auc.append(compute_impression_auc(y, s_imp))

        # MRR
        base_mrr.append(compute_mrr(y, s_base))
        imp_mrr.append(compute_mrr(y, s_imp))

        # nDCG@5
        base_ndcg5.append(compute_ndcg_at_k(y, s_base, k=5))
        imp_ndcg5.append(compute_ndcg_at_k(y, s_imp, k=5))

        # nDCG@10
        base_ndcg10.append(compute_ndcg_at_k(y, s_base, k=10))
        imp_ndcg10.append(compute_ndcg_at_k(y, s_imp, k=10))

    results = {
        "AUC": compute_paired_bootstrap_ci(base_auc, imp_auc, n_rounds=n_rounds),
        "MRR": compute_paired_bootstrap_ci(base_mrr, imp_mrr, n_rounds=n_rounds),
        "nDCG@5": compute_paired_bootstrap_ci(base_ndcg5, imp_ndcg5, n_rounds=n_rounds),
        "nDCG@10": compute_paired_bootstrap_ci(base_ndcg10, imp_ndcg10, n_rounds=n_rounds),
    }
    return results
