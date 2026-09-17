"""
Bootstrap confidence interval harness for ranking and evaluation metrics (Q4).
"""
from typing import List, Tuple, Dict, Optional
import numpy as np

from src.config import BOOTSTRAP_ROUNDS, BOOTSTRAP_CI

def compute_bootstrap_ci(
    values: List[float],
    n_rounds: int = BOOTSTRAP_ROUNDS,
    ci: float = BOOTSTRAP_CI,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes empirical mean and 95% Bootstrap Confidence Interval.
    Returns: (mean, lower_bound, upper_bound)
    """
    clean_values = [v for v in values if v is not None and not np.isnan(v)]
    if not clean_values:
        return 0.0, 0.0, 0.0

    arr = np.array(clean_values, dtype=np.float64)
    n = len(arr)
    if n <= 1:
        m = float(arr[0]) if n == 1 else 0.0
        return m, m, m

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_rounds, n))
    bootstrap_means = np.mean(arr[indices], axis=1)

    mean_val = float(np.mean(arr))
    alpha = (1.0 - ci) / 2.0
    lower_bound = float(np.percentile(bootstrap_means, alpha * 100))
    upper_bound = float(np.percentile(bootstrap_means, (1.0 - alpha) * 100))

    return mean_val, lower_bound, upper_bound
