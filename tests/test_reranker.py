"""
Unit tests for LGBMReranker and ranking factory (Q2 & Q3).
"""
import numpy as np
from src.models.reranker import LGBMReranker, build_reranker_variant

def test_lgbm_reranker_train_and_rank():
    # Synthetic dataset: 50 impressions, each with 5 candidates
    np.random.seed(42)
    n_samples = 250
    n_features = 18
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    # Binary labels correlated with feature index 13 (sim_rec)
    y_train = (X_train[:, 13] + np.random.randn(n_samples) * 0.5 > 0).astype(np.int32)

    reranker = LGBMReranker(params={"n_estimators": 10, "verbose": -1, "random_state": 42})
    reranker.train(X_train, y_train)

    # Test ranking on single impression
    test_X = np.random.randn(4, n_features).astype(np.float32)
    test_cands = ["art1", "art2", "art3", "art4"]
    ranked_ids, ranked_scores, ranks = reranker.rank_impression(test_X, test_cands)

    assert len(ranked_ids) == 4
    assert len(ranked_scores) == 4
    assert len(ranks) == 4
    # Scores must be sorted descending
    assert all(ranked_scores[i] >= ranked_scores[i+1] for i in range(len(ranked_scores) - 1))
    # Ranks must be a permutation of 1..4
    assert sorted(ranks) == [1, 2, 3, 4]

def test_build_reranker_variant():
    for var in ["baseline", "improved", "no_freshness", "no_dwell", "no_category", "no_cross_sim"]:
        cfg, model = build_reranker_variant(var)
        assert isinstance(cfg, dict)
        assert model.name == f"lgbm_{var}"
