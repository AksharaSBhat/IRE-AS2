"""
Configuration settings and constants for the IRE Assignment 2 pipeline:
Learning from Click-Logs on EB-NeRD and MIND.
"""
from pathlib import Path
import os

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "feature_store_cache"
MODELS_DIR = BASE_DIR / "models_cache"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

for d in [CACHE_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Raw dataset locations
EBNERD_DEMO_DIR = DATA_DIR / "ebnerd_demo"
EBNERD_SMALL_DIR = DATA_DIR / "ebnerd_small"
EBNERD_TESTSET_DIR = DATA_DIR / "ebnerd_testset" / "ebnerd_testset"
EBNERD_WORD2VEC_DIR = DATA_DIR / "Ekstra_Bladet_word2vec" / "Ekstra_Bladet_word2vec"

MIND_DIR = BASE_DIR / "mind_datasets"
MIND_SMALL_TRAIN_DIR = MIND_DIR / "MINDsmall_train"
MIND_SMALL_DEV_DIR = MIND_DIR / "MINDsmall_dev"
MIND_LARGE_TRAIN_DIR = MIND_DIR / "MINDlarge_train"
MIND_LARGE_TEST_DIR = MIND_DIR / "MINDlarge_test"

# BM25 Parameters
BM25_K1 = 1.5
BM25_B = 0.75
BM25_HISTORY_MAX_ARTICLES = 5

# Re-ranking Parameters
RERANK_TOP_K = 100
FRESHNESS_HALF_LIFE_HOURS = 24.0  # News freshness decay
RECENCY_DECAY_LAMBDA = 0.05       # User history exponential decay factor
MAX_HISTORY_LEN = 30
DWELL_TIME_CAP_SEC = 300.0        # Cap extreme dwell times

# LightGBM Re-Ranker hyperparameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "random_state": 42,
    "n_estimators": 120
}

# Evaluation Parameters
RECALL_K_VALUES = [50, 100, 200]
NDCG_K_VALUES = [5, 10]
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_CI = 0.95

# Slicing thresholds
COLD_START_THRESHOLD = 5      # <= 5 historical clicks is cold start
POPULAR_HEAD_QUANTILE = 0.8  # Top 20% most clicked articles are head
