"""
Re-Ranker Models (Assignment 2 - Q2 & Q3).
Implements GBDT Ranking models using LightGBM, Baseline vs. Improved Ranker,
and Ablation variants for systematic empirical evaluation.
"""
import pickle
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from src.data.schema import ImpressionRecord
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.config import LGBM_PARAMS, MODELS_DIR

class LGBMReranker:
    """
    Two-stage Re-ranking model using LightGBM over engineered behavioral features.
    """
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        name: str = "lgbm_reranker"
    ):
        self.params = params or LGBM_PARAMS.copy()
        self.name = name
        self.model = None
        self.feature_names = FEATURE_NAMES

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: Optional[List[int]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        groups_val: Optional[List[int]] = None
    ):
        """
        Trains the LightGBM ranker on candidate feature matrices.
        """
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)] if eval_set else None
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predicted click probability scores for candidate articles.
        """
        if self.model is None:
            raise ValueError(f"Model {self.name} is not fitted yet.")
        # Prob of positive class
        return self.model.predict_proba(X)[:, 1]

    def rank_impression(
        self,
        X: np.ndarray,
        candidate_ids: List[str]
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Scores and ranks candidates for a single impression.
        Returns (ranked_candidate_ids, sorted_scores, 1_based_ranks).
        """
        scores = self.predict_proba(X)
        order = np.argsort(-scores)  # Descending order

        ranked_ids = [candidate_ids[i] for i in order]
        ranked_scores = [float(scores[i]) for i in order]

        # 1-based ranks aligned with original candidate_ids order (for Codabench submission)
        # Ranks: rank of original candidate at index i (1 = highest score)
        ranks_original_order = [0] * len(candidate_ids)
        for rank, orig_idx in enumerate(order, start=1):
            ranks_original_order[orig_idx] = rank

        return ranked_ids, ranked_scores, ranks_original_order

    def save(self, path: Optional[Path] = None):
        save_path = path or (MODELS_DIR / f"{self.name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "LGBMReranker":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_reranker_variant(variant: str = "improved") -> Tuple[FeatureExtractor, LGBMReranker]:
    """
    Factory function to construct FeatureExtractor and LGBMReranker for:
    - 'baseline': Standard features without freshness decay or dwell weighting
    - 'improved': Full features (freshness decay + dwell weighting + category match)
    - 'no_freshness': Improved variant with freshness features disabled
    - 'no_dwell': Improved variant with dwell-time weighting disabled
    - 'no_category': Improved variant with category affinity disabled
    - 'no_cross_sim': Improved variant with semantic cross-similarity disabled
    """
    flags = {
        "baseline": {
            "include_freshness": False,
            "include_dwell": False,
            "include_category_match": True,
            "include_cross_sim": True,
            "include_bm25": True
        },
        "improved": {
            "include_freshness": True,
            "include_dwell": True,
            "include_category_match": True,
            "include_cross_sim": True,
            "include_bm25": True
        },
        "no_freshness": {
            "include_freshness": False,
            "include_dwell": True,
            "include_category_match": True,
            "include_cross_sim": True,
            "include_bm25": True
        },
        "no_dwell": {
            "include_freshness": True,
            "include_dwell": False,
            "include_category_match": True,
            "include_cross_sim": True,
            "include_bm25": True
        },
        "no_category": {
            "include_freshness": True,
            "include_dwell": True,
            "include_category_match": False,
            "include_cross_sim": True,
            "include_bm25": True
        },
        "no_cross_sim": {
            "include_freshness": True,
            "include_dwell": True,
            "include_category_match": True,
            "include_cross_sim": False,
            "include_bm25": True
        }
    }

    cfg = flags.get(variant, flags["improved"])
    model = LGBMReranker(name=f"lgbm_{variant}")
    return cfg, model
