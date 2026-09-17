"""
Semantic Candidate Generation and Embedding-based Retrieval pipeline (Q3).
"""
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import numpy as np

from src.models.vector_index import VectorIndex
from src.data.feature_store import FeatureStore
from src.data.schema import ImpressionRecord
from src.config import RECALL_K_VALUES, COLD_START_THRESHOLD

class SemanticRetriever:
    """End-to-end Semantic Candidate Generator using Mean-Pooled History Vectors & FAISS."""

    def __init__(
        self,
        vector_index: VectorIndex,
        embeddings: Dict[str, np.ndarray],
        feature_store: FeatureStore,
        dim: int = 300,
        decay_factor: float = 0.95
    ):
        self.vector_index = vector_index
        self.embeddings = embeddings
        self.feature_store = feature_store
        self.dim = dim
        self.decay_factor = decay_factor

        # Precompute global popularity fallback centroid
        sorted_popular = sorted(
            feature_store.articles.keys(),
            key=lambda aid: feature_store.get_article_popularity(aid),
            reverse=True
        )
        self.fallback_popular_ids = sorted_popular[:500]

        pop_vecs = [self.embeddings[aid] for aid in sorted_popular[:100] if aid in self.embeddings]
        if pop_vecs:
            self.global_centroid = np.mean(pop_vecs, axis=0)
            norm = np.linalg.norm(self.global_centroid)
            self.global_centroid = self.global_centroid / (norm if norm > 0 else 1.0)
        else:
            self.global_centroid = np.zeros((dim,), dtype=np.float32)

    def build_user_vector(
        self,
        user_id: str,
        before_time: Optional[datetime] = None,
        max_articles: int = 15
    ) -> np.ndarray:
        """
        Computes a user representation by pooling embeddings of clicked historical articles.
        Applies exponential decay weighting to recent interactions.
        """
        hist_ids = self.feature_store.get_user_history(user_id, before_time=before_time)
        if not hist_ids:
            return self.global_centroid

        recent_ids = hist_ids[-max_articles:]
        valid_vecs = []
        weights = []

        for idx, aid in enumerate(recent_ids):
            if aid in self.embeddings:
                valid_vecs.append(self.embeddings[aid])
                # Exponential decay weight: higher weight for recent clicks
                w = self.decay_factor ** (len(recent_ids) - 1 - idx)
                weights.append(w)

        if not valid_vecs:
            return self.global_centroid

        w_arr = np.array(weights, dtype=np.float32)[:, np.newaxis]
        pooled = np.sum(np.array(valid_vecs) * w_arr, axis=0)
        norm = np.linalg.norm(pooled)
        return pooled / (norm if norm > 0 else 1.0)

    def rank_candidates(
        self,
        user_id: str,
        candidate_article_ids: List[str],
        before_time: Optional[datetime] = None
    ) -> List[Tuple[str, float]]:
        """
        Ranks candidate articles by cosine similarity against the user representation.
        """
        if not candidate_article_ids:
            return []

        user_vec = self.build_user_vector(user_id, before_time=before_time)
        sim_scores = self.vector_index.score_candidates_batch(user_vec, candidate_article_ids)

        ranked = []
        for aid, score in zip(candidate_article_ids, sim_scores):
            pop = self.feature_store.get_article_popularity(aid)
            pop_bonus = 1e-6 * (pop / max(1.0, self.feature_store.total_clicks or 1.0))
            ranked.append((aid, score + pop_bonus))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def retrieve_global_top_k(
        self,
        user_id: str,
        k: int = 200,
        before_time: Optional[datetime] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieves top-K candidate articles from the entire corpus via FAISS.
        """
        user_vec = self.build_user_vector(user_id, before_time=before_time)
        doc_ids_batch, scores_batch = self.vector_index.search(user_vec[np.newaxis, :], k=k)
        
        if not doc_ids_batch or not doc_ids_batch[0]:
            return [(aid, 0.0) for aid in self.fallback_popular_ids[:k]]

        doc_ids = doc_ids_batch[0]
        scores = scores_batch[0]
        return [(doc_ids[i], float(scores[i])) for i in range(len(doc_ids))]

    def evaluate_recall_at_k(
        self,
        impressions: List[ImpressionRecord],
        k_values: List[int] = RECALL_K_VALUES
    ) -> Dict[str, float]:
        """
        Evaluates Semantic Recall@K over a collection of impressions.
        """
        recalls: Dict[int, List[float]] = {k: [] for k in k_values}
        
        for imp in impressions:
            if not imp.clicked_article_ids:
                continue

            clicked_set = set(imp.clicked_article_ids)
            retrieved = self.retrieve_global_top_k(imp.user_id, k=max(k_values), before_time=imp.impression_time)
            retrieved_ids = [aid for aid, _ in retrieved]

            for k in k_values:
                top_k_set = set(retrieved_ids[:k])
                hits = len(top_k_set.intersection(clicked_set))
                recalls[k].append(hits / len(clicked_set))

        results = {}
        for k in k_values:
            arr = recalls[k]
            results[f"Recall@{k}"] = float(np.mean(arr)) if arr else 0.0
        results["Evaluated_Impressions"] = len(next(iter(recalls.values())))
        return results
