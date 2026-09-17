"""
Lexical Candidate Generation and Reranking pipeline using BM25.
"""
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.models.bm25 import BM25Index
from src.models.query_builder import HistoryQueryBuilder
from src.data.feature_store import FeatureStore
from src.data.schema import ImpressionRecord

class LexicalRetriever:
    """End-to-end BM25 lexical candidate generator and reranker."""

    def __init__(self, index: BM25Index, feature_store: FeatureStore, query_builder: HistoryQueryBuilder):
        self.index = index
        self.feature_store = feature_store
        self.query_builder = query_builder
        
        # Precompute popular articles for cold-start fallback
        sorted_popular = sorted(
            feature_store.articles.keys(),
            key=lambda aid: feature_store.get_article_popularity(aid),
            reverse=True
        )
        self.fallback_popular_ids = sorted_popular[:500]

    def rank_candidates(
        self, user_id: str, candidate_article_ids: List[str], custom_history: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Ranks candidate articles for an impression using BM25 scoring.
        """
        if not candidate_article_ids:
            return []

        query_tokens = self.query_builder.build_query_tokens(user_id, custom_history=custom_history)
        bm25_scores = self.index.score_candidates(query_tokens, candidate_article_ids)
        
        # Combined score with small popularity tie-breaker to handle zero-overlap cases
        ranked = []
        for aid, score in zip(candidate_article_ids, bm25_scores):
            pop = self.feature_store.get_article_popularity(aid)
            # Normalization scale for popularity tie breaker
            pop_score = 1e-6 * (pop / max(1.0, self.feature_store.total_clicks or 1.0))
            ranked.append((aid, score + pop_score))

        # Sort descending by score
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def retrieve_global_top_k(
        self, user_id: str, k: int = 200, custom_history: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieves global top-K articles from the entire corpus matching the user history.
        """
        query_tokens = self.query_builder.build_query_tokens(user_id, custom_history=custom_history)
        if not query_tokens:
            # Fallback to top popular articles
            return [(aid, float(self.feature_store.get_article_popularity(aid))) for aid in self.fallback_popular_ids[:k]]

        retrieved = self.index.retrieve_top_k(query_tokens, k=k)
        if len(retrieved) < k:
            # Pad with popular articles if needed
            existing_ids = {aid for aid, _ in retrieved}
            for pop_id in self.fallback_popular_ids:
                if pop_id not in existing_ids:
                    retrieved.append((pop_id, 0.0))
                if len(retrieved) >= k:
                    break
        return retrieved[:k]

    def evaluate_recall_at_k(
        self,
        impressions: List[ImpressionRecord],
        k_values: List[int] = [50, 100, 200],
        mode: str = "global"
    ) -> Dict[str, float]:
        """
        Evaluates Recall@K across a collection of validation impressions.
        
        mode: 'global' (retrieve top-K from entire corpus) or 'inview' (rerank inview candidates).
        """
        recalls: Dict[int, List[float]] = {k: [] for k in k_values}
        
        for imp in impressions:
            if not imp.clicked_article_ids:
                continue

            clicked_set = set(imp.clicked_article_ids)
            
            if mode == "global":
                top_items = self.retrieve_global_top_k(imp.user_id, k=max(k_values))
                ranked_ids = [aid for aid, _ in top_items]
            else:
                ranked = self.rank_candidates(imp.user_id, imp.inview_article_ids)
                ranked_ids = [aid for aid, _ in ranked]

            for k in k_values:
                top_k_ids = set(ranked_ids[:k])
                hits = len(top_k_ids.intersection(clicked_set))
                recalls[k].append(hits / len(clicked_set))

        results = {}
        for k in k_values:
            arr = recalls[k]
            results[f"Recall@{k}"] = float(np.mean(arr)) if arr else 0.0
            results[f"Evaluated_Impressions"] = len(arr)

        return results
