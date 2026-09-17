"""
Constructs search queries from user historical interactions.
"""
from typing import List, Optional
from src.data.feature_store import FeatureStore
from src.utils.text import tokenize
from src.config import BM25_HISTORY_MAX_ARTICLES

class HistoryQueryBuilder:
    """Builds search queries from user click history."""

    def __init__(self, feature_store: FeatureStore, language: str = "english", max_articles: int = BM25_HISTORY_MAX_ARTICLES):
        self.feature_store = feature_store
        self.language = language
        self.max_articles = max_articles

    def build_query_tokens(self, user_id: str, custom_history: Optional[List[str]] = None) -> List[str]:
        """
        Extracts tokens from the user's recent clicked articles.
        """
        hist_ids = custom_history if custom_history is not None else self.feature_store.get_user_history(user_id)
        if not hist_ids:
            return []

        recent_ids = hist_ids[-self.max_articles:]
        texts = [self.feature_store.get_article_text(aid) for aid in recent_ids]
        combined_text = " ".join([t for t in texts if t])
        
        return tokenize(combined_text, language=self.language, remove_stopwords=True)

    def build_query_text(self, user_id: str, custom_history: Optional[List[str]] = None) -> str:
        """
        Returns raw text representation of user's recent history.
        """
        tokens = self.build_query_tokens(user_id, custom_history=custom_history)
        return " ".join(tokens)
