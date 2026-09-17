"""Unit tests for semantic vector indexing and retrieval."""
import pytest
import numpy as np
from datetime import datetime

from src.models.vector_index import VectorIndex
from src.models.semantic_retriever import SemanticRetriever
from src.data.feature_store import FeatureStore
from src.data.schema import ArticleRecord, UserHistoryRecord, ImpressionRecord

def test_vector_index_search():
    dim = 4
    v_index = VectorIndex(dim=dim)
    
    embeddings = {
        "A1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "A2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "A3": np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float32),
    }
    v_index.build(embeddings)
    assert v_index.index.ntotal == 3

    # Query matching A1 exactly
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    doc_ids, scores = v_index.search(query, k=2)
    assert doc_ids[0][0] == "A1"
    assert scores[0][0] > 0.99
    assert doc_ids[0][1] == "A3"

def test_semantic_retriever_ranking():
    dim = 4
    v_index = VectorIndex(dim=dim)
    embeddings = {
        "A1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "A2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }
    v_index.build(embeddings)

    fs = FeatureStore()
    fs.articles = {
        "A1": ArticleRecord(article_id="A1", title="Sports News"),
        "A2": ArticleRecord(article_id="A2", title="Finance News"),
    }
    t1 = datetime(2023, 1, 1, 10, 0, 0)
    fs.user_histories = {
        "U1": UserHistoryRecord(user_id="U1", history_article_ids=["A1"], history_timestamps=[t1])
    }

    retriever = SemanticRetriever(
        vector_index=v_index,
        embeddings=embeddings,
        feature_store=fs,
        dim=dim
    )

    ranked = retriever.rank_candidates("U1", ["A1", "A2"], before_time=datetime(2023, 1, 1, 12, 0, 0))
    assert len(ranked) == 2
    assert ranked[0][0] == "A1"  # A1 is in user history, similarity is 1.0
    assert ranked[1][0] == "A2"  # A2 is orthogonal, similarity is 0.0
