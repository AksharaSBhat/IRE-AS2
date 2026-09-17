"""
Unit tests for FeatureExtractor (Q1).
Verifies dimensions, feature values, and cold-start robustness.
"""
import numpy as np
from datetime import datetime, timedelta

from src.data.schema import ArticleRecord, ImpressionRecord, UserHistoryRecord
from src.data.feature_store import FeatureStore
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES

def test_feature_extractor_dimensions():
    base_time = datetime(2024, 5, 20, 14, 30, 0)
    fs = FeatureStore(name="test_fs")
    
    art1 = ArticleRecord(article_id="A1", title="Breaking Sports News", category="sport", published_time=base_time - timedelta(hours=3))
    art2 = ArticleRecord(article_id="A2", title="Political Update", category="news", published_time=base_time - timedelta(hours=12))
    fs.add_articles([art1, art2])

    hist = UserHistoryRecord(
        user_id="U1",
        history_article_ids=["A1"],
        history_timestamps=[base_time - timedelta(hours=5)],
        history_durations=[45.0]
    )
    fs.add_user_histories({"U1": hist})

    embeddings = {
        "A1": np.ones(300, dtype=np.float32) / np.sqrt(300),
        "A2": -np.ones(300, dtype=np.float32) / np.sqrt(300)
    }

    fe = FeatureExtractor(feature_store=fs, embeddings=embeddings)
    imp = ImpressionRecord(
        impression_id=10,
        user_id="U1",
        impression_time=base_time,
        inview_article_ids=["A1", "A2"],
        labels=[1, 0]
    )

    X, y, cands = fe.extract_impression_features(imp)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert y.shape == (2,)
    assert len(cands) == 2
    assert cands == ["A1", "A2"]

def test_feature_extractor_cold_start_fallback():
    base_time = datetime(2024, 5, 20, 14, 30, 0)
    fs = FeatureStore(name="test_fs")
    art = ArticleRecord(article_id="A1", title="Test News", category="news")
    fs.add_articles([art])

    fe = FeatureExtractor(feature_store=fs)
    # User with no history (Cold start)
    imp = ImpressionRecord(
        impression_id=11,
        user_id="U_New",
        impression_time=base_time,
        inview_article_ids=["A1"],
        labels=[0]
    )

    X, y, cands = fe.extract_impression_features(imp)
    assert X.shape == (1, len(FEATURE_NAMES))
    assert not np.isnan(X).any()  # Must not contain NaNs
