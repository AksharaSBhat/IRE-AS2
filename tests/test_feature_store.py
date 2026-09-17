"""
Unit tests for Feature Store persistence and lookups.
"""
from pathlib import Path
from src.data.feature_store import FeatureStore
from src.data.schema import ArticleRecord, UserHistoryRecord, ImpressionRecord

def test_feature_store_crud_and_persistence(tmp_path: Path):
    fs = FeatureStore(name="test_store")
    art = ArticleRecord(article_id="A1", title="Title", abstract="Abstract", category="Tech")
    fs.add_articles({"A1": art})
    
    hist = UserHistoryRecord(user_id="U1", history_article_ids=["A1"])
    fs.add_user_histories({"U1": hist})

    imp = ImpressionRecord(impression_id=1, user_id="U1", clicked_article_ids=["A1"], inview_article_ids=["A1", "A2"])
    fs.compute_popularity([imp])

    assert fs.get_article_text("A1") == "Title Abstract"
    assert fs.get_article_category("A1") == "Tech"
    assert fs.get_article_popularity("A1") == 1.0
    assert fs.get_user_click_count("U1") == 1
    assert fs.is_cold_start_user("U1", threshold=5) is True

    # Test serialization
    cache_file = tmp_path / "fs_test.pkl"
    fs.save(cache_file)
    assert cache_file.exists()

    loaded = FeatureStore.load(cache_file)
    assert loaded.get_article_text("A1") == "Title Abstract"
    assert loaded.get_article_popularity("A1") == 1.0
