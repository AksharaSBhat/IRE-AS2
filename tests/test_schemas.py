"""
Unit tests asserting schema consistency across MIND and EB-NeRD.
"""
from datetime import datetime
from src.data.schema import ArticleRecord, ImpressionRecord, UserHistoryRecord

def test_article_record_creation():
    art = ArticleRecord(
        article_id="12345",
        title="Breaking News",
        abstract="Summary of the news",
        category="news",
        subcategory="politics"
    )
    assert art.article_id == "12345"
    assert art.full_text == "Breaking News Summary of the news"

def test_impression_record_creation():
    imp = ImpressionRecord(
        impression_id=1,
        user_id="U1001",
        impression_time=datetime(2023, 5, 20, 12, 0, 0),
        inview_article_ids=["A1", "A2", "A3"],
        clicked_article_ids=["A2"],
        labels=[0, 1, 0]
    )
    assert imp.impression_id == 1
    assert len(imp.inview_article_ids) == len(imp.labels)
    assert imp.labels[1] == 1

def test_user_history_record():
    hist = UserHistoryRecord(
        user_id="U1001",
        history_article_ids=["A10", "A20"],
        history_timestamps=[datetime(2023, 5, 19, 10, 0, 0), datetime(2023, 5, 19, 15, 0, 0)]
    )
    assert len(hist.history_article_ids) == 2
    assert len(hist.history_timestamps) == 2
