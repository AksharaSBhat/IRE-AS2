"""
Anti-Gaming assertion tests (Q9).
Asserts that no user clicks occurring at or after impression time are visible.
"""
import pytest
from datetime import datetime, timedelta

from src.data.schema import ImpressionRecord, UserHistoryRecord
from src.utils.anti_gaming import filter_history_before_time, assert_no_future_click_leakage
from src.features.feature_extractor import FeatureExtractor
from src.data.feature_store import FeatureStore

def test_temporal_boundary_filter():
    base_time = datetime(2024, 5, 20, 12, 0, 0)
    
    history = UserHistoryRecord(
        user_id="U1",
        history_article_ids=["past1", "past2", "future1", "future2"],
        history_timestamps=[
            base_time - timedelta(days=2),
            base_time - timedelta(hours=1),
            base_time + timedelta(hours=1),
            base_time + timedelta(days=1),
        ],
        history_durations=[10.0, 20.0, 30.0, 40.0]
    )

    clean_hist = filter_history_before_time(history, cutoff_time=base_time)
    assert clean_hist.history_article_ids == ["past1", "past2"]
    assert len(clean_hist.history_timestamps) == 2
    for t in clean_hist.history_timestamps:
        assert t < base_time

def test_feature_extractor_temporal_leakage_guard():
    base_time = datetime(2024, 5, 20, 12, 0, 0)
    fs = FeatureStore(name="test_fs")
    
    history = UserHistoryRecord(
        user_id="U_Test",
        history_article_ids=["past_art", "future_leak_art"],
        history_timestamps=[
            base_time - timedelta(hours=2),
            base_time + timedelta(hours=3)
        ],
        history_durations=[30.0, 60.0]
    )
    fs.add_user_histories({"U_Test": history})

    fe = FeatureExtractor(feature_store=fs)
    imp = ImpressionRecord(
        impression_id=1,
        user_id="U_Test",
        impression_time=base_time,
        inview_article_ids=["past_art", "cand1"],
        clicked_article_ids=["cand1"],
        labels=[0, 1]
    )

    X, y, cands = fe.extract_impression_features(imp)
    # user_click_count is feature index 0
    # Must only count past_art (count = 1), future_leak_art must be pruned!
    assert X[0, 0] == 1.0
    assert X[1, 0] == 1.0

def test_assert_no_future_click_leakage_detector():
    base_time = datetime(2024, 5, 20, 12, 0, 0)
    
    # 1. Contaminated history
    leaky_hist = UserHistoryRecord(
        user_id="U_Leaky",
        history_article_ids=["past", "future"],
        history_timestamps=[base_time - timedelta(hours=1), base_time + timedelta(hours=1)]
    )
    imp = ImpressionRecord(
        impression_id=101,
        user_id="U_Leaky",
        impression_time=base_time,
        inview_article_ids=["cand1"],
        labels=[1]
    )
    
    with pytest.raises(AssertionError, match="Future-click leakage detected"):
        assert_no_future_click_leakage([imp], {"U_Leaky": leaky_hist})

    # 2. Causal filtered history passes
    clean_hist = filter_history_before_time(leaky_hist, cutoff_time=base_time)
    assert_no_future_click_leakage([imp], {"U_Leaky": clean_hist})

