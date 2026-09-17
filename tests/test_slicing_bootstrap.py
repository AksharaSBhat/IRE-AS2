"""Unit tests for slicing and bootstrap CI."""
import pytest
import numpy as np

from src.evaluation.bootstrap import compute_bootstrap_ci
from src.evaluation.slicing import slice_by_user_warmth, get_head_and_tail_articles
from src.data.feature_store import FeatureStore
from src.data.schema import ArticleRecord, ImpressionRecord, UserHistoryRecord

def test_bootstrap_ci():
    values = [0.5, 0.5, 0.5, 0.5, 0.5]
    m, l, u = compute_bootstrap_ci(values, n_rounds=100)
    assert m == 0.5
    assert l == 0.5
    assert u == 0.5

def test_slicing_warmth():
    fs = FeatureStore()
    fs.user_histories = {
        "U_cold": UserHistoryRecord(user_id="U_cold", history_article_ids=["A1", "A2"]),
        "U_warm": UserHistoryRecord(user_id="U_warm", history_article_ids=[f"A{i}" for i in range(10)])
    }
    
    imps = [
        ImpressionRecord(impression_id=1, user_id="U_cold"),
        ImpressionRecord(impression_id=2, user_id="U_warm")
    ]
    slices = slice_by_user_warmth(imps, fs, threshold=5)
    assert len(slices["Cold-Start (<= 5 clicks)"]) == 1
    assert len(slices["Warm Users (> 5 clicks)"]) == 1
