"""
Data slicing utilities for offline evaluation (Q4).
Supports slicing by User Warmth (Cold-Start vs. Warm) and Article Popularity (Head vs. Tail).
"""
from typing import List, Dict, Tuple, Set
import numpy as np

from src.data.schema import ImpressionRecord
from src.data.feature_store import FeatureStore
from src.config import COLD_START_THRESHOLD, POPULAR_HEAD_QUANTILE

def slice_by_user_warmth(
    impressions: List[ImpressionRecord],
    feature_store: FeatureStore,
    threshold: int = COLD_START_THRESHOLD
) -> Dict[str, List[ImpressionRecord]]:
    """
    Slices impressions into Cold-Start (<= threshold past clicks) and Warm (> threshold).
    """
    cold_slice = []
    warm_slice = []

    for imp in impressions:
        click_count = feature_store.get_user_click_count(imp.user_id)
        if click_count <= threshold:
            cold_slice.append(imp)
        else:
            warm_slice.append(imp)

    return {
        "Cold-Start (<= 5 clicks)": cold_slice,
        "Warm Users (> 5 clicks)": warm_slice
    }

def get_head_and_tail_articles(
    feature_store: FeatureStore,
    head_quantile: float = POPULAR_HEAD_QUANTILE
) -> Tuple[Set[str], Set[str]]:
    """
    Partitions all catalog articles into Head (top (1-head_quantile)) and Tail items.
    """
    click_counts = [feature_store.get_article_popularity(aid) for aid in feature_store.articles.keys()]
    if not click_counts or max(click_counts) == 0:
        return set(feature_store.articles.keys()), set()

    cutoff_val = float(np.quantile(click_counts, head_quantile))
    head_articles = {
        aid for aid in feature_store.articles.keys()
        if feature_store.get_article_popularity(aid) >= cutoff_val and feature_store.get_article_popularity(aid) > 0
    }
    tail_articles = set(feature_store.articles.keys()) - head_articles
    return head_articles, tail_articles

def slice_by_article_head_tail(
    impressions: List[ImpressionRecord],
    head_article_ids: Set[str]
) -> Dict[str, List[ImpressionRecord]]:
    """
    Slices impressions based on whether the clicked articles belong to Head or Tail.
    """
    head_impressions = []
    tail_impressions = []

    for imp in impressions:
        if not imp.clicked_article_ids:
            continue
        # If any clicked article is a head item -> head slice, else tail
        has_head = any(aid in head_article_ids for aid in imp.clicked_article_ids)
        if has_head:
            head_impressions.append(imp)
        else:
            tail_impressions.append(imp)

    return {
        "Head Articles Clicked": head_impressions,
        "Tail Articles Only": tail_impressions
    }
