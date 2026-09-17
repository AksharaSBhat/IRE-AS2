"""
Behavioral Feature Engineering Engine (Assignment 2 - Q1).
Extracts rich user-history, session, article, and cross-interaction features
with strict point-in-time causal boundary enforcement (Anti-Gaming Q9).
"""
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from collections import Counter

from src.data.schema import ArticleRecord, ImpressionRecord, UserHistoryRecord
from src.data.feature_store import FeatureStore
from src.utils.anti_gaming import filter_history_before_time
from src.config import (
    FRESHNESS_HALF_LIFE_HOURS,
    RECENCY_DECAY_LAMBDA,
    MAX_HISTORY_LEN,
    DWELL_TIME_CAP_SEC
)

FEATURE_NAMES = [
    # Click-history features
    "user_click_count",
    "user_avg_dwell",
    "user_category_entropy",
    # Session / Context features
    "cand_position",
    "cand_inview_count",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    # Article features
    "cand_freshness_hours",
    "cand_freshness_decay",
    "cand_popularity_clicks",
    "cand_title_length",
    "cand_abstract_length",
    # Cross (User x Candidate) features
    "sim_user_cand_recency",
    "sim_user_cand_dwell",
    "category_match",
    "user_category_affinity",
    "bm25_score"
]


class FeatureExtractor:
    """
    Engineers feature matrices from raw interaction logs and FeatureStore.
    Guarantees that no future interactions after impression_time are accessed.
    """
    def __init__(
        self,
        feature_store: FeatureStore,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        article_popularities: Optional[Dict[str, int]] = None,
        include_freshness: bool = True,
        include_dwell: bool = True,
        include_category_match: bool = True,
        include_cross_sim: bool = True,
        include_bm25: bool = True
    ):
        self.fs = feature_store
        self.embeddings = embeddings or {}
        self.article_popularities = article_popularities or {}
        self.include_freshness = include_freshness
        self.include_dwell = include_dwell
        self.include_category_match = include_category_match
        self.include_cross_sim = include_cross_sim
        self.include_bm25 = include_bm25

        # Compute global embedding centroid for cold-start fallback
        if self.embeddings:
            all_vecs = list(self.embeddings.values())
            self.global_centroid = np.mean(all_vecs, axis=0)
            norm = np.linalg.norm(self.global_centroid)
            if norm > 1e-8:
                self.global_centroid = self.global_centroid / norm
        else:
            self.global_centroid = np.zeros(300, dtype=np.float32)

    def compute_user_profile_vectors(
        self,
        history_article_ids: List[str],
        history_timestamps: Optional[List[datetime]],
        history_durations: Optional[List[float]],
        imp_time: Optional[datetime]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes (recency_weighted_vec, dwell_weighted_vec) for a user.
        Uses exponential time decay and dwell-time scaling.
        """
        if not history_article_ids or not self.embeddings:
            return self.global_centroid.copy(), self.global_centroid.copy()

        recency_vecs = []
        recency_weights = []
        dwell_vecs = []
        dwell_weights = []

        now = imp_time or datetime.now()

        for idx, aid in enumerate(history_article_ids[-MAX_HISTORY_LEN:]):
            if aid not in self.embeddings:
                continue
            vec = self.embeddings[aid]

            # 1. Recency weight
            if history_timestamps and idx < len(history_timestamps) and history_timestamps[idx] is not None:
                delta_h = max(0.0, (now - history_timestamps[idx]).total_seconds() / 3600.0)
                rec_w = math.exp(-RECENCY_DECAY_LAMBDA * delta_h)
            else:
                rec_w = 1.0 / (len(history_article_ids) - idx)

            recency_vecs.append(vec)
            recency_weights.append(rec_w)

            # 2. Dwell weight
            if history_durations and idx < len(history_durations) and history_durations[idx] is not None:
                dur = min(history_durations[idx], DWELL_TIME_CAP_SEC)
                dwell_w = rec_w * (1.0 + math.log1p(dur))
            else:
                dwell_w = rec_w

            dwell_vecs.append(vec)
            dwell_weights.append(dwell_w)

        # Compute recency profile
        if recency_vecs and sum(recency_weights) > 1e-8:
            rec_profile = np.average(recency_vecs, axis=0, weights=recency_weights)
            norm = np.linalg.norm(rec_profile)
            if norm > 1e-8:
                rec_profile = rec_profile / norm
        else:
            rec_profile = self.global_centroid.copy()

        # Compute dwell profile
        if dwell_vecs and sum(dwell_weights) > 1e-8:
            dwell_profile = np.average(dwell_vecs, axis=0, weights=dwell_weights)
            norm = np.linalg.norm(dwell_profile)
            if norm > 1e-8:
                dwell_profile = dwell_profile / norm
        else:
            dwell_profile = self.global_centroid.copy()

        return rec_profile, dwell_profile

    def extract_impression_features(
        self,
        imp: ImpressionRecord,
        bm25_scores: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extracts feature matrix X, binary labels y, and candidate article IDs
        for all inview articles in a single impression session.
        """
        imp_time = imp.impression_time or datetime.now()

        # Strictly enforce causal point-in-time boundary (Anti-Gaming Q9)
        raw_hist = self.fs.user_histories.get(imp.user_id)
        if raw_hist:
            clean_hist = filter_history_before_time(raw_hist, cutoff_time=imp_time)
            hist_aids = clean_hist.history_article_ids
            hist_times = clean_hist.history_timestamps
            hist_durs = clean_hist.history_durations
        else:
            hist_aids = []
            hist_times = None
            hist_durs = None

        click_count = len(hist_aids)

        # Average historical dwell time
        if hist_durs:
            valid_durs = [min(d, DWELL_TIME_CAP_SEC) for d in hist_durs if d is not None]
            avg_dwell = float(np.mean(valid_durs)) if valid_durs else 0.0
        else:
            avg_dwell = 0.0

        # Category distribution and entropy
        categories = []
        for aid in hist_aids:
            art = self.fs.articles.get(aid)
            if art and art.category:
                categories.append(art.category)
        
        cat_counts = Counter(categories)
        total_cats = len(categories)
        if total_cats > 0:
            probs = [c / total_cats for c in cat_counts.values()]
            cat_entropy = -sum(p * math.log2(p + 1e-12) for p in probs)
            top_category = cat_counts.most_common(1)[0][0]
        else:
            cat_entropy = 0.0
            top_category = ""

        # User profile vectors
        rec_profile, dwell_profile = self.compute_user_profile_vectors(
            hist_aids, hist_times, hist_durs, imp_time
        )

        # Temporal session context
        hour = imp_time.hour + imp_time.minute / 60.0
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        dow = float(imp_time.weekday())
        inview_count = float(len(imp.inview_article_ids))

        rows = []
        labels = []
        candidate_ids = []

        bm25_scores = bm25_scores or {}

        for pos, cand_id in enumerate(imp.inview_article_ids):
            cand_art = self.fs.articles.get(cand_id)
            lbl = imp.labels[pos] if pos < len(imp.labels) else 0

            # 1. Freshness
            if cand_art and cand_art.published_time:
                freshness_h = max(0.0, (imp_time - cand_art.published_time).total_seconds() / 3600.0)
            else:
                freshness_h = 24.0  # Default 1 day
            
            freshness_decay = math.exp(-freshness_h / FRESHNESS_HALF_LIFE_HOURS) if self.include_freshness else 0.0
            freshness_h_feat = math.log1p(freshness_h) if self.include_freshness else 0.0

            # 2. Popularity & lengths
            pop = float(self.article_popularities.get(cand_id, 0))
            title_len = float(len(cand_art.title.split())) if cand_art else 0.0
            abstract_len = float(len(cand_art.abstract.split())) if cand_art else 0.0

            # 3. Cross similarities
            if self.include_cross_sim and cand_id in self.embeddings:
                cand_vec = self.embeddings[cand_id]
                c_norm = np.linalg.norm(cand_vec)
                if c_norm > 1e-8:
                    cand_vec = cand_vec / c_norm
                sim_rec = float(np.dot(rec_profile, cand_vec))
                sim_dwell = float(np.dot(dwell_profile, cand_vec)) if self.include_dwell else sim_rec
            else:
                sim_rec = 0.0
                sim_dwell = 0.0

            # 4. Category affinity
            if self.include_category_match and cand_art and cand_art.category:
                cat_match = 1.0 if cand_art.category == top_category else 0.0
                cat_affinity = float(cat_counts.get(cand_art.category, 0)) / float(max(1, total_cats))
            else:
                cat_match = 0.0
                cat_affinity = 0.0

            # 5. BM25 retrieval score
            bm25_val = float(bm25_scores.get(cand_id, 0.0)) if self.include_bm25 else 0.0

            row = [
                float(click_count),
                float(avg_dwell) if self.include_dwell else 0.0,
                float(cat_entropy),
                float(pos),
                inview_count,
                hour_sin,
                hour_cos,
                dow,
                freshness_h_feat,
                freshness_decay,
                pop,
                title_len,
                abstract_len,
                sim_rec,
                sim_dwell,
                cat_match,
                cat_affinity,
                bm25_val
            ]

            rows.append(row)
            labels.append(lbl)
            candidate_ids.append(cand_id)

        X = np.array(rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        return X, y, candidate_ids
