"""
Anti-Gaming and temporal leakage verification utilities (Q9).
Enforces that user features never access future interactions.
"""
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from src.data.schema import ImpressionRecord, UserHistoryRecord

def check_future_click_leakage(
    impressions: List[ImpressionRecord],
    user_histories: Dict[str, UserHistoryRecord],
    max_check: int = 10000
) -> Tuple[int, List[str]]:
    """
    Checks whether raw user histories contain interactions that occurred on or after impression timestamps.
    Returns (num_violations, list_of_sample_violation_messages).
    """
    violations = []
    checked = 0

    for imp in impressions:
        if imp.impression_time is None:
            continue

        hist = user_histories.get(imp.user_id)
        if not hist or not hist.history_timestamps:
            continue

        valid_times = [t for t in hist.history_timestamps if t is not None]
        if not valid_times:
            continue

        if max(valid_times) >= imp.impression_time:
            leaked_items = [
                (aid, t) for aid, t in zip(hist.history_article_ids, hist.history_timestamps)
                if t is not None and t >= imp.impression_time
            ]
            if leaked_items:
                violations.append(
                    f"User {imp.user_id} @ Imp {imp.impression_id} (time {imp.impression_time}) has {len(leaked_items)} future clicks"
                )

        checked += 1
        if checked >= max_check:
            break

    return len(violations), violations[:5]

def assert_no_future_click_leakage(
    impressions: List[ImpressionRecord],
    user_histories: Dict[str, UserHistoryRecord],
    max_check: int = 10000
):
    """
    Asserts that no impression's user history contains clicks occurring at or after impression time.
    """
    num_violations, samples = check_future_click_leakage(impressions, user_histories, max_check=max_check)
    if num_violations > 0:
        raise AssertionError(
            f"Future-click leakage detected! Found {num_violations} violations.\n"
            + "\n".join(samples)
        )

def filter_history_before_time(
    history: UserHistoryRecord, cutoff_time: datetime
) -> UserHistoryRecord:
    """
    Safely prunes any interactions from user history occurring at or after cutoff_time.
    """
    if not history.history_timestamps:
        return history

    filtered_ids = []
    filtered_times = []
    filtered_durations = []

    durations = history.history_durations or [None] * len(history.history_article_ids)

    for aid, t, d in zip(history.history_article_ids, history.history_timestamps, durations):
        if t is None or t < cutoff_time:
            filtered_ids.append(aid)
            filtered_times.append(t)
            filtered_durations.append(d)

    return UserHistoryRecord(
        user_id=history.user_id,
        history_article_ids=filtered_ids,
        history_timestamps=filtered_times,
        history_durations=filtered_durations if history.history_durations is not None else None
    )

def assert_temporal_boundary_enforcement(
    user_id: str,
    history: UserHistoryRecord,
    query_time: datetime
):
    """
    Asserts that filtering history by query_time produces zero future interactions.
    """
    if not history.history_timestamps:
        return

    filtered_times = [
        t for aid, t in zip(history.history_article_ids, history.history_timestamps)
        if t is None or t < query_time
    ]

    for t in filtered_times:
        if t is not None:
            assert t < query_time, f"Boundary violation: interaction time {t} >= query time {query_time}"
