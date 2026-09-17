"""
Unit tests for ServingProfiler (Q4).
"""
import numpy as np
from src.data.schema import ArticleRecord, ImpressionRecord
from src.data.feature_store import FeatureStore
from src.features.feature_extractor import FeatureExtractor
from src.models.reranker import LGBMReranker
from src.serving.profiler import ServingProfiler

def test_serving_profiler():
    fs = FeatureStore(name="test")
    art = ArticleRecord(article_id="A1", title="Headline", category="news")
    fs.add_articles([art])

    fe = FeatureExtractor(feature_store=fs)
    # Fit dummy ranker
    ranker = LGBMReranker(params={"n_estimators": 5, "verbose": -1})
    X = np.random.randn(10, 18).astype(np.float32)
    y = np.array([0, 1] * 5, dtype=np.int32)
    ranker.train(X, y)

    profiler = ServingProfiler(feature_extractor=fe, reranker=ranker, target_sla_ms=100.0)

    # Memory test
    mem = profiler.measure_memory_footprint()
    assert "total_serving_memory_mb" in mem
    assert mem["total_serving_memory_mb"] > 0

    # Latency test
    imp = ImpressionRecord(impression_id=1, user_id="U1", inview_article_ids=["A1"], labels=[1])
    lat = profiler.benchmark_latency([imp], n_warmup=2, n_trials=10)
    assert "p99_latency_ms" in lat
    assert "single_thread_qps" in lat
    assert "cost_per_1k_queries_usd" in lat
