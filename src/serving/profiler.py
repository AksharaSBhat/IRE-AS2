"""
Production Serving and Scale Analysis Engine (Assignment 2 - Q4).
Benchmarks index memory, end-to-end p50/p95/p99 request latency,
throughput (QPS), and back-of-envelope cloud serving costs at target SLAs.
"""
import time
import sys
import os
import psutil
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.data.schema import ImpressionRecord
from src.features.feature_extractor import FeatureExtractor
from src.models.reranker import LGBMReranker

class ServingProfiler:
    """
    Profiles end-to-end retrieve-then-rank serving performance and cost economics.
    """
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        reranker: LGBMReranker,
        target_sla_ms: float = 100.0,
        aws_hourly_rate: float = 0.34  # AWS c6i.2xlarge (8 vCPUs, 16 GB RAM)
    ):
        self.fe = feature_extractor
        self.reranker = reranker
        self.target_sla_ms = target_sla_ms
        self.aws_hourly_rate = aws_hourly_rate

    def measure_memory_footprint(self) -> Dict[str, float]:
        """
        Measures memory consumption of ANN vectors, FeatureStore, and Ranker model in MB.
        """
        # FeatureStore articles and histories
        fs = self.fe.fs
        articles_count = len(fs.articles)
        histories_count = len(fs.user_histories)
        
        # Estimate articles memory (title, abstract, category)
        articles_mem_mb = (articles_count * 1024) / (1024 * 1024)  # ~1 KB per article record
        
        # Estimate user history memory (list of IDs and timestamps)
        histories_mem_mb = (histories_count * 512) / (1024 * 1024)  # ~512 bytes per history
        
        # Vector embeddings memory (N x D x 4 bytes)
        emb_count = len(self.fe.embeddings)
        dim = 300
        vectors_mem_mb = (emb_count * dim * 4) / (1024 * 1024)
        
        # FAISS HNSW graph overhead is ~1.5x of raw vectors
        ann_index_mem_mb = vectors_mem_mb * 1.5

        # LightGBM model footprint
        import pickle
        model_bytes = len(pickle.dumps(self.reranker))
        model_mem_mb = model_bytes / (1024 * 1024)

        total_mem_mb = articles_mem_mb + histories_mem_mb + ann_index_mem_mb + model_mem_mb

        return {
            "num_articles": articles_count,
            "num_user_histories": histories_count,
            "num_embeddings": emb_count,
            "articles_memory_mb": round(articles_mem_mb, 2),
            "histories_memory_mb": round(histories_mem_mb, 2),
            "vector_index_memory_mb": round(ann_index_mem_mb, 2),
            "model_memory_mb": round(model_mem_mb, 2),
            "total_serving_memory_mb": round(total_mem_mb, 2)
        }

    def benchmark_latency(
        self,
        sample_impressions: List[ImpressionRecord],
        n_warmup: int = 20,
        n_trials: int = 200
    ) -> Dict[str, Any]:
        """
        Measures end-to-end request latency (retrieval + feature extraction + re-ranking).
        Returns p50, p90, p95, p99, and max latency in milliseconds.
        """
        latencies_ms = []
        retrieval_latencies_ms = []
        feature_latencies_ms = []
        inference_latencies_ms = []

        if not sample_impressions:
            return {}

        n_samples = len(sample_impressions)

        # Warmup phase
        for i in range(min(n_warmup, n_samples)):
            imp = sample_impressions[i % n_samples]
            X, y, cands = self.fe.extract_impression_features(imp)
            if len(cands) > 0:
                _ = self.reranker.predict_proba(X)

        # Timed benchmark phase
        for i in range(n_trials):
            imp = sample_impressions[i % n_samples]
            
            t0 = time.perf_counter()
            # Stage 1: Candidate retrieval simulation (vector search lookup)
            t_ret_start = time.perf_counter()
            # Mock / lookup of candidate vectors
            _ = [self.fe.embeddings.get(aid) for aid in imp.inview_article_ids]
            t_ret_end = time.perf_counter()
            
            # Stage 2: Feature extraction
            t_feat_start = time.perf_counter()
            X, y, cands = self.fe.extract_impression_features(imp)
            t_feat_end = time.perf_counter()
            
            # Stage 3: Inference and ranking
            t_inf_start = time.perf_counter()
            if len(cands) > 0:
                _ = self.reranker.predict_proba(X)
            t_inf_end = time.perf_counter()
            
            t_total = time.perf_counter() - t0

            latencies_ms.append(t_total * 1000.0)
            retrieval_latencies_ms.append((t_ret_end - t_ret_start) * 1000.0)
            feature_latencies_ms.append((t_feat_end - t_feat_start) * 1000.0)
            inference_latencies_ms.append((t_inf_end - t_inf_start) * 1000.0)

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        max_lat = float(np.max(lat_arr))

        # Cost & QPS Economics
        # Single worker QPS: 1000 ms / mean_latency_ms
        single_thread_qps = 1000.0 / max(0.1, mean_lat)
        # On an 8-vCPU instance at 70% safe load factor
        node_qps = single_thread_qps * 8 * 0.70

        # Cost per 1,000 queries
        # Total queries per hour per node = node_qps * 3600
        queries_per_hour = max(1.0, node_qps * 3600.0)
        cost_per_1k_queries = (self.aws_hourly_rate / queries_per_hour) * 1000.0

        sla_met = p99 <= self.target_sla_ms

        return {
            "n_trials": n_trials,
            "mean_latency_ms": round(mean_lat, 2),
            "p50_latency_ms": round(p50, 2),
            "p90_latency_ms": round(p90, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "max_latency_ms": round(max_lat, 2),
            "stage1_retrieval_ms": round(float(np.mean(retrieval_latencies_ms)), 2),
            "stage2_feature_extract_ms": round(float(np.mean(feature_latencies_ms)), 2),
            "stage2_inference_ms": round(float(np.mean(inference_latencies_ms)), 2),
            "target_sla_ms": self.target_sla_ms,
            "sla_achieved": sla_met,
            "single_thread_qps": round(single_thread_qps, 1),
            "node_qps_8vcpu": round(node_qps, 1),
            "cost_per_1k_queries_usd": round(cost_per_1k_queries, 6)
        }
