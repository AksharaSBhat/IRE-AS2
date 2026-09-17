"""
Serving & Scale Analysis Benchmark (Assignment 2 - Q4).
Measures index memory footprint, end-to-end request latency percentiles (p50/p95/p99),
throughput (QPS), and cost-per-1000 queries at SLA, plus 10x scale breakdown.
"""
import time
import json
import numpy as np
from pathlib import Path

from src.config import CACHE_DIR, MODELS_DIR, EBNERD_WORD2VEC_DIR
from src.data.feature_store import FeatureStore
from src.data.ebnerd_loader import EBNeRDLoader
from src.models.embedding_loader import load_ebnerd_embeddings
from src.features.feature_extractor import FeatureExtractor
from src.models.reranker import LGBMReranker
from src.serving.profiler import ServingProfiler

def run_serving_benchmark():
    print("\n=======================================================")
    print(" Running Serving & Scale Analysis Benchmark (Q4)")
    print("=======================================================")

    # 1. Load FeatureStore, Embeddings & Trained Model
    fs_path = CACHE_DIR / "ebnerd_small_fs.pkl"
    print(f"Loading FeatureStore from {fs_path}...")
    fs = FeatureStore.load(fs_path)

    print("Loading Embeddings...")
    embeddings = load_ebnerd_embeddings(EBNERD_WORD2VEC_DIR)

    model_path = MODELS_DIR / "ebnerd_lgbm_improved.pkl"
    print(f"Loading trained Improved Ranker from {model_path}...")
    reranker = LGBMReranker.load(model_path)

    fe = FeatureExtractor(
        feature_store=fs,
        embeddings=embeddings,
        article_popularities={aid: fs.get_article_popularity(aid) for aid in fs.articles.keys()}
    )

    # 2. Initialize Profiler
    profiler = ServingProfiler(
        feature_extractor=fe,
        reranker=reranker,
        target_sla_ms=100.0,
        aws_hourly_rate=0.34  # AWS c6i.2xlarge (8 vCPUs, 16 GB RAM)
    )

    # 3. Measure Index & Feature Store Memory
    print("\n--- 1. Memory Footprint Analysis ---")
    mem_stats = profiler.measure_memory_footprint()
    for k, v in mem_stats.items():
        print(f"  {k:<28}: {v}")

    # 4. Latency Benchmark
    print("\n--- 2. End-to-End Latency Benchmark (500 trials) ---")
    from src.config import EBNERD_SMALL_DIR
    loader = EBNeRDLoader(data_dir=EBNERD_SMALL_DIR)
    val_imps = loader.load_behaviors(split="validation")[:500]
    lat_stats = profiler.benchmark_latency(val_imps, n_warmup=25, n_trials=500)

    print(f"  Mean Request Latency      : {lat_stats['mean_latency_ms']:.2f} ms")
    print(f"  p50 (Median) Latency      : {lat_stats['p50_latency_ms']:.2f} ms")
    print(f"  p90 Latency               : {lat_stats['p90_latency_ms']:.2f} ms")
    print(f"  p95 Latency               : {lat_stats['p95_latency_ms']:.2f} ms")
    print(f"  p99 Latency               : {lat_stats['p99_latency_ms']:.2f} ms")
    print(f"  Max Latency               : {lat_stats['max_latency_ms']:.2f} ms")
    print(f"  Stage 1 (Retrieval)       : {lat_stats['stage1_retrieval_ms']:.2f} ms")
    print(f"  Stage 2 (Feature Extract) : {lat_stats['stage2_feature_extract_ms']:.2f} ms")
    print(f"  Stage 2 (GBDT Inference)  : {lat_stats['stage2_inference_ms']:.2f} ms")
    print(f"  Target SLA (< 100ms)      : {'MET (PASS)' if lat_stats['sla_achieved'] else 'VIOLATED (FAIL)'}")

    # 5. Cost & QPS Economics
    print("\n--- 3. Throughput & Serving Cost Economics ---")
    print(f"  Single-Thread QPS         : {lat_stats['single_thread_qps']:.1f} queries/sec")
    print(f"  8-vCPU Instance QPS (70%) : {lat_stats['node_qps_8vcpu']:.1f} queries/sec")
    print(f"  Cost per 1,000 Queries    : ${lat_stats['cost_per_1k_queries_usd']:.6f} USD")
    print(f"  Monthly Cost for 10M reqs : ${lat_stats['cost_per_1k_queries_usd'] * 10000:.2f} USD")

    # 6. 10x Scale Breakdown
    print("\n--- 4. 10x Scale Breakdown & Mitigation Architecture ---")
    scale_analysis = {
        "current_scale": {
            "catalog_articles": len(fs.articles),
            "users": len(fs.user_histories),
            "serving_memory_mb": mem_stats["total_serving_memory_mb"],
            "p99_latency_ms": lat_stats["p99_latency_ms"]
        },
        "10x_scale": {
            "catalog_articles": len(fs.articles) * 10,
            "users": len(fs.user_histories) * 10,
            "serving_memory_mb": round(mem_stats["total_serving_memory_mb"] * 10, 2),
            "projected_cost_per_1k": lat_stats["cost_per_1k_queries_usd"]
        },
        "bottlenecks": [
            {
                "subsystem": "In-Memory Feature Store",
                "current_behavior": f"Stores all {len(fs.user_histories):,} user histories in local Python process memory (~{mem_stats['histories_memory_mb']} MB).",
                "failure_point_at_10x": "At 2M+ users and 10x history volume, Python dict memory exceeds 16GB, causing OOM crashes and high garbage collection latency pauses.",
                "mitigation": "Migrate to a distributed key-value store (Redis / Dragonfly) with LRU/TTL sliding-window eviction (retaining only last 30 days)."
            },
            {
                "subsystem": "Vector ANN Index",
                "current_behavior": f"Loads all {len(embeddings):,} 300-d vectors in local RAM.",
                "failure_point_at_10x": "At 1.2M articles, brute-force lookup latency degrades linearly from 1ms to >15ms.",
                "mitigation": "Deploy sharded HNSW indices (Qdrant / Milvus / ScaNN) with Product Quantization (PQ8), compressing 300-d vectors by 4x."
            },
            {
                "subsystem": "Real-Time Feature Freshness",
                "current_behavior": "Batch periodic calculation of article click counts and freshness.",
                "failure_point_at_10x": "Breaking news viral loops occur within minutes; batch feature updates miss fresh click spikes.",
                "mitigation": "Streaming click ingestion pipeline via Apache Kafka -> Apache Flink streaming stateful aggregator -> Redis."
            }
        ]
    }

    for b in scale_analysis["bottlenecks"]:
        print(f"\n[Bottleneck: {b['subsystem']}]")
        print(f"  Failure: {b['failure_point_at_10x']}")
        print(f"  Mitigation: {b['mitigation']}")

    serving_data = {
        "memory_footprint": mem_stats,
        "latency_benchmark": lat_stats,
        "scale_analysis": scale_analysis
    }

    out_file = Path("results_serving.json")
    with open(out_file, "w") as f:
        json.dump(serving_data, f, indent=2)
    print(f"\nSaved serving profile & scale analysis to {out_file}")

if __name__ == "__main__":
    run_serving_benchmark()
