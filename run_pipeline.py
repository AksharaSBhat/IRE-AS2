"""
Single-command CLI to build, temporally validate, and cache the unified Feature Store.
"""
import argparse
import time
from pathlib import Path

from src.config import CACHE_DIR
from src.data.download import ensure_ebnerd_data, ensure_mind_data
from src.data.ebnerd_loader import EBNeRDLoader
from src.data.mind_loader import MINDLoader
from src.data.feature_store import FeatureStore
from src.utils.anti_gaming import check_future_click_leakage

def build_ebnerd_pipeline(variant: str = "demo"):
    t0 = time.time()
    print(f"\n=======================================================")
    print(f" Building Unified Data Pipeline: EB-NeRD ({variant})")
    print(f"=======================================================")

    data_dir = ensure_ebnerd_data(variant)
    loader = EBNeRDLoader(data_dir=data_dir)

    # 1. Load articles
    print("Loading articles...")
    articles = loader.load_articles()
    print(f" Loaded {len(articles):,} articles.")

    # 2. Load behaviors & history
    print("Loading train behaviors and history...")
    train_impressions = loader.load_behaviors(split="train")
    train_history = loader.load_history(split="train")
    print(f" Loaded {len(train_impressions):,} train impressions, {len(train_history):,} user histories.")

    print("Loading validation behaviors and history...")
    val_impressions = loader.load_behaviors(split="validation")
    val_history = loader.load_history(split="validation")
    print(f" Loaded {len(val_impressions):,} val impressions, {len(val_history):,} user histories.")

    # 3. Combine user histories
    all_histories = {}
    all_histories.update(train_history)
    all_histories.update(val_history)

    # 4. Temporal Split Analysis & Anti-Gaming Check
    print("Analyzing temporal split & checking raw history leakage...")
    train_times = [imp.impression_time for imp in train_impressions if imp.impression_time]
    val_times = [imp.impression_time for imp in val_impressions if imp.impression_time]
    if train_times and val_times:
        print(f"  Train time window: {min(train_times)} to {max(train_times)}")
        print(f"  Val time window:   {min(val_times)} to {max(val_times)}")
        if min(val_times) >= min(train_times):
            print("  Temporal order verified: Validation follows Train.")

    raw_leaks, samples = check_future_click_leakage(train_impressions, all_histories)
    if raw_leaks > 0:
        print(f"  [Q9 Anti-Gaming Insight] Raw history contains {raw_leaks:,} future interactions.")
        print(f"  FeatureStore enforces strict temporal boundary (timestamp < impression_time) to prevent leakage.")
    else:
        print("  Clean history: No future interactions found.")

    # 5. Build Feature Store
    print("Constructing Feature Store & computing popularity...")
    fs = FeatureStore(name=f"ebnerd_{variant}")
    fs.add_articles(articles)
    fs.add_user_histories(all_histories)
    fs.compute_popularity(train_impressions)

    cache_path = CACHE_DIR / f"ebnerd_{variant}_fs.pkl"
    fs.save(cache_path)
    print(f" Feature Store saved successfully to: {cache_path}")

    elapsed = time.time() - t0
    print(f"\n--- EB-NeRD ({variant}) Pipeline Completed in {elapsed:.2f}s ---")
    print(f"Total Articles: {len(articles):,}")
    print(f"Total Unique Users: {len(all_histories):,}")
    print(f"Train Impressions: {len(train_impressions):,}")
    print(f"Val Impressions: {len(val_impressions):,}")
    print(f"Total Click Counts: {fs.total_clicks:,}")

def build_mind_pipeline(variant: str = "small"):
    t0 = time.time()
    print(f"\n=======================================================")
    print(f" Building Unified Data Pipeline: MIND ({variant})")
    print(f"=======================================================")

    train_dir, dev_dir = ensure_mind_data(variant)
    
    # 1. Load Train
    print("Loading MIND train split...")
    train_loader = MINDLoader(data_dir=train_dir)
    train_articles = train_loader.load_articles()
    train_impressions, train_histories = train_loader.load_behaviors()
    print(f" Train: {len(train_articles):,} articles, {len(train_impressions):,} impressions, {len(train_histories):,} user histories.")

    # 2. Load Dev
    print("Loading MIND dev split...")
    dev_loader = MINDLoader(data_dir=dev_dir)
    dev_articles = dev_loader.load_articles()
    dev_impressions, dev_histories = dev_loader.load_behaviors()
    print(f" Dev: {len(dev_articles):,} articles, {len(dev_impressions):,} impressions, {len(dev_histories):,} user histories.")

    # 3. Combine Articles & User Histories
    all_articles = {}
    all_articles.update(train_articles)
    all_articles.update(dev_articles)

    all_histories = {}
    all_histories.update(train_histories)
    all_histories.update(dev_histories)

    # 4. Temporal Split Verification
    print("Verifying temporal separation...")
    train_times = [imp.impression_time for imp in train_impressions if imp.impression_time]
    dev_times = [imp.impression_time for imp in dev_impressions if imp.impression_time]
    if train_times and dev_times:
        print(f"  Train time window: {min(train_times)} to {max(train_times)}")
        print(f"  Dev time window:   {min(dev_times)} to {max(dev_times)}")
        if min(dev_times) >= min(train_times):
            print("  Temporal order verified: Dev set follows Train set.")

    # 5. Build Feature Store
    print("Constructing Feature Store & computing popularity...")
    fs = FeatureStore(name=f"mind_{variant}")
    fs.add_articles(all_articles)
    fs.add_user_histories(all_histories)
    fs.compute_popularity(train_impressions)

    cache_path = CACHE_DIR / f"mind_{variant}_fs.pkl"
    fs.save(cache_path)
    print(f" Feature Store saved successfully to: {cache_path}")

    elapsed = time.time() - t0
    print(f"\n--- MIND ({variant}) Pipeline Completed in {elapsed:.2f}s ---")
    print(f"Total Articles: {len(all_articles):,}")
    print(f"Total Unique Users: {len(all_histories):,}")
    print(f"Train Impressions: {len(train_impressions):,}")
    print(f"Dev Impressions: {len(dev_impressions):,}")
    print(f"Total Click Counts: {fs.total_clicks:,}")

def main():
    parser = argparse.ArgumentParser(description="One-command reproducible data pipeline and feature store builder.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["ebnerd_demo", "ebnerd_small", "mind_small", "all"],
        help="Dataset pipeline to build."
    )
    args = parser.parse_args()

    if args.dataset in ("ebnerd_demo", "all"):
        build_ebnerd_pipeline("demo")
    if args.dataset in ("ebnerd_small", "all"):
        build_ebnerd_pipeline("small")
    if args.dataset in ("mind_small", "all"):
        build_mind_pipeline("small")

if __name__ == "__main__":
    main()
