"""
Codabench Submission File Generator (Q5).
Generates prediction.zip for MIND and predictions.zip for EB-NeRD (RecSys 2024).
"""
import argparse
import time
import zipfile
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
import numpy as np

from src.config import BASE_DIR, CACHE_DIR
from src.data.feature_store import FeatureStore

SUBMISSION_DIR = BASE_DIR / "submissions"

def generate_mind_submission(batch_size: int = 200_000):
    print("\n=======================================================")
    print(" Generating MIND Codabench Submission (MINDlarge_test)")
    print("=======================================================")
    
    test_path = BASE_DIR / "mind_datasets" / "MINDlarge_test" / "behaviors.tsv"
    if not test_path.exists():
        print(f"MIND test file not found at {test_path}")
        return

    # Load MIND Feature Store for popularity and embeddings
    fs_path = CACHE_DIR / "mind_small_fs.pkl"
    store = FeatureStore.load(fs_path) if fs_path.exists() else None
    
    # Pre-build popularity dict
    pop_lookup = store.article_click_counts if store else {}
    max_pop = max(pop_lookup.values()) if pop_lookup else 1

    print(f"Reading test behaviors from {test_path}...")
    t0 = time.time()
    test_df = pl.read_csv(
        test_path,
        separator="\t",
        quote_char=None,
        has_header=False,
        new_columns=["impression_id", "user_id", "time", "history", "impressions"],
        schema_overrides={"impression_id": pl.Int64, "impressions": pl.Utf8}
    )
    n_rows = len(test_df)
    print(f" Loaded {n_rows:,} test impressions in {time.time() - t0:.2f}s.")

    raw_txt_path = SUBMISSION_DIR / "mind_prediction.txt"
    zip_path = SUBMISSION_DIR / "mind_prediction.zip"

    print(f"Scoring candidates and generating {raw_txt_path.name} in batches of {batch_size:,}...")
    t_gen = time.time()
    
    with open(raw_txt_path, "w") as f:
        for start_idx in range(0, n_rows, batch_size):
            batch = test_df.slice(start_idx, batch_size)
            imp_ids = batch["impression_id"].to_list()
            imps = batch["impressions"].to_list()

            lines = []
            for imp_id, raw_imp in zip(imp_ids, imps):
                if not raw_imp:
                    lines.append(f"{imp_id} []\n")
                    continue
                cand_ids = raw_imp.split(" ")
                # Score based on empirical click frequency
                scores = [pop_lookup.get(cid, 0) for cid in cand_ids]
                # Rank: 1 for highest score, 2 for next, etc.
                order = np.argsort(-np.array(scores))
                ranks = [0] * len(cand_ids)
                for rank, idx in enumerate(order, start=1):
                    ranks[idx] = rank
                ranks_str = ",".join(str(r) for r in ranks)
                lines.append(f"{imp_id} [{ranks_str}]\n")

            f.writelines(lines)

    print(f" Generated {n_rows:,} predictions in {time.time() - t_gen:.2f}s.")

    # Create submission zip containing prediction.txt
    print(f"Creating submission zip: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(raw_txt_path, arcname="prediction.txt")
    print(f" MIND Codabench submission ready at: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

def generate_ebnerd_submission():
    print("\n=======================================================")
    print(" Generating EB-NeRD Codabench Submission (ebnerd_testset)")
    print("=======================================================")

    test_behaviors = BASE_DIR / "data" / "ebnerd_testset" / "ebnerd_testset" / "test" / "behaviors.parquet"
    if not test_behaviors.exists():
        print(f"EB-NeRD test file not found at {test_behaviors}")
        return

    # Load popularity lookup
    fs_path = CACHE_DIR / "ebnerd_small_fs.pkl"
    if not fs_path.exists():
        fs_path = CACHE_DIR / "ebnerd_demo_fs.pkl"
    
    store = FeatureStore.load(fs_path) if fs_path.exists() else None
    pop_lookup = {int(k): v for k, v in store.article_click_counts.items()} if store else {}

    raw_txt_path = SUBMISSION_DIR / "ebnerd_predictions.txt"
    zip_path = SUBMISSION_DIR / "ebnerd_predictions.zip"

    print(f"Processing row groups from {test_behaviors.name}...")
    t0 = time.time()
    pf = pq.ParquetFile(test_behaviors)
    n_row_groups = pf.metadata.num_row_groups
    print(f" Total row groups: {n_row_groups}")

    total_written = 0
    with open(raw_txt_path, "w") as f:
        for rg in range(n_row_groups):
            table = pf.read_row_group(rg, columns=["impression_id", "article_ids_inview"])
            imp_ids = table.column("impression_id").to_pylist()
            inviews_list = table.column("article_ids_inview").to_pylist()

            lines = []
            for imp_id, inviews in zip(imp_ids, inviews_list):
                if not inviews:
                    lines.append(f"{imp_id} []\n")
                    continue
                scores = [pop_lookup.get(aid, 0) for aid in inviews]
                order = np.argsort(-np.array(scores))
                ranks = [0] * len(inviews)
                for rank, idx in enumerate(order, start=1):
                    ranks[idx] = rank
                ranks_str = ",".join(str(r) for r in ranks)
                lines.append(f"{imp_id} [{ranks_str}]\n")

            f.writelines(lines)
            total_written += len(imp_ids)

    print(f" Generated {total_written:,} predictions in {time.time() - t0:.2f}s.")

    # Create submission zip containing predictions.txt
    print(f"Creating submission zip: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(raw_txt_path, arcname="predictions.txt")
    print(f" EB-NeRD Codabench submission ready at: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Generate Codabench submission zip files.")
    parser.add_argument(
        "--competition",
        type=str,
        default="all",
        choices=["mind", "ebnerd", "all"],
        help="Competition to generate predictions for."
    )
    args = parser.parse_args()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    if args.competition in ("mind", "all"):
        generate_mind_submission()
    if args.competition in ("ebnerd", "all"):
        generate_ebnerd_submission()

if __name__ == "__main__":
    main()
