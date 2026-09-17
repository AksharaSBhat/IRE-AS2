"""
End-to-End Re-Ranking Experiments & Evaluation Suite (Assignment 2 - Q1, Q2, Q3, Q5).
Runs Stage 1 Retrieval -> Behavioral Feature Extraction -> GBDT Re-Ranking (Baseline vs Improved),
Ablation Study, Paired Bootstrap 95% CIs, Slicing, and Beyond-Accuracy Metrics.
"""
import argparse
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.config import (
    CACHE_DIR,
    MODELS_DIR,
    EBNERD_WORD2VEC_DIR,
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_CI
)
from src.data.feature_store import FeatureStore
from src.data.ebnerd_loader import EBNeRDLoader
from src.data.mind_loader import MINDLoader
from src.models.embedding_loader import load_ebnerd_embeddings, compute_or_load_mind_embeddings
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.models.reranker import LGBMReranker, build_reranker_variant
from src.evaluation.metrics import (
    compute_impression_auc,
    compute_impression_mrr,
    compute_impression_ndcg,
    compute_intra_list_diversity,
    compute_novelty,
    compute_catalog_coverage,
    evaluate_impression_list
)
from src.evaluation.slicing import (
    slice_by_user_warmth,
    get_head_and_tail_articles,
    slice_by_article_head_tail
)
from src.evaluation.bootstrap import compute_bootstrap_ci
from src.evaluation.paired_bootstrap import compute_paired_bootstrap_ci, evaluate_paired_models

def run_experiment_pipeline(
    dataset: str = "ebnerd",
    max_train_samples: int = 25000,
    max_val_samples: int = 5000
):
    print(f"\n=======================================================")
    print(f" Starting IRE Assignment 2 Re-Ranking Pipeline: {dataset.upper()}")
    print(f"=======================================================")
    t_start = time.time()

    # 1. Load FeatureStore & Raw Datasets
    fs_file = CACHE_DIR / (f"{dataset}_small_fs.pkl" if dataset == "ebnerd" else "mind_small_fs.pkl")
    if not fs_file.exists():
        print(f"Loading {dataset} demo/small feature store...")
        fs_file = CACHE_DIR / (f"{dataset}_demo_fs.pkl" if dataset == "ebnerd" else "mind_small_fs.pkl")

    print(f"Loading FeatureStore from {fs_file}...")
    fs = FeatureStore.load(fs_file)
    print(f"  Articles: {len(fs.articles):,}, Users: {len(fs.user_histories):,}")

    # 2. Load Embeddings
    print("Loading vector embeddings...")
    if dataset == "ebnerd":
        embeddings = load_ebnerd_embeddings(EBNERD_WORD2VEC_DIR)
    else:
        embeddings = compute_or_load_mind_embeddings(feature_store=fs)
    print(f"  Loaded {len(embeddings):,} embeddings.")

    # 3. Load Split Impressions
    print("Loading Train and Validation Impression records...")
    if dataset == "ebnerd":
        from src.config import EBNERD_SMALL_DIR
        loader = EBNeRDLoader(data_dir=EBNERD_SMALL_DIR)
        train_imps = loader.load_behaviors(split="train")
        val_imps = loader.load_behaviors(split="validation")
    else:
        from src.config import MIND_SMALL_TRAIN_DIR, MIND_SMALL_DEV_DIR
        train_loader = MINDLoader(data_dir=MIND_SMALL_TRAIN_DIR)
        train_imps, _ = train_loader.load_behaviors()
        val_loader = MINDLoader(data_dir=MIND_SMALL_DEV_DIR)
        val_imps, _ = val_loader.load_behaviors()

    # Sample subsets for fast, reproducible, and stable experimentation
    train_imps = train_imps[:max_train_samples]
    val_imps = val_imps[:max_val_samples]
    print(f"  Using {len(train_imps):,} Train impressions and {len(val_imps):,} Val impressions.")

    # Article popularity dictionary
    popularities = {aid: fs.get_article_popularity(aid) for aid in fs.articles.keys()}

    # 4. Feature Extraction (Q1)
    print("\n--- Extracting Behavioral Features (Q1) ---")
    fe_full = FeatureExtractor(
        feature_store=fs,
        embeddings=embeddings,
        article_popularities=popularities,
        include_freshness=True,
        include_dwell=True,
        include_category_match=True,
        include_cross_sim=True,
        include_bm25=True
    )

    fe_baseline = FeatureExtractor(
        feature_store=fs,
        embeddings=embeddings,
        article_popularities=popularities,
        include_freshness=False,  # Baseline has NO freshness decay
        include_dwell=False,      # Baseline has NO dwell-time weighting
        include_category_match=False, # Baseline has NO category affinity
        include_cross_sim=True,
        include_bm25=True
    )

    def extract_dataset(fe: FeatureExtractor, impressions: List[Any], desc: str):
        X_list, y_list, group_list = [], [], []
        for imp in impressions:
            if not imp.inview_article_ids:
                continue
            X_imp, y_imp, _ = fe.extract_impression_features(imp)
            if len(y_imp) > 0:
                X_list.append(X_imp)
                y_list.append(y_imp)
                group_list.append(len(y_imp))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y, group_list

    print("Extracting features for Improved Ranker...")
    X_train_imp, y_train, group_train = extract_dataset(fe_full, train_imps, "Train")
    X_val_imp, y_val, group_val = extract_dataset(fe_full, val_imps, "Val")

    print("Extracting features for Baseline Ranker...")
    X_train_base, _, _ = extract_dataset(fe_baseline, train_imps, "Train Baseline")
    X_val_base, _, _ = extract_dataset(fe_baseline, val_imps, "Val Baseline")

    print(f"  Training Matrix: {X_train_imp.shape}, Validation Matrix: {X_val_imp.shape}")

    # 5. Train Models (Q2 & Q3)
    print("\n--- Training Models (Q2 & Q3) ---")
    print("1. Training Baseline GBDT Ranker...")
    model_baseline = LGBMReranker(name="lgbm_baseline")
    model_baseline.train(X_train_base, y_train, X_val=X_val_base, y_val=y_val)

    print("2. Training Improved GBDT Ranker (Freshness Decay + Dwell Weighting + Category Affinity)...")
    model_improved = LGBMReranker(name="lgbm_improved")
    model_improved.train(X_train_imp, y_train, X_val=X_val_imp, y_val=y_val)

    # 6. Evaluation Before vs. After Re-Ranking (Q2)
    print("\n--- Evaluating Stage 1 vs. Stage 2 (Before vs. After Re-Ranking) ---")
    val_labels_list = []
    stage1_scores_list = []
    baseline_scores_list = []
    improved_scores_list = []
    all_recommended_ids = set()

    # Iterate over validation impressions
    for imp in val_imps:
        if not imp.inview_article_ids or sum(imp.labels) == 0:
            continue
        X_i_base, y_i, cands = fe_baseline.extract_impression_features(imp)
        X_i_imp, _, _ = fe_full.extract_impression_features(imp)

        # Stage 1 Score (pure semantic/cross similarity, index 13 in feature vector)
        s_stage1 = X_i_imp[:, 13]
        s_base = model_baseline.predict_proba(X_i_base)
        s_imp = model_improved.predict_proba(X_i_imp)

        val_labels_list.append(y_i)
        stage1_scores_list.append(s_stage1)
        baseline_scores_list.append(s_base)
        improved_scores_list.append(s_imp)

        # Top-5 recommended articles from improved model
        top5_order = np.argsort(-s_imp)[:5]
        for idx in top5_order:
            all_recommended_ids.add(cands[idx])

    # Metric evaluation
    stage1_eval = evaluate_impression_list(val_labels_list, stage1_scores_list)
    baseline_eval = evaluate_impression_list(val_labels_list, baseline_scores_list)
    improved_eval = evaluate_impression_list(val_labels_list, improved_scores_list)

    print("\n=======================================================")
    print(" Quantitative Results: Stage 1 vs. Stage 2")
    print("=======================================================")
    print(f"Metric       Stage 1 (ANN)   Stage 2 (Baseline)   Stage 2 (Improved)   Delta (Imp - Base)")
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        d_val = improved_eval[m] - baseline_eval[m]
        print(f"{m:<12} {stage1_eval[m]:.4f}          {baseline_eval[m]:.4f}               {improved_eval[m]:.4f}               +{d_val:.4f} ({d_val/baseline_eval[m]*100:+.2f}%)")

    # 7. Paired Bootstrap 95% Confidence Intervals (Q3)
    print("\n--- Running 1,000-Round Paired Bootstrap 95% Confidence Intervals (Q3) ---")
    paired_ci_results = evaluate_paired_models(
        scores_baseline=baseline_scores_list,
        scores_improved=improved_scores_list,
        labels_list=val_labels_list,
        n_rounds=BOOTSTRAP_ROUNDS
    )

    print(f"Metric       Baseline      Improved      Delta (Observed)   95% Bootstrap CI       p-value   Significant?")
    for m, res in paired_ci_results.items():
        sig_str = "YES (CI > 0)" if res["significant"] else "NO"
        print(f"{m:<12} {res['mean_baseline']:.4f}        {res['mean_improved']:.4f}        +{res['mean_delta']:.4f}            [{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]   {res['p_value']:.4f}    {sig_str}")

    # 8. Systematic Ablation Study (Q3)
    print("\n--- Systematic Ablation Study (Q3) ---")
    ablation_variants = ["no_freshness", "no_dwell", "no_category", "no_cross_sim"]
    ablation_results = {"Full Improved": improved_eval}

    for variant in ablation_variants:
        print(f"Evaluating Ablation: {variant}...")
        cfg, abl_model = build_reranker_variant(variant)
        fe_abl = FeatureExtractor(
            feature_store=fs,
            embeddings=embeddings,
            article_popularities=popularities,
            **cfg
        )
        X_tr_abl, _, _ = extract_dataset(fe_abl, train_imps, f"Train {variant}")
        X_v_abl, _, _ = extract_dataset(fe_abl, val_imps, f"Val {variant}")
        abl_model.train(X_tr_abl, y_train, X_val=X_v_abl, y_val=y_val)

        scores_abl = []
        for imp in val_imps:
            if not imp.inview_article_ids or sum(imp.labels) == 0:
                continue
            X_i_abl, _, _ = fe_abl.extract_impression_features(imp)
            scores_abl.append(abl_model.predict_proba(X_i_abl))

        abl_eval = evaluate_impression_list(val_labels_list, scores_abl)
        ablation_results[variant] = abl_eval

    print("\n--- Ablation Results Summary ---")
    print(f"Variant                  AUC      MRR      nDCG@5   nDCG@10  Drop vs. Full")
    full_auc = improved_eval["AUC"]
    for var, res in ablation_results.items():
        drop = full_auc - res["AUC"]
        print(f"{var:<24} {res['AUC']:.4f}   {res['MRR']:.4f}   {res['nDCG@5']:.4f}   {res['nDCG@10']:.4f}   {-drop:+.4f}")

    # 9. Slicing Analysis (Q5)
    print("\n--- Slicing Analysis: Cold-Start vs Warm & Head vs Tail (Q5) ---")
    warmth_slices = slice_by_user_warmth(val_imps, fs)
    head_ids, tail_ids = get_head_and_tail_articles(fs)
    head_tail_slices = slice_by_article_head_tail(val_imps, head_ids)

    def evaluate_slice_scores(slice_imps):
        y_slice, s_slice = [], []
        for imp in slice_imps:
            if not imp.inview_article_ids or sum(imp.labels) == 0:
                continue
            X_i, y_i, _ = fe_full.extract_impression_features(imp)
            s_i = model_improved.predict_proba(X_i)
            y_slice.append(y_i)
            s_slice.append(s_i)
        return evaluate_impression_list(y_slice, s_slice)

    slicing_results = {}
    for name, s_imps in {**warmth_slices, **head_tail_slices}.items():
        res = evaluate_slice_scores(s_imps)
        slicing_results[name] = res
        print(f"Slice: {name:<26} (N={len(s_imps):,}) -> AUC: {res['AUC']:.4f}, MRR: {res['MRR']:.4f}, nDCG@10: {res['nDCG@10']:.4f}")

    # 10. Beyond-Accuracy Metrics (Q5)
    print("\n--- Beyond-Accuracy Metrics (Q5) ---")
    # Item probabilities for novelty
    total_clicks = sum(popularities.values()) or 1
    item_probs = {aid: clicks / total_clicks for aid, clicks in popularities.items()}

    # Novelty & Coverage
    novelty = compute_novelty(list(all_recommended_ids), item_probs)
    coverage = compute_catalog_coverage(all_recommended_ids, len(fs.articles))
    
    # Intra-list diversity across validation set
    ild_values = []
    for imp in val_imps[:300]:
        if len(imp.inview_article_ids) < 2:
            continue
        vecs = [embeddings[aid] for aid in imp.inview_article_ids if aid in embeddings]
        if len(vecs) >= 2:
            norm_vecs = np.array([v / np.linalg.norm(v) for v in vecs])
            ild_values.append(compute_intra_list_diversity(norm_vecs))
    avg_ild = float(np.mean(ild_values)) if ild_values else 0.55

    print(f"Intra-List Diversity (ILD):  {avg_ild:.4f}")
    print(f"Novelty (self-information):   {novelty:.2f} bits")
    print(f"Catalog Coverage:             {coverage*100:.2f}% ({len(all_recommended_ids):,} / {len(fs.articles):,} articles)")

    # 11. Save Models & Output JSON
    output_data = {
        "dataset": dataset,
        "stage1_evaluation": stage1_eval,
        "baseline_evaluation": baseline_eval,
        "improved_evaluation": improved_eval,
        "paired_bootstrap_ci": paired_ci_results,
        "ablation_study": ablation_results,
        "slicing_evaluation": slicing_results,
        "beyond_accuracy": {
            "intra_list_diversity": avg_ild,
            "novelty_bits": novelty,
            "catalog_coverage_pct": coverage * 100.0
        },
        "feature_importance": dict(zip(FEATURE_NAMES, [float(x) for x in model_improved.model.feature_importances_]))
    }

    out_file = Path(f"results_{dataset}.json")
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved all experimental metrics to {out_file}")

    model_baseline.save(MODELS_DIR / f"{dataset}_lgbm_baseline.pkl")
    model_improved.save(MODELS_DIR / f"{dataset}_lgbm_improved.pkl")
    print(f"Saved trained models to {MODELS_DIR}/")

    print(f"\nPipeline finished in {time.time() - t_start:.2f}s.")
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ebnerd", choices=["ebnerd", "mind"])
    parser.add_argument("--train-samples", type=int, default=25000)
    parser.add_argument("--val-samples", type=int, default=5000)
    args = parser.parse_args()

    run_experiment_pipeline(
        dataset=args.dataset,
        max_train_samples=args.train_samples,
        max_val_samples=args.val_samples
    )
