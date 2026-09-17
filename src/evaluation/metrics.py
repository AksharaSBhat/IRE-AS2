"""
Official accuracy and beyond-accuracy evaluation metrics (Q4).
Includes AUC, MRR, nDCG@5, nDCG@10, Intra-List Diversity, Novelty, and Coverage.
"""
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
import math
from sklearn.metrics import roc_auc_score

def compute_impression_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    """
    Computes ROC-AUC for a single impression.
    Returns None if all labels are 0 or all are 1.
    """
    y_true = np.array(labels)
    y_score = np.array(scores)
    
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)
    if pos_count == 0 or neg_count == 0:
        return None

    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None

def compute_impression_mrr(labels: List[int], scores: List[float]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) of the first clicked article for an impression.
    """
    # Sort indices descending by score
    sorted_indices = np.argsort(-np.array(scores))
    for rank, idx in enumerate(sorted_indices, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0

def compute_impression_ndcg(labels: List[int], scores: List[float], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (nDCG@K) for an impression.
    """
    y_true = np.array(labels)
    num_pos = int(np.sum(y_true == 1))
    if num_pos == 0:
        return 0.0

    # Ideal DCG@K
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, num_pos)))
    if idcg <= 0.0:
        return 0.0

    # Actual DCG@K
    sorted_indices = np.argsort(-np.array(scores))[:k]
    dcg = 0.0
    for rank, idx in enumerate(sorted_indices):
        if labels[idx] == 1:
            dcg += 1.0 / math.log2(rank + 2)

    return float(dcg / idcg)

def compute_intra_list_diversity(
    recommended_item_vectors: np.ndarray
) -> float:
    """
    Computes Intra-List Diversity (ILD) as average pairwise cosine distance (1 - cosine_sim).
    Expects normalized L2 embedding vectors.
    """
    k = len(recommended_item_vectors)
    if k <= 1:
        return 0.0

    # Pairwise cosine similarity matrix = vectors @ vectors.T
    sim_matrix = np.dot(recommended_item_vectors, recommended_item_vectors.T)
    # Cosine distance = 1 - sim
    dist_matrix = 1.0 - sim_matrix
    # Average of upper triangle
    triu_indices = np.triu_indices(k, k=1)
    return float(np.mean(dist_matrix[triu_indices]))

def compute_novelty(
    recommended_article_ids: List[str],
    item_probabilities: Dict[str, float]
) -> float:
    """
    Computes Novelty as average self-information: -log2(P(item)).
    """
    if not recommended_article_ids:
        return 0.0

    self_infos = []
    for aid in recommended_article_ids:
        prob = item_probabilities.get(aid, 1e-7)
        self_infos.append(-math.log2(max(1e-9, prob)))
    return float(np.mean(self_infos))

def compute_catalog_coverage(
    all_recommended_article_ids: Set[str],
    total_catalog_size: int
) -> float:
    """
    Computes Catalog Coverage: fraction of catalog items recommended across all users.
    """
    if total_catalog_size <= 0:
        return 0.0
    return float(len(all_recommended_article_ids) / total_catalog_size)

# Convenience aliases and macro aggregators
compute_mrr = compute_impression_mrr
compute_ndcg_at_k = compute_impression_ndcg

def compute_macro_auc(auc_list: List[Optional[float]]) -> float:
    """Computes mean AUC across all valid impressions."""
    valid = [a for a in auc_list if a is not None]
    return float(np.mean(valid)) if valid else 0.5

def evaluate_impression_list(
    impressions_labels: List[List[int]],
    impressions_scores: List[List[float]],
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """Evaluates a batch of impressions returning mean AUC, MRR, and nDCG@K."""
    aucs = []
    mrrs = []
    ndcgs = {k: [] for k in k_values}

    for y, s in zip(impressions_labels, impressions_scores):
        auc = compute_impression_auc(y, s)
        if auc is not None:
            aucs.append(auc)
        mrrs.append(compute_impression_mrr(y, s))
        for k in k_values:
            ndcgs[k].append(compute_impression_ndcg(y, s, k=k))

    res = {
        "AUC": float(np.mean(aucs)) if aucs else 0.5,
        "MRR": float(np.mean(mrrs)) if mrrs else 0.0
    }
    for k in k_values:
        res[f"nDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return res
