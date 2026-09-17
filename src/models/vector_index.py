"""
High-performance FAISS vector index for cosine similarity candidate retrieval (Q3).
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss

class VectorIndex:
    """FAISS-backed Inner Product / Cosine Similarity Index."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.doc_ids: List[str] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        self.matrix: Optional[np.ndarray] = None

    def build(self, embeddings: Dict[str, np.ndarray]):
        """
        Builds the FAISS index from {article_id: vector}.
        """
        if not embeddings:
            return

        self.doc_ids = list(embeddings.keys())
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        
        raw_mat = np.array([embeddings[doc_id] for doc_id in self.doc_ids], dtype=np.float32)
        # Ensure L2 normalized for cosine similarity
        faiss.normalize_L2(raw_mat)
        
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(raw_mat)
        self.matrix = raw_mat

    def search(self, query_vectors: np.ndarray, k: int = 200) -> Tuple[List[List[str]], np.ndarray]:
        """
        Batch searches top-K items for multiple query vectors.
        query_vectors: shape (N, dim)
        Returns: (top_k_doc_ids_per_query, top_k_scores_per_query)
        """
        if self.index.ntotal == 0 or len(query_vectors) == 0:
            return [], np.zeros((0, k), dtype=np.float32)

        q_mat = np.array(query_vectors, dtype=np.float32)
        faiss.normalize_L2(q_mat)

        scores, indices = self.index.search(q_mat, min(k, self.index.ntotal))
        
        results_doc_ids = []
        for row in indices:
            row_ids = [self.doc_ids[idx] for idx in row if 0 <= idx < len(self.doc_ids)]
            results_doc_ids.append(row_ids)

        return results_doc_ids, scores

    def score_candidates_batch(
        self, query_vector: np.ndarray, candidate_doc_ids: List[str]
    ) -> List[float]:
        """
        Scores specific candidate articles against a single query vector using dot product.
        """
        if not candidate_doc_ids or self.matrix is None:
            return [0.0] * len(candidate_doc_ids)

        q_norm = query_vector / max(1e-9, np.linalg.norm(query_vector))
        
        scores = []
        for cid in candidate_doc_ids:
            idx = self.doc_id_to_idx.get(cid)
            if idx is not None:
                score = float(np.dot(q_norm, self.matrix[idx]))
            else:
                score = 0.0
            scores.append(score)
        return scores
