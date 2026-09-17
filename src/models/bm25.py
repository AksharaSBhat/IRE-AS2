"""
Optimized Inverted Index and BM25Okapi scoring implementation.
"""
from typing import Dict, List, Tuple, Optional, Set
import math
import numpy as np
from collections import Counter

from src.utils.text import tokenize
from src.config import BM25_K1, BM25_B

class BM25Index:
    """
    Inverted Index and BM25 scoring engine for news articles.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B, language: str = "english"):
        self.k1 = k1
        self.b = b
        self.language = language
        
        # Inverted index: term -> list of (doc_idx, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_lengths: List[int] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        self.idx_to_doc_id: List[str] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_len: float = 0.0
        self.num_docs: int = 0

    def fit(self, corpus: Dict[str, str]):
        """
        Builds inverted index and calculates IDF statistics from {article_id: text}.
        """
        self.num_docs = len(corpus)
        if self.num_docs == 0:
            return

        self.idx_to_doc_id = list(corpus.keys())
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.idx_to_doc_id)}
        self.doc_lengths = [0] * self.num_docs
        
        doc_freqs: Dict[str, int] = {}
        self.inverted_index = {}

        total_len = 0
        for doc_id, text in corpus.items():
            doc_idx = self.doc_id_to_idx[doc_id]
            tokens = tokenize(text, language=self.language, remove_stopwords=True)
            doc_len = len(tokens)
            self.doc_lengths[doc_idx] = doc_len
            total_len += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                    doc_freqs[term] = 0
                self.inverted_index[term].append((doc_idx, count))
                doc_freqs[term] += 1

        self.avg_doc_len = total_len / max(1, self.num_docs)

        # Calculate BM25 IDF: log(1 + (N - df + 0.5) / (df + 0.5))
        self.idf = {}
        for term, df in doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Computes BM25 score for a specific document index given query tokens."""
        if doc_idx < 0 or doc_idx >= self.num_docs or not query_tokens:
            return 0.0

        doc_len = self.doc_lengths[doc_idx]
        denom_part = self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len)))

        score = 0.0
        # We need term frequencies for this doc
        # For candidate scoring, term lookup is fast
        for term in query_tokens:
            if term in self.idf:
                idf_val = self.idf[term]
                # Find TF in doc
                tf = 0
                for d_idx, count in self.inverted_index.get(term, []):
                    if d_idx == doc_idx:
                        tf = count
                        break
                if tf > 0:
                    term_score = idf_val * (tf * (self.k1 + 1.0)) / (tf + denom_part)
                    score += term_score
        return score

    def score_candidates(self, query_tokens: List[str], candidate_doc_ids: List[str]) -> List[float]:
        """
        Scores a list of candidate document IDs against query tokens.
        """
        if not query_tokens or not candidate_doc_ids:
            return [0.0] * len(candidate_doc_ids)

        # Fast lookup: create map of candidate doc_idx -> position in list
        cand_indices = [self.doc_id_to_idx.get(cid, -1) for cid in candidate_doc_ids]
        doc_scores = [0.0] * len(candidate_doc_ids)
        
        cand_map: Dict[int, List[int]] = {}
        for pos, c_idx in enumerate(cand_indices):
            if c_idx != -1:
                cand_map.setdefault(c_idx, []).append(pos)

        # For terms in query, match against postings
        query_counts = Counter(query_tokens)
        for term, q_cnt in query_counts.items():
            if term not in self.inverted_index:
                continue
            idf_val = self.idf[term]
            postings = self.inverted_index[term]
            for doc_idx, tf in postings:
                if doc_idx in cand_map:
                    doc_len = self.doc_lengths[doc_idx]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len)))
                    contrib = idf_val * (tf * (self.k1 + 1.0)) / max(1e-6, denom)
                    for pos in cand_map[doc_idx]:
                        doc_scores[pos] += contrib

        return doc_scores

    def retrieve_top_k(self, query_tokens: List[str], k: int = 200) -> List[Tuple[str, float]]:
        """
        Retrieves the global top-K scoring document IDs across the entire index.
        """
        if not query_tokens or self.num_docs == 0:
            return []

        scores: Dict[int, float] = {}
        query_counts = Counter(query_tokens)

        for term in query_counts:
            if term not in self.inverted_index:
                continue
            idf_val = self.idf[term]
            for doc_idx, tf in self.inverted_index[term]:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len)))
                term_score = idf_val * (tf * (self.k1 + 1.0)) / max(1e-6, denom)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        if not scores:
            return []

        # Sort top-k
        top_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.idx_to_doc_id[idx], score) for idx, score in top_indices]
