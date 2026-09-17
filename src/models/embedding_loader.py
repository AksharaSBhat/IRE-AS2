"""
Loads, computes, and standardizes dense article embeddings for EB-NeRD and MIND.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src.config import EBNERD_WORD2VEC_DIR, CACHE_DIR
from src.data.feature_store import FeatureStore

def normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    """L2-normalizes an array of vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms

class EmbeddingLoader:
    """Loads or computes dense semantic embeddings for news articles."""

    @staticmethod
    def load_ebnerd_embeddings(
        w2v_dir: Path = EBNERD_WORD2VEC_DIR,
        feature_store: Optional[FeatureStore] = None
    ) -> Dict[str, np.ndarray]:
        """
        Loads pre-computed 300-d Word2Vec document vectors for EB-NeRD.
        """
        w2v_path = Path(w2v_dir) / "document_vector.parquet"
        if not w2v_path.exists():
            raise FileNotFoundError(f"EB-NeRD Word2Vec parquet not found at: {w2v_path}")

        print(f"Loading EB-NeRD document vectors from: {w2v_path}...")
        df = pl.read_parquet(w2v_path)
        
        aids = df["article_id"].to_list()
        vectors_raw = df["document_vector"].to_list()

        embeddings: Dict[str, np.ndarray] = {}
        for aid, vec in zip(aids, vectors_raw):
            if vec is not None:
                v = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
                embeddings[str(aid)] = v

        print(f" Loaded {len(embeddings):,} EB-NeRD article embeddings (dim={len(next(iter(embeddings.values())))}).")
        return embeddings

    @staticmethod
    def load_or_compute_mind_embeddings(
        feature_store: FeatureStore,
        dim: int = 300,
        cache_name: str = "mind_embeddings.npy"
    ) -> Dict[str, np.ndarray]:
        """
        Computes or loads dense semantic embeddings for MIND articles using high-quality
        TF-IDF + TruncatedSVD / Sentence-Transformers representation, normalized to unit sphere.
        """
        cache_path = CACHE_DIR / cache_name
        cache_keys_path = CACHE_DIR / f"{cache_name}.keys.txt"

        if cache_path.exists() and cache_keys_path.exists():
            print(f"Loading cached MIND embeddings from: {cache_path}...")
            matrix = np.load(cache_path)
            with open(cache_keys_path, "r") as f:
                keys = [line.strip() for line in f]
            return {k: matrix[i] for i, k in enumerate(keys)}

        print("Computing dense semantic embeddings for MIND articles...")
        article_ids = list(feature_store.articles.keys())
        texts = [feature_store.get_article_text(aid) for aid in article_ids]

        # Try SentenceTransformers if available, else fast LSA (TF-IDF + SVD)
        matrix = None
        try:
            from sentence_transformers import SentenceTransformer
            print(" Using SentenceTransformer('all-MiniLM-L6-v2')...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            matrix = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
            matrix = np.array(matrix, dtype=np.float32)
        except Exception as e:
            print(f" SentenceTransformer not ready ({e}). Falling back to fast TruncatedSVD LSA (dim={dim})...")
            tfidf = TfidfVectorizer(max_features=50000, stop_words="english")
            tfidf_mat = tfidf.fit_transform(texts)
            actual_dim = min(dim, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
            svd = TruncatedSVD(n_components=actual_dim, random_state=42)
            matrix = svd.fit_transform(tfidf_mat)
            matrix = normalize_vectors(np.array(matrix, dtype=np.float32))

        # Save cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
        with open(cache_keys_path, "w") as f:
            f.write("\n".join(article_ids))

        print(f" Computed and cached {len(article_ids):,} MIND embeddings (dim={matrix.shape[1]}).")
        return {aid: matrix[i] for i, aid in enumerate(article_ids)}

load_ebnerd_embeddings = EmbeddingLoader.load_ebnerd_embeddings
load_or_compute_mind_embeddings = EmbeddingLoader.load_or_compute_mind_embeddings
compute_or_load_mind_embeddings = EmbeddingLoader.load_or_compute_mind_embeddings
