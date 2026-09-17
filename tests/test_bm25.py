"""
Unit tests for BM25 inverted index, query construction, and candidate scoring.
"""
from src.models.bm25 import BM25Index
from src.models.query_builder import HistoryQueryBuilder
from src.data.feature_store import FeatureStore
from src.data.schema import ArticleRecord, UserHistoryRecord

def test_bm25_indexing_and_retrieval():
    corpus = {
        "art1": "Premier league football match result and goals",
        "art2": "Stock market economy inflation rate and finance",
        "art3": "Premier league football championship standings"
    }
    index = BM25Index(k1=1.5, b=0.75, language="english")
    index.fit(corpus)

    assert index.num_docs == 3
    assert "football" in index.inverted_index
    assert "stock" in index.inverted_index

    # Query football
    query_tokens = ["football", "premier"]
    scores = index.score_candidates(query_tokens, ["art1", "art2", "art3"])
    
    assert len(scores) == 3
    assert scores[0] > scores[1]  # art1 > art2
    assert scores[2] > scores[1]  # art3 > art2
    assert scores[1] == 0.0       # art2 has no overlap

def test_query_builder():
    fs = FeatureStore(name="test")
    fs.add_articles({
        "A1": ArticleRecord(article_id="A1", title="Danish Football", abstract="Copenhagen wins match"),
        "A2": ArticleRecord(article_id="A2", title="Weather Report", abstract="Sunny weekend ahead")
    })
    fs.add_user_histories({
        "U1": UserHistoryRecord(user_id="U1", history_article_ids=["A1"])
    })

    qb = HistoryQueryBuilder(feature_store=fs, language="english", max_articles=5)
    tokens = qb.build_query_tokens("U1")
    assert "football" in tokens
    assert "copenhagen" in tokens
