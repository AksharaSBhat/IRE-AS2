# Industrial Retrieve-then-Rank News Recommendation
### IRE (CS4.406) --- Assignment 2: Learning from Click-Logs

This repository implements a production-grade **Two-Stage Retrieve-then-Rank News Recommendation System** evaluated on both the **EB-NeRD** (Ekstra Bladet RecSys'24 Challenge) and **MIND** (Microsoft News Dataset) benchmarks.

---

## 1. Quick Start: One-Command Reproduction

### Run All Unit Tests (24 Passing Tests)
```bash
pytest tests/ -v
```

### Run End-to-End Re-Ranking Experiments (Q1, Q2, Q3, Q5)
```bash
# Run on EB-NeRD (Extracts features, trains Baseline & Improved LightGBM, runs 1,000-round paired bootstrap, ablations, slicing)
python run_experiments.py --dataset ebnerd

# Run on MIND
python run_experiments.py --dataset mind
```

### Run Production Serving & Scale Profiler (Q4)
```bash
python run_serving_benchmark.py
```

### Compile Academic Design Note PDF (Q6)
```bash
cd report && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
# Generates report/main.pdf (clean 6-page academic paper adhering to 11pt, 1-inch margins)
```

---

## 2. Key Architecture & Methodology

```
                      +---------------------------------------+
                      |   Full News Catalog (~125k Articles)  |
                      +-------------------+-------------------+
                                          |
                                          v  [Stage 1: Candidate Generation]
                                      FAISS ANN (300-d) + BM25Okapi
                                          |
                                          v
                      +---------------------------------------+
                      |       Top-K Candidates (K = 100)      |
                      +-------------------+-------------------+
                                          |
                                          v  [Point-in-Time Causal Feature Store]
                                18 Behavioral Features (tau_i < t_imp)
                                          |
                                          v  [Stage 2: GBDT Scoring]
                                 LightGBM Re-Ranker Ensemble
                                          |
                                          v
                      +---------------------------------------+
                      |        Top-N Ranked News Feed (N = 10)|
                      +---------------------------------------+
```

### Engineered Feature Taxonomy (18 Features - Q1)
1. **Click-History Features**:
   - `user_click_count`: $|\mathcal{H}_u(t_{\text{imp}})|$ historical interaction depth.
   - `user_avg_dwell`: $\frac{1}{|\mathcal{H}_u|} \sum \min(d_i, 300\text{s})$ reading engagement depth (300s cap).
   - `user_category_entropy`: $H(u) = -\sum_{c} p(c) \log_2 p(c)$ topical specialization.
2. **Session and Context Features**:
   - `cand_position`: Integer visual rank position ($0 \dots M-1$) modeling feed position bias.
   - `cand_inview_count`: Candidate set cardinality $M = |\mathcal{C}_u|$.
   - `hour_sin`, `hour_cos`: Cyclical diurnal encodings capturing circadian reading habits.
   - `day_of_week`: Weekday vs. weekend preferences ($0\text{--}6$).
3. **Article Content and Freshness Features**:
   - `cand_freshness_hours`: $\log(1 + \max(0, t_{\text{imp}} - t_{\text{pub}}))$.
   - `cand_freshness_decay`: Exponential decay $\exp(-\Delta t / 24.0)$ with a 24-hour half-life.
   - `cand_popularity_clicks`: Prior historical engagement count $C(a)$.
   - `cand_title_length`, `cand_abstract_length`: Editorial document formats.
4. **Cross-Interaction Features (User $\times$ Candidate)**:
   - `sim_user_cand_recency`: Cosine similarity with exponential recency profile $\mathbf{u}_{\text{rec}} \propto \sum_i e^{-\lambda(t_{\text{imp}}-\tau_i)} \hat{\mathbf{v}}_{a_i}$.
   - `sim_user_cand_dwell`: Cosine similarity with dwell-scaled profile $w_i = e^{-\lambda \Delta \tau_i}(1 + \log(1 + d_i))$ downweighting clickbait bounces.
   - `category_match`: Binary flag $\mathbf{1}[\text{cat}(a) = \text{top\_cat}(u)]$.
   - `user_category_affinity`: Empirical category preference ratio $\frac{|\{i \mid \text{cat}(a_i) = \text{cat}(a)\}|}{|\mathcal{H}_u|}$.
   - `bm25_score`: Lexical retrieval relevance score from Stage 1.

---

## 3. Experimental Findings & Quantitative Results

### Quantitative Progression: Stage 1 vs. Stage 2 (Q2)

| Dataset & Pipeline Stage | Macro AUC | MRR | nDCG@5 | nDCG@10 |
|:---|:---:|:---:|:---:|:---:|
| **EB-NeRD (Danish News)** | | | | |
| Stage 1: Candidate Generation (ANN) | 0.4700 | 0.3066 | 0.3350 | 0.4214 |
| Stage 2: GBDT Re-Ranker (Baseline) | 0.5144 | 0.3211 | 0.3604 | 0.4440 |
| **Stage 2: GBDT Re-Ranker (Improved)** | **0.6847** | **0.4578** | **0.5178** | **0.5709** |
| *Relative Gain (Improved vs. Stage 1)* | *+45.68%* | *+49.31%* | *+54.57%* | *+35.48%* |
| **MIND (English News)** | | | | |
| Stage 1: Candidate Generation (ANN) | 0.5669 | 0.3034 | 0.2835 | 0.3436 |
| Stage 2: GBDT Re-Ranker (Baseline) | 0.5644 | 0.2997 | 0.2774 | 0.3450 |
| **Stage 2: GBDT Re-Ranker (Improved)** | **0.6233** | **0.3405** | **0.3160** | **0.3818** |
| *Relative Gain (Improved vs. Stage 1)* | *+9.95%* | *+12.23%* | *+11.46%* | *+11.12%* |

### Statistical Significance: Paired Bootstrap 95% CIs ($B = 1,000$ Rounds - Q3)

| Metric | Baseline | Improved | $\Delta$ (Observed) | 95% Bootstrap CI | Empirical $p$-value | Significant? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EB-NeRD** AUC | 0.5144 | 0.6847 | +0.1702 | **[+0.1453, +0.1965]** | $< 0.0001$ | **YES ($CI > 0$)** |
| **EB-NeRD** MRR | 0.3211 | 0.4578 | +0.1367 | **[+0.1125, +0.1634]** | $< 0.0001$ | **YES ($CI > 0$)** |
| **EB-NeRD** nDCG@10 | 0.4440 | 0.5709 | +0.1269 | **[+0.1064, +0.1495]** | $< 0.0001$ | **YES ($CI > 0$)** |
| **MIND** AUC | 0.5644 | 0.6233 | +0.0589 | **[+0.0421, +0.0742]** | $< 0.0001$ | **YES ($CI > 0$)** |
| **MIND** MRR | 0.2997 | 0.3405 | +0.0408 | **[+0.0258, +0.0557]** | $< 0.0001$ | **YES ($CI > 0$)** |
| **MIND** nDCG@10 | 0.3450 | 0.3818 | +0.0369 | **[+0.0240, +0.0487]** | $< 0.0001$ | **YES ($CI > 0$)** |

*All 95% confidence intervals strictly exclude zero ($p < 0.0001$), confirming statistically significant improvements.*

### Systematic Ablation Analysis (EB-NeRD)
- **Full Improved Model**: AUC = **0.6847**
- **w/o Freshness Decay**: AUC = 0.5186 ($\Delta = \mathbf{-0.1660}$) $\to$ Freshness is the single dominant driver in news recommendation.
- **w/o Category Affinity**: AUC = 0.6411 ($\Delta = \mathbf{-0.0436}$)
- **w/o Dwell Weighting**: AUC = 0.6592 ($\Delta = \mathbf{-0.0255}$) $\to$ Dwell weighting filters accidental clickbait bounces.
- **w/o Semantic Cross-Sim**: AUC = 0.6725 ($\Delta = \mathbf{-0.0122}$)

---

## 4. Production Serving & Scale Analysis (Q4)

- **Serving In-Memory Footprint**:
  - Article Metadata: 20.25 MB (20,738 records)
  - User Histories: 9.19 MB (18,827 users)
  - Vector Index (HNSW 300-d): 215.51 MB (125,541 vectors)
  - LightGBM Trees: 0.03 MB (120 decision trees)
  - **Total Serving RAM: 244.98 MB**
- **End-to-End Latency Benchmark (500 Real-World Impressions)**:
  - Mean: **1.82 ms**
  - $p_{50}$ (Median): **1.73 ms**
  - $p_{95}$: **2.54 ms**
  - $p_{99}$: **3.60 ms** (Target SLA $<100\text{ ms}$: **PASS**, $3.60\text{ ms} \ll 100\text{ ms}$)
- **Throughput & Cloud Cost Economics**:
  - Single-Thread Capacity: $550.5\text{ QPS}$
  - 8-vCPU AWS `c6i.2xlarge` Node Capacity ($70\%$ load): $\mathbf{3,082.5\text{ QPS}}$
  - Cost per 1,000 Queries: $\mathbf{\$0.000031\text{ USD}}$ ($10\text{M requests} \approx \$0.31\text{ USD}$)
- **$10\times$ Scaling Breakdown & Mitigations**:
  - *User History Store*: Transitions to distributed Redis cluster sharded by `user_id` with 30-day TTL.
  - *Vector Index*: Flat index degrades $>15\text{ ms}$; mitigated via sharded HNSW + PQ8 compression (38 bytes/vec, sub-2ms).
  - *Feature Freshness*: Kafka + Apache Flink stateful stream aggregation $\to$ sub-50ms rolling popularity counters.

---

## 5. Anti-Gaming & Strict Temporal Boundary Enforcement (Q9)

- **Causal Isolation Principle**: In news feeds, user histories cannot access future interactions ($\tau_i \ge t_{\text{imp}}$).
- **Automated Guard**: `filter_history_before_time` rigorously prunes all events with $\tau_i \ge t_{\text{imp}}$.
- **Metrics With vs. Without Features Unavailable at Serving Time**:
  - *Serving-Compliant (Strict Causal)*: AUC = **0.6847** (EB-NeRD), **0.6233** (MIND).
  - *Serving-Unavailable Leaked Signals (Candidate Dwell Leaked)*: AUC = **0.9124** (+33.3% artificial inflation). Candidate dwell is unknown at serving time before presentation; exposing it creates an unviable, leaking model.
- **Unit Test Suite**: `tests/test_anti_gaming.py` contains 3 automated tests verifying zero future-click leakage and asserting that contaminated histories trigger immediate assertions.

---

## 6. Repository Structure

```
as2/
├── README.md                          # Reproduction guide and system documentation
├── run_experiments.py                 # End-to-end training, bootstrap CIs, ablations, slicing
├── run_serving_benchmark.py           # Memory profiling, 500-sample latency, QPS & cost economics
├── run_pipeline.py                    # Modular retrieve-then-rank pipeline runner
├── run_submission.py                  # Full-scale Codabench prediction zip generator
├── conftest.py                        # Pytest path resolution
├── src/
│   ├── config.py                      # System hyperparameters, SLAs, and paths
│   ├── data/
│   │   ├── schema.py                  # Unified Pydantic-style data classes
│   │   ├── feature_store.py           # In-memory point-in-time feature store
│   │   ├── ebnerd_loader.py           # Polars/PyArrow streaming EB-NeRD loader
│   │   └── mind_loader.py             # Polars/PyArrow streaming MIND loader
│   ├── features/
│   │   └── feature_extractor.py       # 18 behavioral feature extractor with causal filter
│   ├── models/
│   │   ├── bm25.py                    # BM25Okapi lexical candidate retriever
│   │   ├── semantic.py                # FAISS dense semantic ANN candidate retriever
│   │   ├── embedding_loader.py        # 300-d Word2Vec/BERT embedding loader
│   │   └── reranker.py                # LightGBM GBDT Re-Ranker and ablation variants
│   ├── serving/
│   │   └── profiler.py                # Memory profiler, latency timer, cost calculator
│   ├── evaluation/
│   │   ├── metrics.py                 # Macro AUC, MRR, nDCG@K, ILD, Novelty, Coverage
│   │   ├── bootstrap.py               # Standard bootstrap CI
│   │   ├── paired_bootstrap.py        # Vectorized 1,000-round paired bootstrap CI
│   │   └── slicing.py                 # Warmth and popularity slice partitioners
│   └── utils/
│       └── anti_gaming.py             # Temporal boundary checker and assertion guards
├── tests/                             # 24 Unit tests covering all subsystems
│   ├── test_anti_gaming.py
│   ├── test_bm25.py
│   ├── test_features.py
│   ├── test_feature_store.py
│   ├── test_metrics.py
│   ├── test_paired_bootstrap.py
│   ├── test_reranker.py
│   ├── test_schemas.py
│   ├── test_semantic.py
│   ├── test_serving.py
│   └── test_slicing_bootstrap.py
├── report/
│   ├── main.tex                       # Academic Design Note LaTeX source (target 6-page)
│   └── main.pdf                       # Compiled 6-page PDF design note
├── results_ebnerd.json                # Experimental metrics for EB-NeRD
├── results_mind.json                  # Experimental metrics for MIND
├── results_serving.json               # Serving and scale benchmark metrics
└── submissions/                       # Codabench submission archives
    ├── ebnerd_predictions.zip         # 13,536,710 prediction lines
    └── mind_prediction.zip            # 2,370,727 prediction lines (Finished)
```
