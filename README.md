<div align="center">

# Link Prediction in an Actor Co-occurrence Network

![Rank](https://img.shields.io/badge/Rank-1st%20Place-FFB800?style=for-the-badge&logo=trophy&logoColor=white)
![AUC](https://img.shields.io/badge/AUC-0.867-26D0CE?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Predicting missing edges in a sparse actor co-occurrence graph using hand-crafted features and gradient boosting.**

</div>

---

## The Challenge

Given a partially observed graph where **nodes = actors** and **edges = co-occurrence on Wikipedia pages**, predict which edges were randomly deleted. Each node has **932 binary keyword features**. Evaluation metric: **AUC-ROC**.

The catch: the graph is extremely sparse (mean degree ~2.9) and **91.5% of test pairs share zero common neighbors**, making classical graph heuristics alone insufficient.

## Strategy Overview

```mermaid
flowchart LR
    A[Raw Data] --> B[Feature Engineering]
    B --> C[Self-Training]
    C --> D[Ensemble]
    D --> E[0.867 AUC]

    style A fill:#1A2980,color:#fff,stroke:none
    style B fill:#7C5CFC,color:#fff,stroke:none
    style C fill:#FF922B,color:#fff,stroke:none
    style D fill:#51CF66,color:#fff,stroke:none
    style E fill:#FFB800,color:#fff,stroke:none
```

The core insight behind our approach: **with only 10K training samples, scalar hand-crafted features massively outperform learned representations**. Every embedding-based method we tried (SVD, Node2Vec, GCN) overfit and hurt performance. Instead, we engineered 36 carefully chosen scalar features across three complementary families, each targeting a different aspect of what makes two actors likely to co-occur.

### 1. Feature Engineering (36 features)

We build features that answer three distinct questions about each node pair:

| | Family | # | Question it answers |
|---|---|---|---|
| **Graph** | Topology | 14 | _Are these nodes close in the graph?_ |
| **Text** | Similarity | 9 | _Do these actors share similar keywords?_ |
| **Transductive** | Meta-signals | 6 | _Does the test set structure hint at a link?_ |
| **Hybrid** | Neighborhood text | 4 | _Does this actor look like the other's friends?_ |
| **Derived** | Katz, hubs, interactions | 3 | _What do higher-order paths tell us?_ |

**Graph topology** — We compute classical link prediction indices from the literature: Common Neighbors, Jaccard, Adamic-Adar, Resource Allocation, Sorensen, and Preferential Attachment. We also extract degree features (raw, sum, difference, log-transformed, min/max) and component membership flags. For pairs with CN=0 (the vast majority), we go beyond direct neighbors with **paths of length 3** ($A^3[u,v]$ via sparse matrix multiplication), which captures indirect connectivity that CN misses entirely.

> **Leakage correction**: For positive training pairs, the direct edge $u$-$v$ inflates $A^3[u,v]$ by $\deg(u) + \deg(v) - 1$ spurious walks. We subtract this analytically to prevent the model from simply memorizing which edges exist.

**Text similarity** — From the 932 binary keyword columns, we extract raw dot product, cosine similarity, TF-IDF cosine (downweights common keywords like "actor"), TF-IDF L2 distance, keyword Jaccard, and asymmetric overlap (captures whether one actor's keywords are a subset of the other's).

**Neighborhood text** — For each node, we average its neighbors' TF-IDF vectors, then ask: _"Does node $v$'s text profile match what node $u$'s friends look like?"_ This is essentially a manual 1-hop GCN aggregation expressed as scalar features — capturing the same signal without the overfitting risk of learned embeddings. We also correct for leakage on positive training pairs by excluding $v$'s contribution from $u$'s neighborhood average.

**Transductive features** — The biggest single improvement (+0.011 AUC). We count how many times each node appears across all pairs (train + test). Nodes appearing frequently in the test set likely had more edges deleted. We separate train-only and test-only counts and add interaction terms for richer signal.

### 2. Self-Training Graph Augmentation

The sparse graph limits what structural features can capture. To enrich it:

1. Train an initial model on base features
2. Predict test pairs — add the ~320 with probability >= 0.95 as pseudo-edges
3. Rebuild the entire graph and recompute all graph-dependent features

This single conservative round improved downstream features (CN, paths3, neighborhood text all benefit from a denser graph). We tried multiple rounds and lower thresholds — both added noise and hurt Kaggle score.

### 3. Ensemble & Seed Averaging

Final predictions combine **HistGradientBoosting** and **CatBoost** (15 random seeds each) via **rank averaging**. Since AUC is rank-based, converting each model's predictions to ranks before blending eliminates calibration mismatches. The two boosting algorithms make different errors, and their combination is more stable than either alone.

## Results

| Approach | Features | AUC |
|---|---|---|
| GCN (end-to-end) | adjacency + keywords | 0.670 |
| Node2Vec embeddings | 87 | 0.770 |
| SVD/NMF embeddings | 151 | 0.790 |
| Graph heuristics only | 14 | 0.831 |
| + Text similarity | 23 | 0.850 |
| + Self-training | 23 | 0.851 |
| + Transductive counts | 27 | 0.861 |
| **+ Paths3, Neighborhood text** | **36** | **0.867** |

> Every embedding-based method made things worse. On small datasets, feature discipline beats model complexity.

## What Didn't Work

| Approach | AUC | Why it failed |
|---|---|---|
| GCN (2-layer) | 0.670 | Massive overfitting on 10K samples |
| Node2Vec (64-d) | 0.770 | Too many dimensions for small data |
| SVD/NMF embeddings | 0.790 | Curse of dimensionality |
| Multi-round self-training | < 0.867 | Lower thresholds add noisy edges |
| Personalized PageRank | ~1.0 OOF | Leakage too complex to correct analytically |
| Stacking meta-learner | < 0.861 | Further overfits the small dataset |

## Repo Structure

```
├── best_solution.py          # Final solution (v19 — 36 features + ensemble)
├── logistic_regression.py    # Baseline: LR with 16 features
├── hist_gradient_boosting.py # HGB baseline
├── xgb_hgb_lr_ensemble.py   # Early ensemble experiment
├── train.txt                 # 10,496 labeled pairs (50% positive)
├── test.txt                  # 3,498 test pairs
├── node_information.csv      # 3,597 nodes x 932 keyword features
├── report.tex                # 3-page LaTeX report
└── submission_v16b_blend.csv # Best submission (0.867)
```

## Reproduce

```bash
pip install numpy pandas scikit-learn scipy catboost
python best_solution.py
```

Generates submission CSVs ready for Kaggle upload. Runs in ~40 seconds on a laptop.

## Key References

- Liben-Nowell & Kleinberg — [The Link Prediction Problem for Social Networks](https://doi.org/10.1002/asi.20591) (JASIST 2007)
- Adamic & Adar — [Friends and Neighbors on the Web](https://doi.org/10.1016/j.socnet.2004.11.003) (Social Networks 2003)
- Katz — [A New Status Index Derived from Sociometric Analysis](https://doi.org/10.1007/BF02289026) (Psychometrika 1953)
- Ke et al. — [LightGBM](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) (NeurIPS 2017)
- Prokhorenkova et al. — [CatBoost](https://papers.nips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) (NeurIPS 2018)
