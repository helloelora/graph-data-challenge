# Link Prediction - Results Summary

## Overview

| Strategy | Code | Submission CSV | Local CV AUC | Local CV Acc@0.5 | Kaggle Score |
|----------|------|----------------|-------------|------------------|--------------|
| Logistic Regression | `logistic_regression.py` | `submission_lr_baseline.csv` | 0.5662 | 0.6180 | — |
| HistGradientBoosting | `hist_gradient_boosting.py` | `submission_hgb_intermediate.csv` | 0.5648 | 0.6288 | **0.84927** |
| XGB+HGB+LR Ensemble | `xgb_hgb_lr_ensemble.py` | `submission_best.csv` | 0.5371 | 0.5685 | 0.76833 |

> Local CV metrics are strict out-of-fold scores on `train.txt` (graph rebuilt per fold, no leakage).
> Kaggle score is the public leaderboard AUC.

---

## Strategy details

### 1. Logistic Regression (`logistic_regression.py`)

**Model:** Logistic Regression (with StandardScaler), C=0.7

**Features (16 compact features):**
- **Graph heuristics:** degree (u, v, diff, sum, log), common neighbors, Jaccard (graph), Adamic-Adar, resource allocation, preferential attachment, same component, both isolated
- **Text overlap:** raw keyword dot product, cosine similarity, keyword Jaccard

**Approach:** Simple, interpretable baseline. Single linear model on handcrafted structural + text features.

---

### 2. HistGradientBoosting (`hist_gradient_boosting.py`)

**Model:** HistGradientBoostingClassifier (lr=0.07, max_depth=4, max_iter=250, min_samples_leaf=30)

**Features:** Same 16 compact features as Logistic Regression (imports `build_features` from `logistic_regression.py`)

**Approach:** Replaces the linear model with a gradient boosting tree to capture non-linear feature interactions, while keeping the same simple feature set. Best Kaggle score.

---

### 3. XGB+HGB+LR Ensemble (`xgb_hgb_lr_ensemble.py`)

**Models (blended with optimized weights):**
- XGBoost (n_estimators=600, lr=0.04, max_depth=6)
- HistGradientBoostingClassifier (lr=0.05, max_depth=6, max_iter=500)
- Logistic Regression (C=2.0, with StandardScaler)

**Features (29 scalar + 256 embedding dimensions):**
- **Graph heuristics:** same as baseline + Sorensen, hub promoted, hub depressed
- **Text similarities:** raw dot/cosine/L1/L2, TF-IDF cosine, keyword Jaccard, shared keyword ratio
- **Text embeddings:** TF-IDF + SVD (64d) -> hadamard product + absolute difference
- **Graph embeddings:** Adjacency SVD (64d) -> hadamard product + absolute difference

**Approach:** Rich feature engineering combining graph topology, text content, and learned embeddings. Three diverse models blended via Dirichlet weight optimization on OOF predictions (5000 random trials).
