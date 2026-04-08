"""
v21 — controlled experiments built strictly on v19 features (proven 0.86766).
Each submission isolates ONE variable so we can identify what actually helps.

Reuses build_features from best_solution.py (v19 — 41 features) UNCHANGED.

Submissions generated:
  v21_baseline       — v19 reproduced (HGB+CatBoost rank, 30 seeds) — sanity check
  v21_50seeds        — v19 with 50 seeds (vs 30)
  v21_3way_equal     — v19 + LightGBM, equal 3-way rank blend (50 seeds each)
  v21_3way_2hgbcat   — v19 + LGB, 40/40/20 (HGB/Cat/LGB) — minor LGB perturbation
  v21_geomean        — v19 with geometric mean of HGB+CatBoost ranks
  v21_st97           — v19 with stricter self-training (threshold 0.97)
  v21_st_blend       — Average of pseudo-edge sets at thresholds 0.93, 0.95, 0.97
  v21_lgb_only       — pure LightGBM with v19 features, 50 seeds
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import lightgbm as lgb

# Import v19 helpers UNCHANGED
from best_solution import (
    load_data,
    build_graph,
    build_sparse_adj,
    build_features,
)

SEED = 42
EPS = 1e-12


# Same hyperparams as v19
HGB_PARAMS = dict(
    learning_rate=0.05, max_depth=3, max_iter=400,
    min_samples_leaf=40, l2_regularization=0.1,
)

CAT_PARAMS = dict(
    iterations=300, learning_rate=0.05, depth=3,
    l2_leaf_reg=10, verbose=0,
)

# LightGBM tuned to be similar in capacity to v19's HGB
LGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, max_depth=4,
    num_leaves=15, min_child_samples=40, reg_lambda=1.0,
    subsample=0.9, colsample_bytree=0.9, verbose=-1,
)


def predict_hgb(X_tr, y_tr, X_te, n_seeds, base_seed=SEED):
    pred = np.zeros(len(X_te), dtype=np.float64)
    for s in range(n_seeds):
        m = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=base_seed + s * 31)
        m.fit(X_tr, y_tr)
        pred += m.predict_proba(X_te)[:, 1]
    return pred / n_seeds


def predict_cat(X_tr, y_tr, X_te, n_seeds, base_seed=SEED):
    pred = np.zeros(len(X_te), dtype=np.float64)
    for s in range(n_seeds):
        m = CatBoostClassifier(**CAT_PARAMS, random_seed=base_seed + s * 31)
        m.fit(X_tr, y_tr)
        pred += m.predict_proba(X_te)[:, 1]
    return pred / n_seeds


def predict_lgb(X_tr, y_tr, X_te, n_seeds, base_seed=SEED):
    pred = np.zeros(len(X_te), dtype=np.float64)
    for s in range(n_seeds):
        m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=base_seed + s * 31)
        m.fit(X_tr, y_tr)
        pred += m.predict_proba(X_te)[:, 1]
    return pred / n_seeds


def normalize_rank(arr):
    r = rankdata(arr)
    return (r - r.min()) / (r.max() - r.min() + EPS)


def save_sub(name, pred):
    pred = normalize_rank(pred).astype(np.float32) if pred.max() > 1.0 or pred.min() < 0 else np.clip(pred, 0, 1).astype(np.float32)
    pd.DataFrame({"ID": np.arange(len(pred)), "Predicted": pred}).to_csv(name, index=False)
    print(f"  saved {name}")


def build_v19_features(train_pairs, y_train, test_pairs, node_features, n_nodes,
                       node_tfidf, total_count, train_count, test_count,
                       st_threshold=0.95):
    """Build v19 features with self-training at given threshold."""
    # Initial round
    adj0, deg0, comp0 = build_graph(train_pairs, y_train, n_nodes)
    A0 = build_sparse_adj(adj0, n_nodes)
    d_inv0 = np.zeros(n_nodes, dtype=np.float32)
    m0 = deg0 > 0
    d_inv0[m0] = 1.0 / deg0[m0]
    ntfidf0 = sparse.diags(d_inv0) @ A0 @ node_tfidf

    X_tr0 = build_features(
        train_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count,
        A0, ntfidf0, y=y_train, remove_pos=True,
    )
    X_te0 = build_features(
        test_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count,
        A0, ntfidf0,
    )

    # Initial 5-seed HGB for self-training (matches v19)
    pred_init = predict_hgb(X_tr0, y_train, X_te0, n_seeds=5)
    extra_edges = test_pairs[pred_init >= st_threshold]
    print(f"  self-train (t={st_threshold}): +{len(extra_edges)} edges")

    # Rebuild with pseudo-edges
    adjacency, degree, comp = build_graph(train_pairs, y_train, n_nodes, extra_edges)
    adj_matrix = build_sparse_adj(adjacency, n_nodes)
    d_inv = np.zeros(n_nodes, dtype=np.float32)
    mask = degree > 0
    d_inv[mask] = 1.0 / degree[mask]
    neighbor_tfidf = sparse.diags(d_inv) @ adj_matrix @ node_tfidf

    X_train = build_features(
        train_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
        y=y_train, remove_pos=True,
    )
    X_test = build_features(
        test_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
    )
    return X_train, X_test, len(extra_edges)


def main():
    t0 = time.time()
    np.random.seed(SEED)

    train_pairs, y_train, test_pairs, node_features = load_data(
        Path("train.txt"), Path("test.txt"), Path("node_information.csv")
    )
    n_nodes = node_features.shape[0]
    nf_sparse = sparse.csr_matrix(node_features)
    node_tfidf = TfidfTransformer(
        norm="l2", use_idf=True, smooth_idf=True
    ).fit_transform(nf_sparse)

    # Transductive counts
    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in train_pairs:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in test_pairs:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    # =====================================================
    # Pipeline 1: standard v19 self-training (threshold 0.95)
    # =====================================================
    print("\n[features] standard v19 (st=0.95)")
    X_train, X_test, _ = build_v19_features(
        train_pairs, y_train, test_pairs, node_features, n_nodes,
        node_tfidf, total_count, train_count, test_count,
        st_threshold=0.95,
    )
    print(f"  features: {X_train.shape[1]}")

    # --- Train all 3 models with 50 seeds ---
    print("\n[train] HGB 50 seeds")
    pred_h50 = predict_hgb(X_train, y_train, X_test, n_seeds=50)
    print("[train] CatBoost 50 seeds")
    pred_c50 = predict_cat(X_train, y_train, X_test, n_seeds=50)
    print("[train] LightGBM 50 seeds")
    pred_l50 = predict_lgb(X_train, y_train, X_test, n_seeds=50)

    # Also 30-seed versions for v19 reproduction baseline
    print("[train] HGB 30 seeds (v19 baseline)")
    pred_h30 = predict_hgb(X_train, y_train, X_test, n_seeds=30)
    print("[train] CatBoost 30 seeds (v19 baseline)")
    pred_c30 = predict_cat(X_train, y_train, X_test, n_seeds=30)

    # Ranks
    rh50 = normalize_rank(pred_h50)
    rc50 = normalize_rank(pred_c50)
    rl50 = normalize_rank(pred_l50)
    rh30 = normalize_rank(pred_h30)
    rc30 = normalize_rank(pred_c30)

    # =====================================================
    # SUBMISSION 1: v21_baseline — sanity check (= v19)
    # =====================================================
    save_sub("submission_v21_baseline.csv", 0.5 * rh30 + 0.5 * rc30)

    # =====================================================
    # SUBMISSION 2: v21_50seeds — v19 + more seeds (only diff: 30 -> 50)
    # =====================================================
    save_sub("submission_v21_50seeds.csv", 0.5 * rh50 + 0.5 * rc50)

    # =====================================================
    # SUBMISSION 3: v21_3way_equal — v19 + LGB, equal blend
    # =====================================================
    save_sub("submission_v21_3way_equal.csv", (rh50 + rc50 + rl50) / 3.0)

    # =====================================================
    # SUBMISSION 4: v21_3way_2hgbcat — LGB as small perturbation (40/40/20)
    # =====================================================
    save_sub("submission_v21_3way_2hgbcat.csv", 0.4 * rh50 + 0.4 * rc50 + 0.2 * rl50)

    # =====================================================
    # SUBMISSION 5: v21_geomean — geometric mean of HGB + CatBoost ranks
    # =====================================================
    geo = np.sqrt(rh50 * rc50)
    save_sub("submission_v21_geomean.csv", geo)

    # =====================================================
    # SUBMISSION 6: v21_lgb_only — pure LightGBM
    # =====================================================
    save_sub("submission_v21_lgb_only.csv", pred_l50)

    # =====================================================
    # Pipeline 2: stricter self-training (threshold 0.97)
    # =====================================================
    print("\n[features] stricter self-training (st=0.97)")
    X_train_st97, X_test_st97, n_st97 = build_v19_features(
        train_pairs, y_train, test_pairs, node_features, n_nodes,
        node_tfidf, total_count, train_count, test_count,
        st_threshold=0.97,
    )

    print("[train] HGB+Cat (30 seeds) on st97 features")
    pred_h_st97 = predict_hgb(X_train_st97, y_train, X_test_st97, n_seeds=30)
    pred_c_st97 = predict_cat(X_train_st97, y_train, X_test_st97, n_seeds=30)
    rh_st97 = normalize_rank(pred_h_st97)
    rc_st97 = normalize_rank(pred_c_st97)

    # SUBMISSION 7: v21_st97
    save_sub("submission_v21_st97.csv", 0.5 * rh_st97 + 0.5 * rc_st97)

    # =====================================================
    # SUBMISSION 8: v21_dual_st — average of st=0.95 and st=0.97 predictions
    # Different graphs => different model perspectives
    # =====================================================
    save_sub(
        "submission_v21_dual_st.csv",
        0.25 * rh50 + 0.25 * rc50 + 0.25 * rh_st97 + 0.25 * rc_st97,
    )

    # =====================================================
    # SUBMISSION 9: v21_mega — 3way + dual_st
    # =====================================================
    save_sub(
        "submission_v21_mega.csv",
        0.20 * rh50 + 0.20 * rc50 + 0.20 * rl50 +
        0.20 * rh_st97 + 0.20 * rc_st97,
    )

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
