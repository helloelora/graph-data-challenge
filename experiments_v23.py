"""
v23 — Symmetric augmentation. Link prediction is undirected, but v19 features
are not all symmetric in (u,v) (e.g., deg_u, deg_v, tc_u, tc_v, neigh_text_uv).
The model learns spurious order patterns. Fix:

  TTA only:           predict on (u,v) and (v,u), average  -> SAFE
  Train aug only:     train on doubled data, predict normally
  TTA + train aug:    both at training and test time
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from catboost import CatBoostClassifier

from best_solution import (
    load_data,
    build_graph,
    build_sparse_adj,
    build_features,
)

SEED = 42
EPS = 1e-12

HGB_PARAMS = dict(
    learning_rate=0.05, max_depth=3, max_iter=400,
    min_samples_leaf=40, l2_regularization=0.1,
)

CAT_PARAMS = dict(
    iterations=300, learning_rate=0.05, depth=3,
    l2_leaf_reg=10, verbose=0,
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


def normalize_rank(arr):
    r = rankdata(arr)
    return (r - r.min()) / (r.max() - r.min() + EPS)


def save_sub(name, pred):
    if pred.max() > 1.0 or pred.min() < 0:
        pred = normalize_rank(pred).astype(np.float32)
    else:
        pred = np.clip(pred, 0, 1).astype(np.float32)
    pd.DataFrame({"ID": np.arange(len(pred)), "Predicted": pred}).to_csv(name, index=False)
    print(f"  saved {name}")


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

    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in train_pairs:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in test_pairs:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    # Build v19 features (with self-training)
    print("[v19] standard pipeline (st=0.95)")
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
    pred_init = predict_hgb(X_tr0, y_train, X_te0, n_seeds=5)
    extra_edges = test_pairs[pred_init >= 0.95]
    print(f"  self-train: +{len(extra_edges)} edges")

    adjacency, degree, comp = build_graph(train_pairs, y_train, n_nodes, extra_edges)
    adj_matrix = build_sparse_adj(adjacency, n_nodes)
    d_inv = np.zeros(n_nodes, dtype=np.float32)
    mask = degree > 0
    d_inv[mask] = 1.0 / degree[mask]
    neighbor_tfidf = sparse.diags(d_inv) @ adj_matrix @ node_tfidf

    # Build features for ORIGINAL ordering
    X_train_uv = build_features(
        train_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
        y=y_train, remove_pos=True,
    )
    X_test_uv = build_features(
        test_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
    )

    # Build features for SWAPPED ordering (v, u)
    train_pairs_vu = train_pairs[:, [1, 0]]
    test_pairs_vu = test_pairs[:, [1, 0]]
    X_train_vu = build_features(
        train_pairs_vu, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
        y=y_train, remove_pos=True,
    )
    X_test_vu = build_features(
        test_pairs_vu, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
    )
    print(f"  features: {X_train_uv.shape[1]}")

    # ============================================================
    # 1) BASELINE: standard v19 (no symmetry tricks) — sanity check
    # ============================================================
    print("\n[1] baseline (no symmetry)")
    pred_h_uv = predict_hgb(X_train_uv, y_train, X_test_uv, n_seeds=30)
    pred_c_uv = predict_cat(X_train_uv, y_train, X_test_uv, n_seeds=30)
    rh_uv = normalize_rank(pred_h_uv)
    rc_uv = normalize_rank(pred_c_uv)
    pred_baseline = 0.5 * rh_uv + 0.5 * rc_uv
    save_sub("submission_v23_baseline.csv", pred_baseline)

    # ============================================================
    # 2) TTA only: predict on (u,v) and (v,u), average
    # Trained on (u,v) only — same model as baseline
    # ============================================================
    print("\n[2] TTA (test-time augmentation)")
    pred_h_vu = predict_hgb(X_train_uv, y_train, X_test_vu, n_seeds=30)
    pred_c_vu = predict_cat(X_train_uv, y_train, X_test_vu, n_seeds=30)
    rh_vu = normalize_rank(pred_h_vu)
    rc_vu = normalize_rank(pred_c_vu)

    # Average predictions in rank space
    pred_h_tta = (rh_uv + rh_vu) / 2.0
    pred_c_tta = (rc_uv + rc_vu) / 2.0
    pred_tta = 0.5 * normalize_rank(pred_h_tta) + 0.5 * normalize_rank(pred_c_tta)
    save_sub("submission_v23_tta.csv", pred_tta)

    # ============================================================
    # 3) Training augmentation only
    # Train on doubled data (both orderings), predict on (u,v)
    # ============================================================
    print("\n[3] training augmentation")
    X_train_aug = np.vstack([X_train_uv, X_train_vu])
    y_train_aug = np.concatenate([y_train, y_train])
    print(f"  doubled training: {X_train_aug.shape}")

    pred_h_taug = predict_hgb(X_train_aug, y_train_aug, X_test_uv, n_seeds=30)
    pred_c_taug = predict_cat(X_train_aug, y_train_aug, X_test_uv, n_seeds=30)
    pred_taug = 0.5 * normalize_rank(pred_h_taug) + 0.5 * normalize_rank(pred_c_taug)
    save_sub("submission_v23_train_aug.csv", pred_taug)

    # ============================================================
    # 4) Full symmetry: train aug + TTA
    # ============================================================
    print("\n[4] full symmetry (train aug + TTA)")
    pred_h_taug_vu = predict_hgb(X_train_aug, y_train_aug, X_test_vu, n_seeds=30)
    pred_c_taug_vu = predict_cat(X_train_aug, y_train_aug, X_test_vu, n_seeds=30)

    pred_h_full = (normalize_rank(pred_h_taug) + normalize_rank(pred_h_taug_vu)) / 2.0
    pred_c_full = (normalize_rank(pred_c_taug) + normalize_rank(pred_c_taug_vu)) / 2.0
    pred_full_sym = 0.5 * normalize_rank(pred_h_full) + 0.5 * normalize_rank(pred_c_full)
    save_sub("submission_v23_full_sym.csv", pred_full_sym)

    # ============================================================
    # 5) Safe blend: 70% baseline + 30% TTA
    # ============================================================
    print("\n[5] safe blend")
    save_sub("submission_v23_tta_safe.csv", 0.7 * pred_baseline + 0.3 * pred_tta)

    # ============================================================
    # 6) Strong blend: TTA averaged with v19 (proven)
    # ============================================================
    if Path("submission_v19_hgb_cat_rank.csv").exists():
        v19 = pd.read_csv("submission_v19_hgb_cat_rank.csv")["Predicted"].to_numpy()
        rv19 = normalize_rank(v19)
        save_sub("submission_v23_tta_v19_blend.csv", 0.5 * rv19 + 0.5 * pred_tta)
        save_sub("submission_v23_full_sym_v19_blend.csv", 0.5 * rv19 + 0.5 * pred_full_sym)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
