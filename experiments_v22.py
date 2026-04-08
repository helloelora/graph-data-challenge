"""
v22 — fundamentally different approaches built on v19 base.

The lesson from v21: small perturbations of v19 (more seeds, +LGB, dual self-train)
all hurt slightly. v19 is at a tight local optimum. To improve, we need
fundamentally different mechanisms.

Approaches:
  1. Pseudo-labeling: add high-confidence test pairs as training rows (not edges)
  2. Feature bagging: train on different feature subsets, blend predictions
  3. Hyperparam micro-variations: test if v19 hyperparams are themselves optimal
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


def predict_hgb(X_tr, y_tr, X_te, n_seeds, base_seed=SEED, params=None):
    p = params or HGB_PARAMS
    pred = np.zeros(len(X_te), dtype=np.float64)
    for s in range(n_seeds):
        m = HistGradientBoostingClassifier(**p, random_state=base_seed + s * 31)
        m.fit(X_tr, y_tr)
        pred += m.predict_proba(X_te)[:, 1]
    return pred / n_seeds


def predict_cat(X_tr, y_tr, X_te, n_seeds, base_seed=SEED, params=None):
    p = params or CAT_PARAMS
    pred = np.zeros(len(X_te), dtype=np.float64)
    for s in range(n_seeds):
        m = CatBoostClassifier(**p, random_seed=base_seed + s * 31)
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


def build_v19_features(train_pairs, y_train, test_pairs, node_features, n_nodes,
                       node_tfidf, total_count, train_count, test_count,
                       st_threshold=0.95):
    """Standard v19 feature building with self-training."""
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
    extra_edges = test_pairs[pred_init >= st_threshold]
    print(f"  self-train (t={st_threshold}): +{len(extra_edges)} edges")

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
    return X_train, X_test, pred_init


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

    # ============================================================
    # Build base v19 features
    # ============================================================
    print("\n[v19 base features]")
    X_train, X_test, pred_init = build_v19_features(
        train_pairs, y_train, test_pairs, node_features, n_nodes,
        node_tfidf, total_count, train_count, test_count, st_threshold=0.95,
    )
    print(f"  features: {X_train.shape[1]}")

    # Train base v19 (HGB+Cat 30 seeds) for reference
    print("[base] HGB+Cat 30 seeds")
    pred_h_base = predict_hgb(X_train, y_train, X_test, n_seeds=30)
    pred_c_base = predict_cat(X_train, y_train, X_test, n_seeds=30)
    rh_base = normalize_rank(pred_h_base)
    rc_base = normalize_rank(pred_c_base)
    pred_v19_base = 0.5 * rh_base + 0.5 * rc_base

    # ============================================================
    # APPROACH 1: Pseudo-labeling test → training rows
    # ============================================================
    print("\n[approach 1] Pseudo-labeling test pairs into training set")

    # Identify high-confidence test pairs from base v19
    pseudo_pos_mask = pred_v19_base >= 0.95
    pseudo_neg_mask = pred_v19_base <= 0.05
    n_pos = pseudo_pos_mask.sum()
    n_neg = pseudo_neg_mask.sum()
    print(f"  pseudo: +{n_pos} positives, +{n_neg} negatives at thresholds 0.95/0.05")

    # Build augmented training data
    pseudo_pairs = np.vstack([
        test_pairs[pseudo_pos_mask],
        test_pairs[pseudo_neg_mask],
    ])
    pseudo_labels = np.concatenate([
        np.ones(n_pos, dtype=np.int32),
        np.zeros(n_neg, dtype=np.int32),
    ])

    aug_train_pairs = np.vstack([train_pairs, pseudo_pairs])
    aug_y = np.concatenate([y_train, pseudo_labels])

    # IMPORTANT: rebuild graph with pseudo-positive edges added too
    aug_extra_edges = test_pairs[pseudo_pos_mask]
    adj_aug, deg_aug, comp_aug = build_graph(train_pairs, y_train, n_nodes, aug_extra_edges)
    A_aug = build_sparse_adj(adj_aug, n_nodes)
    d_inv_aug = np.zeros(n_nodes, dtype=np.float32)
    mask_aug = deg_aug > 0
    d_inv_aug[mask_aug] = 1.0 / deg_aug[mask_aug]
    ntfidf_aug = sparse.diags(d_inv_aug) @ A_aug @ node_tfidf

    X_aug_train = build_features(
        aug_train_pairs, adj_aug, deg_aug, comp_aug, node_features, node_tfidf,
        total_count, train_count, test_count,
        A_aug, ntfidf_aug,
        y=aug_y, remove_pos=True,
    )
    X_aug_test = build_features(
        test_pairs, adj_aug, deg_aug, comp_aug, node_features, node_tfidf,
        total_count, train_count, test_count,
        A_aug, ntfidf_aug,
    )
    print(f"  aug train: {X_aug_train.shape[0]} rows, {X_aug_train.shape[1]} features")

    pred_h_aug = predict_hgb(X_aug_train, aug_y, X_aug_test, n_seeds=30)
    pred_c_aug = predict_cat(X_aug_train, aug_y, X_aug_test, n_seeds=30)
    pred_pseudo = 0.5 * normalize_rank(pred_h_aug) + 0.5 * normalize_rank(pred_c_aug)
    save_sub("submission_v22_pseudo.csv", pred_pseudo)

    # Blend pseudo with base v19 (safer)
    save_sub("submission_v22_pseudo_blend.csv", 0.5 * pred_pseudo + 0.5 * pred_v19_base)

    # ============================================================
    # APPROACH 2: Feature bagging
    # ============================================================
    print("\n[approach 2] Feature bagging — train on different feature subsets")

    n_features = X_train.shape[1]
    print(f"  total features: {n_features}")

    # Define feature subsets (hand-crafted by groups)
    # v19 layout: 20 base + 4 transductive + 12 v16 + 5 v19 interactions = 41
    subsets = {
        "all":     np.arange(n_features),
        "graph":   np.concatenate([np.arange(0, 13), np.arange(26, 36)]),  # graph topology
        "text":    np.concatenate([np.arange(13, 20), np.arange(28, 31)]),  # text-only
        "trans":   np.arange(20, 26),  # transductive + tec
        "no_trans": np.concatenate([np.arange(0, 20), np.arange(26, n_features)]),  # everything except trans
    }

    bag_preds = []
    for name, idxs in subsets.items():
        print(f"  bag '{name}': {len(idxs)} features")
        ph = predict_hgb(X_train[:, idxs], y_train, X_test[:, idxs], n_seeds=15)
        pc = predict_cat(X_train[:, idxs], y_train, X_test[:, idxs], n_seeds=15)
        bag = 0.5 * normalize_rank(ph) + 0.5 * normalize_rank(pc)
        bag_preds.append(bag)

    pred_bagging = np.mean(bag_preds, axis=0)
    save_sub("submission_v22_fbagging.csv", pred_bagging)
    save_sub("submission_v22_fbagging_blend.csv", 0.6 * pred_v19_base + 0.4 * pred_bagging)

    # ============================================================
    # APPROACH 3: Hyperparam micro-variations
    # ============================================================
    print("\n[approach 3] Hyperparam micro-variations")

    variants = [
        ("d4", dict(learning_rate=0.05, max_depth=4, max_iter=400, min_samples_leaf=40, l2_regularization=0.1)),
        ("leaf30", dict(learning_rate=0.05, max_depth=3, max_iter=400, min_samples_leaf=30, l2_regularization=0.1)),
        ("leaf60", dict(learning_rate=0.05, max_depth=3, max_iter=400, min_samples_leaf=60, l2_regularization=0.1)),
        ("lr03", dict(learning_rate=0.03, max_depth=3, max_iter=600, min_samples_leaf=40, l2_regularization=0.1)),
        ("l2_05", dict(learning_rate=0.05, max_depth=3, max_iter=400, min_samples_leaf=40, l2_regularization=0.05)),
    ]

    variant_preds = []
    for name, p in variants:
        ph = predict_hgb(X_train, y_train, X_test, n_seeds=15, params=p)
        variant_preds.append(normalize_rank(ph))

    # Average all variants + base
    pred_hp_avg = np.mean(variant_preds + [rh_base], axis=0)
    pred_hp_full = 0.5 * pred_hp_avg + 0.5 * rc_base  # blend HGB variants with CatBoost
    save_sub("submission_v22_hp_avg.csv", pred_hp_full)

    # ============================================================
    # APPROACH 4: Mega blend (everything together)
    # ============================================================
    print("\n[approach 4] Mega blend")

    mega = (
        0.30 * pred_v19_base +
        0.25 * pred_pseudo +
        0.20 * pred_bagging +
        0.25 * pred_hp_full
    )
    save_sub("submission_v22_mega.csv", mega)

    # ============================================================
    # APPROACH 5: Conservative — v19 base + tiny pseudo perturbation
    # ============================================================
    print("\n[approach 5] Conservative blend")
    save_sub("submission_v22_safe.csv", 0.7 * pred_v19_base + 0.3 * pred_pseudo)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
