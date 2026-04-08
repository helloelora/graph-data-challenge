"""
v24 — pair-level transductive features.

v19's biggest gain (+0.011) came from NODE-level transductive counts
(how many times each node appears in train+test). We push this further:

  test_partners(u)  = nodes that appear in a test pair with u
  train_partners(u) = nodes that appear in a train pair with u

  For pair (u, v):
    - shared_test_partners  = |test_partners(u) ∩ test_partners(v)|
    - shared_train_partners = |train_partners(u) ∩ train_partners(v)|
    - both_in_test_pair     = 1 if (u,v) appears in any test pair (always 1 for test set)
    - shared_partners_total = |all_partners(u) ∩ all_partners(v)|
    - jaccard variant of these

These reveal hidden structure: if u and v share many test partners, they
probably belong to the same cluster in the original graph (and edges between
them were likely deleted). This is leakage-free.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

from best_solution import (
    load_data,
    build_graph,
    build_sparse_adj,
    build_features as build_features_v19,
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


def build_partner_sets(train_pairs, test_pairs, n_nodes):
    """Build the set of partners for each node, separately for train and test."""
    train_partners = [set() for _ in range(n_nodes)]
    test_partners = [set() for _ in range(n_nodes)]

    for u, v in train_pairs:
        u, v = int(u), int(v)
        if u != v:
            train_partners[u].add(v)
            train_partners[v].add(u)

    for u, v in test_pairs:
        u, v = int(u), int(v)
        if u != v:
            test_partners[u].add(v)
            test_partners[v].add(u)

    return train_partners, test_partners


def compute_pair_transductive(pairs, train_partners, test_partners):
    """Compute pair-level transductive features for given pairs."""
    n = pairs.shape[0]

    shared_test = np.zeros(n, dtype=np.float32)
    shared_train = np.zeros(n, dtype=np.float32)
    shared_all = np.zeros(n, dtype=np.float32)
    shared_test_jaccard = np.zeros(n, dtype=np.float32)
    shared_all_jaccard = np.zeros(n, dtype=np.float32)
    test_partners_min = np.zeros(n, dtype=np.float32)
    test_partners_max = np.zeros(n, dtype=np.float32)

    for i in range(n):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        tp_u, tp_v = test_partners[u], test_partners[v]
        rp_u, rp_v = train_partners[u], train_partners[v]

        # Shared test partners (excluding u and v themselves)
        st = len((tp_u & tp_v) - {u, v})
        # Shared train partners (excluding u and v themselves)
        rt = len((rp_u & rp_v) - {u, v})
        # Shared all partners
        all_u = (tp_u | rp_u) - {u, v}
        all_v = (tp_v | rp_v) - {u, v}
        sa = len(all_u & all_v)

        # Jaccards
        union_t = len((tp_u | tp_v) - {u, v})
        st_j = st / max(union_t, 1)
        union_a = len(all_u | all_v)
        sa_j = sa / max(union_a, 1)

        shared_test[i] = st
        shared_train[i] = rt
        shared_all[i] = sa
        shared_test_jaccard[i] = st_j
        shared_all_jaccard[i] = sa_j
        test_partners_min[i] = min(len(tp_u), len(tp_v))
        test_partners_max[i] = max(len(tp_u), len(tp_v))

    return np.column_stack([
        shared_test,
        shared_train,
        shared_all,
        shared_test_jaccard,
        shared_all_jaccard,
        test_partners_min,
        test_partners_max,
    ]).astype(np.float32)


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

    # Original transductive counts (v19)
    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in train_pairs:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in test_pairs:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    # NEW: Pair-level transductive features
    print("[v24] computing pair-level transductive features...")
    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    print(f"  test_partners: max={max(len(s) for s in test_partners)} "
          f"mean={np.mean([len(s) for s in test_partners]):.2f}")

    pair_trans_train = compute_pair_transductive(train_pairs, train_partners, test_partners)
    pair_trans_test = compute_pair_transductive(test_pairs, train_partners, test_partners)
    print(f"  pair_trans features: {pair_trans_train.shape}")
    print(f"  shared_test_partners stats:")
    st_train = pair_trans_train[:, 0]
    st_test = pair_trans_test[:, 0]
    print(f"    train: pos_mean={st_train[y_train==1].mean():.2f} "
          f"neg_mean={st_train[y_train==0].mean():.2f}")
    print(f"    test:  mean={st_test.mean():.2f} max={st_test.max():.0f}")

    # ============================================================
    # Build v19 features (with self-training)
    # ============================================================
    print("\n[v19] standard pipeline (st=0.95)")
    adj0, deg0, comp0 = build_graph(train_pairs, y_train, n_nodes)
    A0 = build_sparse_adj(adj0, n_nodes)
    d_inv0 = np.zeros(n_nodes, dtype=np.float32)
    m0 = deg0 > 0
    d_inv0[m0] = 1.0 / deg0[m0]
    ntfidf0 = sparse.diags(d_inv0) @ A0 @ node_tfidf

    X_tr0_v19 = build_features_v19(
        train_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count,
        A0, ntfidf0, y=y_train, remove_pos=True,
    )
    X_te0_v19 = build_features_v19(
        test_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count,
        A0, ntfidf0,
    )
    pred_init = predict_hgb(X_tr0_v19, y_train, X_te0_v19, n_seeds=5)
    extra_edges = test_pairs[pred_init >= 0.95]
    print(f"  self-train: +{len(extra_edges)} edges")

    adjacency, degree, comp = build_graph(train_pairs, y_train, n_nodes, extra_edges)
    adj_matrix = build_sparse_adj(adjacency, n_nodes)
    d_inv = np.zeros(n_nodes, dtype=np.float32)
    mask = degree > 0
    d_inv[mask] = 1.0 / degree[mask]
    neighbor_tfidf = sparse.diags(d_inv) @ adj_matrix @ node_tfidf

    X_train_v19 = build_features_v19(
        train_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
        y=y_train, remove_pos=True,
    )
    X_test_v19 = build_features_v19(
        test_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
    )

    # Concatenate v19 + new pair transductive
    X_train = np.hstack([X_train_v19, pair_trans_train])
    X_test = np.hstack([X_test_v19, pair_trans_test])
    print(f"  features: v19={X_train_v19.shape[1]} + new={pair_trans_train.shape[1]} = {X_train.shape[1]}")

    # ============================================================
    # Quick CV to estimate gain
    # ============================================================
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v19 = np.zeros(len(y_train), dtype=np.float64)
    oof_v24 = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v19[tr], y_train[tr])
        oof_v19[va] = m1.predict_proba(X_train_v19[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train[tr], y_train[tr])
        oof_v24[va] = m2.predict_proba(X_train[va])[:, 1]
    print(f"  v19 OOF AUC = {roc_auc_score(y_train, oof_v19):.5f}")
    print(f"  v24 OOF AUC = {roc_auc_score(y_train, oof_v24):.5f}")
    print(f"  delta       = {roc_auc_score(y_train, oof_v24) - roc_auc_score(y_train, oof_v19):+.5f}")

    # ============================================================
    # Final: train models and produce submissions
    # ============================================================
    print("\n[final] training 30-seed HGB+CatBoost on v24 features")
    pred_h = predict_hgb(X_train, y_train, X_test, n_seeds=30)
    pred_c = predict_cat(X_train, y_train, X_test, n_seeds=30)
    rh = normalize_rank(pred_h)
    rc = normalize_rank(pred_c)

    pred_v24 = 0.5 * rh + 0.5 * rc
    save_sub("submission_v24.csv", pred_v24)

    # Blends with v19 baseline (safety net)
    if Path("submission_v19_hgb_cat_rank.csv").exists():
        v19 = pd.read_csv("submission_v19_hgb_cat_rank.csv")["Predicted"].to_numpy()
        rv19 = normalize_rank(v19)
        save_sub("submission_v24_blend_50.csv", 0.5 * rv19 + 0.5 * pred_v24)
        save_sub("submission_v24_blend_70v19.csv", 0.7 * rv19 + 0.3 * pred_v24)
        save_sub("submission_v24_blend_30v19.csv", 0.3 * rv19 + 0.7 * pred_v24)

        # Print correlation
        corr, _ = spearmanr(rv19, pred_v24)
        print(f"\n  v24 vs v19 corr = {corr:.5f}")

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
