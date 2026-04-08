"""
v25 — Extended pair-level transductive features.

v24 added 7 pair transductive features and gained +0.003 Kaggle (0.86766 -> 0.87091).
Pure v24 outperformed any blend with v19, suggesting MORE features in this
direction would help further. v25 adds 8 more features in the same family:

  Higher-order:
    - test_triangles      = |{w : (u,w) and (w,v) both in test pairs}|
    - train_triangles     = |{w : (u,w) and (w,v) both in train pairs}|
    - mixed_triangles     = |{w : (u,w) in train and (w,v) in test, plus symmetric}|

  Weighted shared partners:
    - shared_test_aa      = sum 1/log(test_count(w)) for w in shared test partners
    - shared_test_ra      = sum 1/test_count(w)
    - shared_total_pa     = |test_partners(u)| * |test_partners(v)|

  Asymmetry:
    - exclusive_test_u    = |test_partners(u) \ test_partners(v)|
    - exclusive_test_v    = |test_partners(v) \ test_partners(u)|
"""

import math
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


def compute_pair_transductive_v24(pairs, train_partners, test_partners):
    """Original v24 features (7) — kept identical to v24 implementation."""
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

        st = len((tp_u & tp_v) - {u, v})
        rt = len((rp_u & rp_v) - {u, v})
        all_u = (tp_u | rp_u) - {u, v}
        all_v = (tp_v | rp_v) - {u, v}
        sa = len(all_u & all_v)

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
        shared_test, shared_train, shared_all,
        shared_test_jaccard, shared_all_jaccard,
        test_partners_min, test_partners_max,
    ]).astype(np.float32)


def compute_pair_transductive_v25(pairs, train_partners, test_partners,
                                   test_count, train_count, total_count):
    """New v25 features (8) — extending v24."""
    n = pairs.shape[0]

    test_triangles = np.zeros(n, dtype=np.float32)
    train_triangles = np.zeros(n, dtype=np.float32)
    mixed_triangles = np.zeros(n, dtype=np.float32)
    shared_test_aa = np.zeros(n, dtype=np.float32)
    shared_test_ra = np.zeros(n, dtype=np.float32)
    shared_total_pa = np.zeros(n, dtype=np.float32)
    exclusive_test_u = np.zeros(n, dtype=np.float32)
    exclusive_test_v = np.zeros(n, dtype=np.float32)

    for i in range(n):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        tp_u, tp_v = test_partners[u], test_partners[v]
        rp_u, rp_v = train_partners[u], train_partners[v]

        # Test triangles: w such that (u,w) AND (w,v) BOTH in test pairs
        # i.e., w ∈ test_partners(u) ∩ test_partners(v), excluding u and v
        shared_test_set = (tp_u & tp_v) - {u, v}
        test_triangles[i] = len(shared_test_set)

        # Train triangles
        shared_train_set = (rp_u & rp_v) - {u, v}
        train_triangles[i] = len(shared_train_set)

        # Mixed triangles: w in train_partners(u) AND test_partners(v), or vice versa
        mixed_a = (rp_u & tp_v) - {u, v}
        mixed_b = (tp_u & rp_v) - {u, v}
        mixed_triangles[i] = len(mixed_a) + len(mixed_b)

        # Adamic-Adar style on test partner space
        # For each w in shared_test_set, weight by 1/log(test_count(w))
        aa_sum = 0.0
        ra_sum = 0.0
        for w in shared_test_set:
            tc = test_count[w]
            if tc > 1:
                aa_sum += 1.0 / math.log(tc)
            if tc > 0:
                ra_sum += 1.0 / tc
        shared_test_aa[i] = aa_sum
        shared_test_ra[i] = ra_sum

        # Preferential attachment in test space
        shared_total_pa[i] = len(tp_u) * len(tp_v)

        # Asymmetry: exclusive partners
        exclusive_test_u[i] = len((tp_u - tp_v) - {v})
        exclusive_test_v[i] = len((tp_v - tp_u) - {u})

    return np.column_stack([
        test_triangles, train_triangles, mixed_triangles,
        shared_test_aa, shared_test_ra,
        shared_total_pa,
        exclusive_test_u, exclusive_test_v,
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

    # v19 transductive counts
    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in train_pairs:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in test_pairs:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    # Pair transductive features (v24 + v25)
    print("[v25] computing pair-level transductive features (v24 + v25 extensions)")
    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)

    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v24_test = compute_pair_transductive_v24(test_pairs, train_partners, test_partners)

    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_test = compute_pair_transductive_v25(
        test_pairs, train_partners, test_partners, test_count, train_count, total_count)

    print(f"  v24 features: {pair_v24_train.shape[1]}")
    print(f"  v25 new features: {pair_v25_train.shape[1]}")
    print(f"  test_triangles stats:")
    tt = pair_v25_train[:, 0]
    print(f"    train pos_mean={tt[y_train==1].mean():.3f} neg_mean={tt[y_train==0].mean():.3f}")
    tt_test = pair_v25_test[:, 0]
    print(f"    test mean={tt_test.mean():.3f} max={tt_test.max():.0f}")

    # ============================================================
    # v19 features (with self-training)
    # ============================================================
    print("\n[v19] standard pipeline (st=0.95)")
    adj0, deg0, comp0 = build_graph(train_pairs, y_train, n_nodes)
    A0 = build_sparse_adj(adj0, n_nodes)
    d_inv0 = np.zeros(n_nodes, dtype=np.float32)
    m0 = deg0 > 0
    d_inv0[m0] = 1.0 / deg0[m0]
    ntfidf0 = sparse.diags(d_inv0) @ A0 @ node_tfidf

    X_tr0 = build_features_v19(
        train_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count,
        A0, ntfidf0, y=y_train, remove_pos=True,
    )
    X_te0 = build_features_v19(
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

    # v24 features = v19 + 7 pair transductive
    X_train_v24 = np.hstack([X_train_v19, pair_v24_train])
    X_test_v24 = np.hstack([X_test_v19, pair_v24_test])

    # v25 features = v24 + 8 new pair transductive
    X_train_v25 = np.hstack([X_train_v24, pair_v25_train])
    X_test_v25 = np.hstack([X_test_v24, pair_v25_test])

    print(f"  v19 features: {X_train_v19.shape[1]}")
    print(f"  v24 features: {X_train_v24.shape[1]}")
    print(f"  v25 features: {X_train_v25.shape[1]}")

    # ============================================================
    # CV comparison v19 vs v24 vs v25
    # ============================================================
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v19 = np.zeros(len(y_train), dtype=np.float64)
    oof_v24 = np.zeros(len(y_train), dtype=np.float64)
    oof_v25 = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v25, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v19[tr], y_train[tr])
        oof_v19[va] = m1.predict_proba(X_train_v19[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v24[tr], y_train[tr])
        oof_v24[va] = m2.predict_proba(X_train_v24[va])[:, 1]
        m3 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m3.fit(X_train_v25[tr], y_train[tr])
        oof_v25[va] = m3.predict_proba(X_train_v25[va])[:, 1]

    auc_v19 = roc_auc_score(y_train, oof_v19)
    auc_v24 = roc_auc_score(y_train, oof_v24)
    auc_v25 = roc_auc_score(y_train, oof_v25)
    print(f"  v19 OOF AUC = {auc_v19:.5f}")
    print(f"  v24 OOF AUC = {auc_v24:.5f}  ({auc_v24-auc_v19:+.5f})")
    print(f"  v25 OOF AUC = {auc_v25:.5f}  ({auc_v25-auc_v24:+.5f})")

    # ============================================================
    # Train final models on v25 features
    # ============================================================
    print("\n[final] training 30-seed HGB+CatBoost on v25")
    pred_h = predict_hgb(X_train_v25, y_train, X_test_v25, n_seeds=30)
    pred_c = predict_cat(X_train_v25, y_train, X_test_v25, n_seeds=30)
    rh = normalize_rank(pred_h)
    rc = normalize_rank(pred_c)
    pred_v25 = 0.5 * rh + 0.5 * rc
    save_sub("submission_v25.csv", pred_v25)

    # Also retrain on v24 for the blend reference
    print("[ref] training v24 for comparison")
    pred_h_v24 = predict_hgb(X_train_v24, y_train, X_test_v24, n_seeds=30)
    pred_c_v24 = predict_cat(X_train_v24, y_train, X_test_v24, n_seeds=30)
    pred_v24 = 0.5 * normalize_rank(pred_h_v24) + 0.5 * normalize_rank(pred_c_v24)

    # Blends
    save_sub("submission_v25_blend_50.csv", 0.5 * pred_v25 + 0.5 * pred_v24)
    save_sub("submission_v25_blend_70v25.csv", 0.7 * pred_v25 + 0.3 * pred_v24)
    save_sub("submission_v25_blend_30v25.csv", 0.3 * pred_v25 + 0.7 * pred_v24)

    # Use the existing winning v24 submission as reference if available
    if Path("submission_v24.csv").exists():
        v24_real = pd.read_csv("submission_v24.csv")["Predicted"].to_numpy()
        rv24 = normalize_rank(v24_real)
        save_sub("submission_v25_realv24_50.csv", 0.5 * pred_v25 + 0.5 * rv24)
        save_sub("submission_v25_realv24_70v25.csv", 0.7 * pred_v25 + 0.3 * rv24)
        corr, _ = spearmanr(rv24, pred_v25)
        print(f"\n  v25 vs real_v24 corr = {corr:.5f}")

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
