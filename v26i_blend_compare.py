"""Quick diagnostic: compute 5-fold HGB+CatBoost rank-blend OOF for
v26h_pure, v26i_canonical, and v26i_full to decide which to submit."""

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

from best_solution import (
    load_data, build_graph, build_sparse_adj,
    build_features as build_features_v19,
)
from experiments_v25 import (
    build_partner_sets, compute_pair_transductive_v24, compute_pair_transductive_v25,
    HGB_PARAMS, CAT_PARAMS, predict_hgb, SEED, EPS,
)
from experiments_v26b import (
    build_candidate_graph, run_louvain, compute_community_features,
    compute_spectral_embedding, compute_spectral_features,
)
from experiments_v26d import compute_comm_cn
from experiments_v26g import build_text_weighted_candidate_graph
from experiments_v26h import compute_consensus_same_community
from experiments_v26i import (
    compute_all_partitions, consensus_comm_cn,
    consensus_community_sizes, build_canonical_partition,
)


def rnk(a):
    r = rankdata(a)
    return (r - r.min()) / (r.max() - r.min() + EPS)


def blend_oof_auc(X, y_train, cv, seed):
    oof_hgb = np.zeros(len(y_train), dtype=np.float64)
    oof_cat = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=seed + fold)
        m1.fit(X[tr], y_train[tr])
        oof_hgb[va] = m1.predict_proba(X[va])[:, 1]
        mc = CatBoostClassifier(**CAT_PARAMS, random_seed=seed + fold)
        mc.fit(X[tr], y_train[tr])
        oof_cat[va] = mc.predict_proba(X[va])[:, 1]
    blend = 0.5 * rnk(oof_hgb) + 0.5 * rnk(oof_cat)
    return (
        roc_auc_score(y_train, oof_hgb),
        roc_auc_score(y_train, oof_cat),
        roc_auc_score(y_train, blend),
    )


def main():
    t0 = time.time()
    np.random.seed(SEED)

    train_pairs, y_train, test_pairs, node_features = load_data(
        Path("train.txt"), Path("test.txt"), Path("node_information.csv")
    )
    n_nodes = node_features.shape[0]
    nf_sparse = sparse.csr_matrix(node_features)
    node_tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(nf_sparse)

    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in train_pairs:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in test_pairs:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)

    adj0, deg0, comp0 = build_graph(train_pairs, y_train, n_nodes)
    A0 = build_sparse_adj(adj0, n_nodes)
    d_inv0 = np.zeros(n_nodes, dtype=np.float32)
    d_inv0[deg0 > 0] = 1.0 / deg0[deg0 > 0]
    ntfidf0 = sparse.diags(d_inv0) @ A0 @ node_tfidf
    X_tr0 = build_features_v19(
        train_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count, A0, ntfidf0, y=y_train, remove_pos=True)
    X_te0 = build_features_v19(
        test_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count, A0, ntfidf0)
    pred_init = predict_hgb(X_tr0, y_train, X_te0, n_seeds=5)
    extra_edges = test_pairs[pred_init >= 0.95]

    adjacency, degree, comp = build_graph(train_pairs, y_train, n_nodes, extra_edges)
    adj_matrix = build_sparse_adj(adjacency, n_nodes)
    d_inv = np.zeros(n_nodes, dtype=np.float32)
    d_inv[degree > 0] = 1.0 / degree[degree > 0]
    neighbor_tfidf = sparse.diags(d_inv) @ adj_matrix @ node_tfidf

    X_train_v19 = build_features_v19(
        train_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count, adj_matrix, neighbor_tfidf,
        y=y_train, remove_pos=True)

    X_train_v25 = np.hstack([X_train_v19, pair_v24_train, pair_v25_train])

    # Build the shared base up to v26h_pure
    G_unwt = build_candidate_graph(train_pairs, test_pairs, extra_edges, n_nodes)
    part_unwt = run_louvain(G_unwt, seed=SEED)
    comm_train = compute_community_features(train_pairs, part_unwt, n_nodes)
    emb, emb_normed = compute_spectral_embedding(G_unwt, n_nodes, k=16, seed=SEED)
    spec_train = compute_spectral_features(train_pairs, emb, emb_normed)
    cn_train = compute_comm_cn(train_pairs, G_unwt, part_unwt)
    X_train_v26d = np.hstack([X_train_v25, comm_train, spec_train, cn_train])

    G_text = build_text_weighted_candidate_graph(
        train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf, alpha=1.0, beta=3.0)
    part_text = run_louvain(G_text, seed=SEED)
    comm_text_train = compute_community_features(train_pairs, part_text, n_nodes)
    X_train_v26g = np.hstack([X_train_v26d, comm_text_train])

    partitions_unwt = compute_all_partitions(G_unwt, n_seeds=20, base_seed=SEED)
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)

    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])

    # v26i_canonical
    canonical = build_canonical_partition(partitions_unwt, n_nodes, threshold=0.5, seed=SEED)
    canon_comm_train = compute_community_features(train_pairs, canonical, n_nodes)
    X_train_v26ic = np.hstack([X_train_v26hp, canon_comm_train])

    # v26i full (canonical + cons_sizes + cons_comm_cn)
    cons_ccn_train = consensus_comm_cn(train_pairs, G_unwt, partitions_unwt)
    cons_smin_train, cons_smax_train = consensus_community_sizes(train_pairs, partitions_unwt)
    X_train_v26if = np.hstack([
        X_train_v26hp,
        cons_ccn_train,
        cons_smin_train,
        cons_smax_train,
        np.hstack([cons_smin_train, cons_smax_train]),  # matches v26i's winners pollution
        canon_comm_train,
    ])

    print(f"shapes: hp {X_train_v26hp.shape}  canonical {X_train_v26ic.shape}  full {X_train_v26if.shape}")

    # Proper HGB+CatBoost rank-blend OOF for each
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("\n[v26h_pure]")
    h, c, b = blend_oof_auc(X_train_v26hp, y_train, cv, SEED)
    print(f"  HGB OOF = {h:.5f}")
    print(f"  Cat OOF = {c:.5f}")
    print(f"  HGB+Cat rank blend OOF = {b:.5f}")
    hp_blend = b

    print("\n[v26i_canonical]  (+canonical same_comm 3 cols)")
    h, c, b = blend_oof_auc(X_train_v26ic, y_train, cv, SEED)
    print(f"  HGB OOF = {h:.5f}")
    print(f"  Cat OOF = {c:.5f}")
    print(f"  HGB+Cat rank blend OOF = {b:.5f}  ({b - hp_blend:+.5f})")

    print("\n[v26i_full]  (+canonical + cons_ccn + cons_sizes)")
    h, c, b = blend_oof_auc(X_train_v26if, y_train, cv, SEED)
    print(f"  HGB OOF = {h:.5f}")
    print(f"  Cat OOF = {c:.5f}")
    print(f"  HGB+Cat rank blend OOF = {b:.5f}  ({b - hp_blend:+.5f})")

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
