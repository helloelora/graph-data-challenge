"""
v26d — v26b + a single `comm_cn` feature. Kaggle public: 0.88050 (1st place).

After v26b hit 0.88038, a broader community-feature addition (v26c, 10 new
columns including size percentiles, internal density, multi-resolution
same-community flags) scored *worse* on Kaggle (0.87930). The marginal
features diluted the gradient-boosted signal rather than helping.

Ablation isolated one feature with a very strong pos/neg gap:

  comm_cn = | (N(u) ∩ N(v)) ∩ C(u) | + | (N(u) ∩ N(v)) ∩ C(v) |

that is, common neighbors in the candidate graph that also belong to u's
or v's Louvain community. On training pairs the gap was +0.272
(positives 0.297 vs. negatives 0.026, ~10x ratio).

v26d is therefore v26b *exactly* plus this one column. Same minimal-risk
single-feature-addition discipline we later reused in v26g.

Feature count: 63 (v26b 62 + 1).
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
    load_data, build_graph, build_sparse_adj,
    build_features as build_features_v19,
)
from experiments_v25 import (
    build_partner_sets,
    compute_pair_transductive_v24,
    compute_pair_transductive_v25,
    HGB_PARAMS, CAT_PARAMS,
    predict_hgb, predict_cat,
    normalize_rank, save_sub,
    SEED, EPS,
)
from experiments_v26b import (
    build_candidate_graph,
    run_louvain,
    compute_community_features,
    compute_spectral_embedding,
    compute_spectral_features,
)

import community as community_louvain
import networkx as nx


def compute_comm_cn(pairs, G, partition):
    """For each pair (u,v), count common neighbors w of u and v that belong
    to the same community as u (or v). This is the strongest feature
    from v26c's ablation (pos/neg gap +0.272, 10x ratio).
    """
    comms = {}
    for node, cid in partition.items():
        comms.setdefault(cid, set()).add(node)

    n = pairs.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        cu = partition.get(u, -1)
        cv = partition.get(v, -2)
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        cn_set = (Nu & Nv) - {u, v}
        members_u = comms.get(cu, set())
        members_v = comms.get(cv, set())
        cnt = 0
        for w in cn_set:
            if w in members_u or w in members_v:
                cnt += 1
        out[i] = cnt
    return out.reshape(-1, 1)


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

    print("[v25] pair-level transductive features")
    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v24_test = compute_pair_transductive_v24(test_pairs, train_partners, test_partners)
    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_test = compute_pair_transductive_v25(
        test_pairs, train_partners, test_partners, test_count, train_count, total_count)

    print("\n[v19] self-training (st=0.95)")
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
    print(f"  self-train: +{len(extra_edges)} pseudo-edges")

    adjacency, degree, comp = build_graph(train_pairs, y_train, n_nodes, extra_edges)
    adj_matrix = build_sparse_adj(adjacency, n_nodes)
    d_inv = np.zeros(n_nodes, dtype=np.float32)
    d_inv[degree > 0] = 1.0 / degree[degree > 0]
    neighbor_tfidf = sparse.diags(d_inv) @ adj_matrix @ node_tfidf

    X_train_v19 = build_features_v19(
        train_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count, adj_matrix, neighbor_tfidf,
        y=y_train, remove_pos=True)
    X_test_v19 = build_features_v19(
        test_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count, adj_matrix, neighbor_tfidf)

    X_train_v25 = np.hstack([X_train_v19, pair_v24_train, pair_v25_train])
    X_test_v25 = np.hstack([X_test_v19, pair_v24_test, pair_v25_test])

    # === v26b base ===
    print("\n[v26b] candidate graph + Louvain + spectral")
    G_cand = build_candidate_graph(train_pairs, test_pairs, extra_edges, n_nodes)
    partition = run_louvain(G_cand, seed=SEED)
    comm_train = compute_community_features(train_pairs, partition, n_nodes)
    comm_test = compute_community_features(test_pairs, partition, n_nodes)
    emb, emb_normed = compute_spectral_embedding(G_cand, n_nodes, k=16, seed=SEED)
    spec_train = compute_spectral_features(train_pairs, emb, emb_normed)
    spec_test = compute_spectral_features(test_pairs, emb, emb_normed)

    X_train_v26b = np.hstack([X_train_v25, comm_train, spec_train])
    X_test_v26b = np.hstack([X_test_v25, comm_test, spec_test])
    print(f"  v26b features: {X_train_v26b.shape[1]}")

    # === NEW: comm_cn only ===
    print("\n[v26d] computing comm_cn (single targeted addition)")
    cn_train = compute_comm_cn(train_pairs, G_cand, partition)
    cn_test = compute_comm_cn(test_pairs, G_cand, partition)
    p_mean = cn_train[y_train == 1].mean()
    n_mean = cn_train[y_train == 0].mean()
    print(f"  comm_cn: pos={p_mean:.3f}  neg={n_mean:.3f}  gap={p_mean-n_mean:+.3f}")

    X_train_v26d = np.hstack([X_train_v26b, cn_train])
    X_test_v26d = np.hstack([X_test_v26b, cn_test])
    print(f"  v26d features: {X_train_v26d.shape[1]} (+1)")

    # === CV ===
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v26b = np.zeros(len(y_train), dtype=np.float64)
    oof_v26d = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26d, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v26b[tr], y_train[tr])
        oof_v26b[va] = m1.predict_proba(X_train_v26b[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26d[tr], y_train[tr])
        oof_v26d[va] = m2.predict_proba(X_train_v26d[va])[:, 1]
    auc_b = roc_auc_score(y_train, oof_v26b)
    auc_d = roc_auc_score(y_train, oof_v26d)
    print(f"  v26b OOF = {auc_b:.5f}")
    print(f"  v26d OOF = {auc_d:.5f}  ({auc_d - auc_b:+.5f})")

    # === Final ensembles ===
    print("\n[final] 30-seed HGB+CatBoost on v26d")
    pred_h = predict_hgb(X_train_v26d, y_train, X_test_v26d, n_seeds=30)
    pred_c = predict_cat(X_train_v26d, y_train, X_test_v26d, n_seeds=30)
    pred_v26d = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
    save_sub("submission_v26d.csv", pred_v26d)

    # Blend with known v26b winner
    v26b_ref = pd.read_csv("submission_v26b.csv")["Predicted"].to_numpy()
    rb = normalize_rank(v26b_ref)
    corr, _ = spearmanr(rb, pred_v26d)
    print(f"\n  v26d vs submission_v26b spearman = {corr:.5f}")

    save_sub("submission_v26d_blend_50v26b.csv", 0.5 * pred_v26d + 0.5 * rb)
    save_sub("submission_v26d_blend_70v26d.csv", 0.7 * pred_v26d + 0.3 * rb)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
