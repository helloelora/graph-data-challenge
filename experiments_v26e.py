"""
v26e — v26d + Leiden community features.

Leiden (Traag et al. 2019, "From Louvain to Leiden") fixes known flaws in
Louvain's refinement step — specifically, Louvain can produce disconnected
communities while Leiden guarantees well-connected ones. On many graphs
Leiden finds meaningfully different partitions than Louvain, which should
give us an orthogonal community feature to blend with the existing one.

We add 3 features (matching the v26b community family) computed on the
SAME label-free candidate graph but with Leiden instead of Louvain:
  - leiden_same_comm
  - leiden_size_min, leiden_size_max

Rationale: v28 (TabPFN blend) showed that v26d features dominate the model
agreement. The only way forward is orthogonal FEATURES, not orthogonal
models. Leiden's partition is likely to identify different cluster
boundaries than Louvain, creating new signal.

Reference:
  Traag, Waltman, van Eck - From Louvain to Leiden: guaranteeing
  well-connected communities. Scientific Reports 9 (2019).
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
from experiments_v26d import compute_comm_cn

import networkx as nx
import igraph as ig
import leidenalg


def run_leiden(G_nx, seed=SEED):
    """Run Leiden on a networkx graph via igraph conversion.
    Returns a dict node_id -> community_id compatible with our
    compute_community_features() helper.
    """
    nodes = sorted(G_nx.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_nx.edges()]
    weights = [G_nx[u][v].get("weight", 1.0) for u, v in G_nx.edges()]

    g_ig = ig.Graph(
        n=len(nodes),
        edges=edges,
        edge_attrs={"weight": weights},
        directed=False,
    )

    # ModularityVertexPartition is the standard Leiden objective
    partition = leidenalg.find_partition(
        g_ig, leidenalg.ModularityVertexPartition,
        weights="weight", seed=seed,
    )

    result = {}
    for cid, members in enumerate(partition):
        for idx in members:
            result[nodes[idx]] = cid
    return result


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

    print("\n[v19] self-training")
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
    print(f"  +{len(extra_edges)} pseudo-edges")

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

    # v26d base = v25 + community + spectral + comm_cn
    print("\n[v26d] candidate graph + Louvain + spectral + comm_cn")
    G_cand = build_candidate_graph(train_pairs, test_pairs, extra_edges, n_nodes)
    part_louvain = run_louvain(G_cand, seed=SEED)
    comm_train = compute_community_features(train_pairs, part_louvain, n_nodes)
    comm_test = compute_community_features(test_pairs, part_louvain, n_nodes)
    emb, emb_normed = compute_spectral_embedding(G_cand, n_nodes, k=16, seed=SEED)
    spec_train = compute_spectral_features(train_pairs, emb, emb_normed)
    spec_test = compute_spectral_features(test_pairs, emb, emb_normed)
    cn_train = compute_comm_cn(train_pairs, G_cand, part_louvain)
    cn_test = compute_comm_cn(test_pairs, G_cand, part_louvain)

    X_train_v26d = np.hstack([X_train_v25, comm_train, spec_train, cn_train])
    X_test_v26d = np.hstack([X_test_v25, comm_test, spec_test, cn_test])
    print(f"  v26d features: {X_train_v26d.shape[1]}")

    # === v26e NEW: Leiden community on the same candidate graph ===
    print("\n[v26e] Leiden community detection")
    part_leiden = run_leiden(G_cand, seed=SEED)
    n_leiden = len(set(part_leiden.values()))
    n_louvain = len(set(part_louvain.values()))
    print(f"  Louvain: {n_louvain} communities")
    print(f"  Leiden:  {n_leiden} communities")

    # How different are the two partitions?
    agreement = sum(
        1 for n in part_louvain
        if part_louvain[n] == part_leiden[n]
    ) / len(part_louvain)
    # That's meaningless for different ID schemes. Use ARI instead.
    from sklearn.metrics import adjusted_rand_score
    nodes_list = sorted(part_louvain.keys())
    l_lab = [part_louvain[n] for n in nodes_list]
    e_lab = [part_leiden[n] for n in nodes_list]
    ari = adjusted_rand_score(l_lab, e_lab)
    print(f"  ARI(Louvain, Leiden) = {ari:.4f}   "
          f"(1=identical, 0=random; lower = more orthogonal signal)")

    leiden_comm_train = compute_community_features(train_pairs, part_leiden, n_nodes)
    leiden_comm_test = compute_community_features(test_pairs, part_leiden, n_nodes)
    print(f"  leiden same_comm: pos={leiden_comm_train[y_train==1, 0].mean():.3f}  "
          f"neg={leiden_comm_train[y_train==0, 0].mean():.3f}")

    # Also add leiden-aware comm_cn
    leiden_cn_train = compute_comm_cn(train_pairs, G_cand, part_leiden)
    leiden_cn_test = compute_comm_cn(test_pairs, G_cand, part_leiden)
    print(f"  leiden comm_cn:  pos={leiden_cn_train[y_train==1].mean():.3f}  "
          f"neg={leiden_cn_train[y_train==0].mean():.3f}")

    X_train_v26e = np.hstack([X_train_v26d, leiden_comm_train, leiden_cn_train])
    X_test_v26e = np.hstack([X_test_v26d, leiden_comm_test, leiden_cn_test])
    print(f"  v26e features: {X_train_v26e.shape[1]} (+{X_train_v26e.shape[1] - X_train_v26d.shape[1]})")

    # === CV comparison ===
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v26d = np.zeros(len(y_train), dtype=np.float64)
    oof_v26e = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26e, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v26d[tr], y_train[tr])
        oof_v26d[va] = m1.predict_proba(X_train_v26d[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26e[tr], y_train[tr])
        oof_v26e[va] = m2.predict_proba(X_train_v26e[va])[:, 1]
    auc_v26d = roc_auc_score(y_train, oof_v26d)
    auc_v26e = roc_auc_score(y_train, oof_v26e)
    print(f"  v26d OOF = {auc_v26d:.5f}")
    print(f"  v26e OOF = {auc_v26e:.5f}  ({auc_v26e - auc_v26d:+.5f})")

    # === Final 30-seed HGB+CatBoost ===
    print("\n[final] 30-seed HGB+CatBoost on v26e")
    pred_h = predict_hgb(X_train_v26e, y_train, X_test_v26e, n_seeds=30)
    pred_c = predict_cat(X_train_v26e, y_train, X_test_v26e, n_seeds=30)
    pred_v26e = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
    save_sub("submission_v26e.csv", pred_v26e)

    # Compare to known v26d submission
    v26d_known = pd.read_csv("submission_v26d.csv")["Predicted"].to_numpy()
    rd = normalize_rank(v26d_known)
    corr, _ = spearmanr(rd, pred_v26e)
    print(f"\n  v26e vs submission_v26d.csv spearman = {corr:.5f}")

    save_sub("submission_v26e_blend_50v26d.csv", 0.5 * pred_v26e + 0.5 * rd)
    save_sub("submission_v26e_blend_70v26e.csv", 0.7 * pred_v26e + 0.3 * rd)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
