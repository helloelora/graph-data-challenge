"""
v26R — compound the v26Q SVD direction across multiple axes.

v26Q's svd4_cos (rank-4 SVD cosine similarity on the candidate-graph
adjacency) delivered +0.00162 Kaggle from one new scalar. That's the
third-biggest jump of the whole session and came from a fundamentally
new view of the graph: "do u and v connect to the same set of other
nodes (role similarity)", not "are u and v in the same cluster".

v26R exploits this direction along three orthogonal axes at once:

  Axis 1 — Multi-rank SVD cosine on the unweighted candidate graph:
    svd2_cos, svd3_cos, svd5_cos, svd6_cos, svd12_cos
    Different ranks compress the graph to different numbers of
    "dominant role archetypes". v26L showed that multi-resolution
    Louvain consensus compounds via bimodal winners; the same may
    hold for SVD ranks.

  Axis 2 — SVD cosine on alternative graph constructions:
    svd4_cos on text-weighted candidate graph (same as v26g's graph)
    svd4_cos on A^2 (squared adjacency — captures 2-hop role similarity)
    Each gives a different low-rank view of the same actor set.

  Axis 3 — SVD Hadamard product features (different feature type):
    svd4_hadamard_0, _1, _2, _3   — element-wise product u*v in the
    rank-4 SVD space. Instead of collapsing to one scalar (cos), we
    give the model all four components and let it learn arbitrary
    non-linear interactions. This is a standard link-prediction trick
    from the graph embedding literature (Grover & Leskovec 2016).

Each candidate is ablated individually on the v26Q base (71 features).
Ship only strict positive deltas on the HGB+CatBoost blend OOF.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
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
from experiments_v26g import build_text_weighted_candidate_graph
from experiments_v26h import compute_consensus_same_community
from experiments_v26L import compute_consensus_at_resolution
from experiments_v26Q import (
    candidate_graph_adjacency,
    truncated_svd_embedding,
    svd_pair_features,
)

import networkx as nx


def svd_hadamard_features(pairs, emb):
    """Element-wise product u * v in the SVD embedding space.
    Returns (n_pairs, k) array where each column is a separate feature."""
    u = pairs[:, 0].astype(np.int64)
    v = pairs[:, 1].astype(np.int64)
    return (emb[u] * emb[v]).astype(np.float32)


def weighted_adjacency_from_graph(G, n_nodes):
    """Return a sparse adjacency matrix with edge weights preserved."""
    rows, cols, data = [], [], []
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        rows.append(u); cols.append(v); data.append(w)
        rows.append(v); cols.append(u); data.append(w)
    if not rows:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)


def rnk(a):
    r = rankdata(a)
    return (r - r.min()) / (r.max() - r.min() + EPS)


def blend_oof(X, y_train, cv, seed):
    oof_h = np.zeros(len(y_train), dtype=np.float64)
    oof_c = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=seed + fold)
        m1.fit(X[tr], y_train[tr])
        oof_h[va] = m1.predict_proba(X[va])[:, 1]
        mc = CatBoostClassifier(**CAT_PARAMS, random_seed=seed + fold)
        mc.fit(X[tr], y_train[tr])
        oof_c[va] = mc.predict_proba(X[va])[:, 1]
    blend = 0.5 * rnk(oof_h) + 0.5 * rnk(oof_c)
    return (
        roc_auc_score(y_train, oof_h),
        roc_auc_score(y_train, oof_c),
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

    # v26d base
    G_unwt = build_candidate_graph(train_pairs, test_pairs, extra_edges, n_nodes)
    part_unwt = run_louvain(G_unwt, seed=SEED)
    comm_train = compute_community_features(train_pairs, part_unwt, n_nodes)
    comm_test = compute_community_features(test_pairs, part_unwt, n_nodes)
    emb_lap, emb_lap_normed = compute_spectral_embedding(G_unwt, n_nodes, k=16, seed=SEED)
    spec_train = compute_spectral_features(train_pairs, emb_lap, emb_lap_normed)
    spec_test = compute_spectral_features(test_pairs, emb_lap, emb_lap_normed)
    cn_train = compute_comm_cn(train_pairs, G_unwt, part_unwt)
    cn_test = compute_comm_cn(test_pairs, G_unwt, part_unwt)
    X_train_v26d = np.hstack([X_train_v25, comm_train, spec_train, cn_train])
    X_test_v26d = np.hstack([X_test_v25, comm_test, spec_test, cn_test])

    # v26g base
    G_text = build_text_weighted_candidate_graph(
        train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf, alpha=1.0, beta=3.0)
    part_text = run_louvain(G_text, seed=SEED)
    comm_text_train = compute_community_features(train_pairs, part_text, n_nodes)
    comm_text_test = compute_community_features(test_pairs, part_text, n_nodes)
    X_train_v26g = np.hstack([X_train_v26d, comm_text_train])
    X_test_v26g = np.hstack([X_test_v26d, comm_text_test])

    # v26h_pure
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        test_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])

    # v26L multi-res winners
    for r in [0.7, 1.3, 2.0]:
        tr_arr, _ = compute_consensus_at_resolution(
            train_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        te_arr, _ = compute_consensus_at_resolution(
            test_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        X_train_v26hp = np.hstack([X_train_v26hp, tr_arr.reshape(-1, 1)])
        X_test_v26hp = np.hstack([X_test_v26hp, te_arr.reshape(-1, 1)])
    X_train_v26L = X_train_v26hp
    X_test_v26L = X_test_v26hp

    # v26Q: + svd4_cos on unweighted candidate graph
    A_unwt = candidate_graph_adjacency(G_unwt, n_nodes)
    emb_svd4_unwt = truncated_svd_embedding(A_unwt, k=4, seed=SEED)
    _, _, svd4_cos_tr = svd_pair_features(train_pairs, emb_svd4_unwt)
    _, _, svd4_cos_te = svd_pair_features(test_pairs, emb_svd4_unwt)
    X_train_v26Q = np.hstack([X_train_v26L, svd4_cos_tr])
    X_test_v26Q = np.hstack([X_test_v26L, svd4_cos_te])
    print(f"\n[v26Q base] {X_train_v26Q.shape[1]} features  (v26L + svd4_cos)")

    # === v26R candidate features ===
    candidates_train = {}
    candidates_test = {}

    # Axis 1: more ranks of SVD cosine on unweighted candidate graph
    print("\n[v26R axis 1] SVD cosine at other ranks (unweighted graph)")
    for k in [2, 3, 5, 6, 12]:
        emb_k = truncated_svd_embedding(A_unwt, k=k, seed=SEED)
        _, _, cos_tr = svd_pair_features(train_pairs, emb_k)
        _, _, cos_te = svd_pair_features(test_pairs, emb_k)
        label = f"svd{k}_cos"
        candidates_train[label] = cos_tr
        candidates_test[label] = cos_te
        p = cos_tr[y_train == 1].mean()
        n = cos_tr[y_train == 0].mean()
        print(f"  {label:12s}  pos={p:.3f}  neg={n:.3f}  gap={p-n:+.3f}")

    # Axis 2: SVD cosine on weighted/alternative graphs
    print("\n[v26R axis 2] SVD cosine on alternative graphs")

    # Text-weighted candidate graph
    A_text_wt = weighted_adjacency_from_graph(G_text, n_nodes)
    emb_text_k4 = truncated_svd_embedding(A_text_wt, k=4, seed=SEED)
    _, _, cos_tr = svd_pair_features(train_pairs, emb_text_k4)
    _, _, cos_te = svd_pair_features(test_pairs, emb_text_k4)
    candidates_train["svd4_cos_text"] = cos_tr
    candidates_test["svd4_cos_text"] = cos_te
    p = cos_tr[y_train == 1].mean()
    n = cos_tr[y_train == 0].mean()
    print(f"  svd4_cos_text (text-weighted graph)  pos={p:.3f}  neg={n:.3f}  gap={p-n:+.3f}")

    # A^2 squared adjacency (2-hop role similarity)
    print("  computing A^2 (2-hop adjacency)")
    t1 = time.time()
    A_sq = (A_unwt @ A_unwt).astype(np.float32)
    # Set diagonal to zero (remove self-loops from squaring)
    A_sq.setdiag(0)
    A_sq.eliminate_zeros()
    print(f"    A^2 built in {time.time()-t1:.1f}s  nnz={A_sq.nnz}")
    emb_sq_k4 = truncated_svd_embedding(A_sq, k=4, seed=SEED)
    _, _, cos_tr = svd_pair_features(train_pairs, emb_sq_k4)
    _, _, cos_te = svd_pair_features(test_pairs, emb_sq_k4)
    candidates_train["svd4_cos_A2"] = cos_tr
    candidates_test["svd4_cos_A2"] = cos_te
    p = cos_tr[y_train == 1].mean()
    n = cos_tr[y_train == 0].mean()
    print(f"  svd4_cos_A2 (squared adjacency)      pos={p:.3f}  neg={n:.3f}  gap={p-n:+.3f}")

    # Axis 3: SVD Hadamard features (element-wise product of u and v in rank-4 SVD space)
    print("\n[v26R axis 3] SVD Hadamard features")
    had_tr = svd_hadamard_features(train_pairs, emb_svd4_unwt)  # (n, 4)
    had_te = svd_hadamard_features(test_pairs, emb_svd4_unwt)   # (n, 4)
    for i in range(4):
        label = f"svd4_had{i}"
        candidates_train[label] = had_tr[:, i:i+1]
        candidates_test[label] = had_te[:, i:i+1]
        p = had_tr[y_train == 1, i].mean()
        n = had_tr[y_train == 0, i].mean()
        print(f"  {label:12s}  pos={p:.4f}  neg={n:.4f}  gap={p-n:+.4f}")

    # === Ablation ===
    print("\n[ablation] HGB+CatBoost blend OOF per candidate (base = v26Q)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_h, base_c, base_b = blend_oof(X_train_v26Q, y_train, cv, SEED)
    print(f"  v26Q base   HGB={base_h:.5f}  Cat={base_c:.5f}  blend={base_b:.5f}")

    deltas = {}
    for label in candidates_train:
        X_tr = np.hstack([X_train_v26Q, candidates_train[label]])
        h, c, b = blend_oof(X_tr, y_train, cv, SEED)
        delta = b - base_b
        deltas[label] = (h, c, b, delta)
        flag = "  *" if delta > 0 else ""
        print(f"  +{label:14s}   HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
              f"({delta:+.5f}){flag}")

    # All 4 Hadamard features together (a natural group)
    had_keys = [f"svd4_had{i}" for i in range(4)]
    X_had = np.hstack([X_train_v26Q] + [candidates_train[k] for k in had_keys])
    h, c, b = blend_oof(X_had, y_train, cv, SEED)
    delta = b - base_b
    flag = "  *" if delta > 0 else ""
    print(f"  +svd4 hadamard all 4   HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
          f"({delta:+.5f}){flag}")

    # All multi-rank cosines together
    rank_keys = [f"svd{k}_cos" for k in [2, 3, 5, 6, 12]]
    X_ranks = np.hstack([X_train_v26Q] + [candidates_train[k] for k in rank_keys])
    h, c, b = blend_oof(X_ranks, y_train, cv, SEED)
    delta = b - base_b
    flag = "  *" if delta > 0 else ""
    print(f"  +multi-rank cos all 5  HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
          f"({delta:+.5f}){flag}")

    # Winners only
    winning = [label for label in candidates_train if deltas[label][3] > 0]
    if not winning:
        print("\n[v26R] no single candidate improved the blend. Stopping.")
        return

    print(f"\n[v26R winners] {winning}")
    X_train_v26R = np.hstack([X_train_v26Q] + [candidates_train[l] for l in winning])
    X_test_v26R = np.hstack([X_test_v26Q] + [candidates_test[l] for l in winning])

    h, c, b = blend_oof(X_train_v26R, y_train, cv, SEED)
    print(f"  v26R HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  ({b - base_b:+.5f})")

    print("\n[final] 30-seed HGB+CatBoost on v26R winners")
    pred_h = predict_hgb(X_train_v26R, y_train, X_test_v26R, n_seeds=30)
    pred_c = predict_cat(X_train_v26R, y_train, X_test_v26R, n_seeds=30)
    pred_v26R = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)
    save_sub("submission_v26R.csv", pred_v26R)

    if Path("submission_v26Q.csv").exists():
        v26Q_known = pd.read_csv("submission_v26Q.csv")["Predicted"].to_numpy()
        rQ = rnk(v26Q_known)
        corr, _ = spearmanr(rQ, pred_v26R)
        print(f"\n  v26R vs submission_v26Q.csv spearman = {corr:.5f}")
        save_sub("submission_v26R_blend_50.csv", 0.5 * pred_v26R + 0.5 * rQ)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
