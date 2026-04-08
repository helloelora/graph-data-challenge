"""
v26 — v25 features + community detection + spectral embedding on the
      AUGMENTED candidate graph (train positives + pseudo-edges + test pairs).

Motivation:
  v25 saturated at 0.87208. All 6 controlled experiments from v21-v23 tried to
  perturb the existing feature set and every one lost ~0.001. The v24->v25 jump
  only came from introducing an orthogonal feature family (pair-level transductive
  intersections). v26 applies the same principle: a fundamentally new information
  source — global community structure + spectral geometry — computed on a graph
  that includes test pairs as candidate edges (no labels used, so leakage-free).

What we add (6 new features on top of v25's 56):
  1. same_community           : are u and v in the same Louvain community?
  2. community_size_min       : min(|C(u)|, |C(v)|)
  3. community_size_max       : max(|C(u)|, |C(v)|)
  4. spectral_dist_l2         : || phi(u) - phi(v) ||_2 on 16-dim Laplacian eigenmap
  5. spectral_dot             : phi(u) . phi(v)
  6. spectral_cos             : cos angle between phi(u) and phi(v)

Why augmented graph:
  - train positives: the edges we know are real
  - self-training pseudo-edges: high-confidence v25 predictions (threshold 0.95)
  - test pairs as UNDIRECTED candidates: nobody uses test labels, so adding them
    as graph edges is purely a transductive structural signal — exactly the same
    idea that made v19 (node counts) and v24/v25 (pair intersections) work

References:
  - Kunegis & Lommatzsch — Learning Spectral Graph Transformations for Link
    Prediction (ICML 2009). Uses Laplacian eigenvectors + learned spectral
    weighting; direct inspiration for `spectral_dist_l2`.
  - Blondel et al. — Fast unfolding of communities in large networks
    (J. Stat. Mech. 2008). The Louvain method.
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
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
from experiments_v25 import (
    build_partner_sets,
    compute_pair_transductive_v24,
    compute_pair_transductive_v25,
    HGB_PARAMS,
    CAT_PARAMS,
    predict_hgb,
    predict_cat,
    normalize_rank,
    save_sub,
    SEED,
    EPS,
)

import community as community_louvain  # python-louvain
import networkx as nx


def build_augmented_graph(train_pairs, y_train, test_pairs, extra_edges, n_nodes):
    """Undirected graph combining:
       - labeled positive train edges
       - self-trained pseudo-edges
       - test pairs as candidate edges (weight 0.5, no labels used)

    Returns a networkx Graph and a scipy sparse adjacency.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Known positives (weight 1.0)
    for u, v in train_pairs[y_train == 1]:
        u, v = int(u), int(v)
        if u != v:
            G.add_edge(u, v, weight=1.0)

    # Self-training pseudo-edges (weight 0.75 — less confident than true positives)
    if extra_edges is not None:
        for u, v in extra_edges:
            u, v = int(u), int(v)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v, weight=0.75)

    # Test pairs as candidates (weight 0.5 — purely structural, no labels)
    for u, v in test_pairs:
        u, v = int(u), int(v)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, weight=0.5)

    return G


def run_louvain(G, seed=SEED):
    """Run Louvain community detection. Returns dict node_id -> community_id."""
    # python-louvain is deterministic only when given a fixed seed through random_state
    partition = community_louvain.best_partition(G, weight="weight", random_state=seed)
    return partition


def compute_community_features(pairs, partition, n_nodes):
    """Compute same_community, community_size_u, community_size_v for each pair."""
    # Build community size map
    sizes = {}
    for node, cid in partition.items():
        sizes[cid] = sizes.get(cid, 0) + 1

    n = pairs.shape[0]
    same_comm = np.zeros(n, dtype=np.float32)
    size_min = np.zeros(n, dtype=np.float32)
    size_max = np.zeros(n, dtype=np.float32)

    for i in range(n):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        cu = partition.get(u, -1)
        cv = partition.get(v, -2)
        same_comm[i] = 1.0 if cu == cv else 0.0
        su = sizes.get(cu, 1)
        sv = sizes.get(cv, 1)
        size_min[i] = min(su, sv)
        size_max[i] = max(su, sv)

    return np.column_stack([same_comm, size_min, size_max]).astype(np.float32)


def compute_spectral_embedding(G, n_nodes, k=16, seed=SEED):
    """Compute k-dimensional Laplacian eigenmap of the (undirected, weighted)
    augmented graph. Returns an (n_nodes, k) ndarray.
    """
    # Build normalized Laplacian as a scipy sparse matrix
    nodes = list(range(n_nodes))
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr", dtype=np.float64)
    # Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(d)
    nz = d > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L_norm = sparse.eye(n_nodes, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

    # Smallest k+1 eigenvalues (skip the trivial zero eigenvalue)
    # Use shift-invert for numerical stability with sigma=0
    try:
        vals, vecs = eigsh(L_norm, k=k + 1, sigma=0, which="LM")
    except Exception:
        # Fallback: smallest magnitude without shift-invert
        vals, vecs = eigsh(L_norm, k=k + 1, which="SM")
    # Sort by eigenvalue ascending and drop the trivial component
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    emb = vecs[:, 1 : k + 1]  # drop lambda_0 ~ 0

    # Row-normalize: standard Ng-Jordan-Weiss spectral clustering normalization
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + EPS
    emb_normed = emb / norms
    return emb, emb_normed


def compute_spectral_features(pairs, emb, emb_normed):
    """Pair-level spectral features: L2 distance, dot product, cosine similarity."""
    u = pairs[:, 0].astype(np.int64)
    v = pairs[:, 1].astype(np.int64)
    diff = emb[u] - emb[v]
    dist_l2 = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
    dot = np.einsum("ij,ij->i", emb[u], emb[v]).astype(np.float32)
    cos = np.einsum("ij,ij->i", emb_normed[u], emb_normed[v]).astype(np.float32)
    return np.column_stack([dist_l2, dot, cos]).astype(np.float32)


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
    print("[v25] computing pair-level transductive features")
    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v24_test = compute_pair_transductive_v24(test_pairs, train_partners, test_partners)
    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_test = compute_pair_transductive_v25(
        test_pairs, train_partners, test_partners, test_count, train_count, total_count)

    # ============================================================
    # v19 pipeline (self-training + v19 features)
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
    print(f"  self-train: +{len(extra_edges)} pseudo-edges")

    adjacency, degree, comp = build_graph(train_pairs, y_train, n_nodes, extra_edges)
    adj_matrix = build_sparse_adj(adjacency, n_nodes)
    d_inv = np.zeros(n_nodes, dtype=np.float32)
    mask = degree > 0
    d_inv[mask] = 1.0 / degree[mask]
    neighbor_tfidf = sparse.diags(d_inv) @ adj_matrix @ node_tfidf

    X_train_v19 = build_features_v19(
        train_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf, y=y_train, remove_pos=True,
    )
    X_test_v19 = build_features_v19(
        test_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
    )

    X_train_v25 = np.hstack([X_train_v19, pair_v24_train, pair_v25_train])
    X_test_v25 = np.hstack([X_test_v19, pair_v24_test, pair_v25_test])
    print(f"  v25 features: {X_train_v25.shape[1]}")

    # ============================================================
    # v26 NEW: community + spectral features on augmented graph
    # ============================================================
    print("\n[v26] building augmented graph (train+ + pseudo-edges + test candidates)")
    G_aug = build_augmented_graph(train_pairs, y_train, test_pairs, extra_edges, n_nodes)
    print(f"  G: |V|={G_aug.number_of_nodes()}  |E|={G_aug.number_of_edges()}  "
          f"components={nx.number_connected_components(G_aug)}")

    print("[v26] Louvain community detection")
    partition = run_louvain(G_aug, seed=SEED)
    n_comms = len(set(partition.values()))
    print(f"  found {n_comms} communities")
    comm_train = compute_community_features(train_pairs, partition, n_nodes)
    comm_test = compute_community_features(test_pairs, partition, n_nodes)
    print(f"  train same_comm rate: pos={comm_train[y_train==1,0].mean():.3f} "
          f"neg={comm_train[y_train==0,0].mean():.3f}")
    print(f"  test  same_comm rate: {comm_test[:,0].mean():.3f}")

    print("[v26] spectral embedding (Laplacian eigenmap, k=16)")
    emb, emb_normed = compute_spectral_embedding(G_aug, n_nodes, k=16, seed=SEED)
    spec_train = compute_spectral_features(train_pairs, emb, emb_normed)
    spec_test = compute_spectral_features(test_pairs, emb, emb_normed)
    print(f"  train spectral_cos: pos_mean={spec_train[y_train==1,2].mean():.3f} "
          f"neg_mean={spec_train[y_train==0,2].mean():.3f}")
    print(f"  train spectral_dist_l2: pos_mean={spec_train[y_train==1,0].mean():.3f} "
          f"neg_mean={spec_train[y_train==0,0].mean():.3f}")

    X_train_v26 = np.hstack([X_train_v25, comm_train, spec_train])
    X_test_v26 = np.hstack([X_test_v25, comm_test, spec_test])
    print(f"  v26 features: {X_train_v26.shape[1]} (+{X_train_v26.shape[1] - X_train_v25.shape[1]})")

    # ============================================================
    # CV comparison v25 vs v26
    # ============================================================
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v25 = np.zeros(len(y_train), dtype=np.float64)
    oof_v26 = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v25[tr], y_train[tr])
        oof_v25[va] = m1.predict_proba(X_train_v25[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26[tr], y_train[tr])
        oof_v26[va] = m2.predict_proba(X_train_v26[va])[:, 1]
    auc_v25 = roc_auc_score(y_train, oof_v25)
    auc_v26 = roc_auc_score(y_train, oof_v26)
    print(f"  v25 OOF AUC = {auc_v25:.5f}")
    print(f"  v26 OOF AUC = {auc_v26:.5f}  ({auc_v26 - auc_v25:+.5f})")

    # ============================================================
    # Final: 30-seed HGB + CatBoost on v26
    # ============================================================
    print("\n[final] training 30-seed HGB+CatBoost on v26")
    pred_h = predict_hgb(X_train_v26, y_train, X_test_v26, n_seeds=30)
    pred_c = predict_cat(X_train_v26, y_train, X_test_v26, n_seeds=30)
    rh = normalize_rank(pred_h)
    rc = normalize_rank(pred_c)
    pred_v26 = 0.5 * rh + 0.5 * rc
    save_sub("submission_v26.csv", pred_v26)

    # Reference: retrain v25 with the same SEED schedule so the comparison blend
    # is apples-to-apples (versus loading the historical submission_v25.csv)
    print("[ref] training 30-seed HGB+CatBoost on v25 for blend reference")
    pred_h25 = predict_hgb(X_train_v25, y_train, X_test_v25, n_seeds=30)
    pred_c25 = predict_cat(X_train_v25, y_train, X_test_v25, n_seeds=30)
    pred_v25_ref = 0.5 * normalize_rank(pred_h25) + 0.5 * normalize_rank(pred_c25)
    save_sub("submission_v26_v25ref.csv", pred_v25_ref)

    # Blends at multiple weights so we can pick the safest for submission
    save_sub("submission_v26_blend_50.csv", 0.5 * pred_v26 + 0.5 * pred_v25_ref)
    save_sub("submission_v26_blend_70v26.csv", 0.7 * pred_v26 + 0.3 * pred_v25_ref)
    save_sub("submission_v26_blend_30v26.csv", 0.3 * pred_v26 + 0.7 * pred_v25_ref)

    corr, _ = spearmanr(pred_v25_ref, pred_v26)
    print(f"\n  v26 vs v25_ref corr = {corr:.5f}")

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
