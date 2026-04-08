"""
v26U — NMF factorization of the candidate adjacency, parallel to v26Q's SVD.

v26Q (svd4_cos) and v26R (svd4_had3) showed that a low-rank SVD of the
candidate-graph adjacency gives pair features that add meaningful
signal on top of v26L. v26S tried to push SVD deeper and hit
saturation: rank 6 and 8 SVD reproduce the same leading axes as rank
4, and alternative graphs (A^2, text-weighted) + alternative
combinators (|u-v|) all regressed.

v26U tries a fundamentally different factorization: **Non-negative
Matrix Factorization** (NMF). Instead of finding orthogonal signed
eigenvectors (SVD), NMF finds a "parts-based" decomposition
A ≈ W H where both W and H are constrained to be non-negative.
This gives two qualitative differences from SVD:

  1. NMF axes represent additive, overlapping cluster-like themes
     (each node is a non-negative sum of "factor activations"),
     rather than orthogonal signed directions
  2. Because the factors are non-negative, NMF is known to be more
     interpretable AND often gives better generalization on sparse
     binary matrices. Adjacency matrices are exactly that case.

The candidate graph adjacency is binary (1 if a pair is asked about
in train/test/pseudo-edges, 0 otherwise), so NMF is mathematically
well-suited.

Candidate features (10 total):

  Rank 4 NMF:
    nmf4_cos              (1)
    nmf4_had0..3          (4)

  Rank 8 NMF:
    nmf8_cos              (1)
    nmf8_had0..7          (8)

Each ablated individually on v26R base (73 features). Greedy forward
selection. Ship only strict positive deltas.

Two risks worth noting up front:

  - Even if NMF gives different embeddings, the downstream HGB+Cat
    may already extract all the relevant adjacency information via
    the v26Q/R svd4_cos and svd4_had3 features. NMF may give
    correlated predictions.

  - NMF is iterative and stochastic. We fix random_state=SEED and
    use multiplicative-update solver (coordinate descent) for
    reproducibility.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import NMF
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
from experiments_v26R import svd_hadamard_features


def fit_nmf_embedding(A, k, seed=SEED):
    """Fit rank-k NMF to the (sparse) adjacency matrix A.
    Returns W (n_nodes x k) as the node embedding."""
    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=500,
        tol=1e-4,
        random_state=seed,
    )
    W = model.fit_transform(A.astype(np.float32))
    return W.astype(np.float32)


def nmf_pair_features(pairs, emb):
    """Same API as svd_pair_features but for NMF embeddings.
    Returns (dot, L2 distance, cosine) as (n, 1) arrays each."""
    u = pairs[:, 0].astype(np.int64)
    v = pairs[:, 1].astype(np.int64)
    diff = emb[u] - emb[v]
    dot = np.einsum("ij,ij->i", emb[u], emb[v]).astype(np.float32)
    l2 = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
    norms_u = np.linalg.norm(emb[u], axis=1)
    norms_v = np.linalg.norm(emb[v], axis=1)
    cos = (dot / (norms_u * norms_v + EPS)).astype(np.float32)
    return dot.reshape(-1, 1), l2.reshape(-1, 1), cos.reshape(-1, 1)


def nmf_hadamard_features(pairs, emb):
    u = pairs[:, 0].astype(np.int64)
    v = pairs[:, 1].astype(np.int64)
    return (emb[u] * emb[v]).astype(np.float32)


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

    # v26d
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

    # v26g
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

    # v26Q
    A_unwt = candidate_graph_adjacency(G_unwt, n_nodes)
    emb_svd4_unwt = truncated_svd_embedding(A_unwt, k=4, seed=SEED)
    _, _, svd4_cos_tr = svd_pair_features(train_pairs, emb_svd4_unwt)
    _, _, svd4_cos_te = svd_pair_features(test_pairs, emb_svd4_unwt)
    X_train_v26Q = np.hstack([X_train_v26L, svd4_cos_tr])
    X_test_v26Q = np.hstack([X_test_v26L, svd4_cos_te])

    # v26R
    emb_svd3_unwt = truncated_svd_embedding(A_unwt, k=3, seed=SEED)
    _, _, svd3_cos_tr = svd_pair_features(train_pairs, emb_svd3_unwt)
    _, _, svd3_cos_te = svd_pair_features(test_pairs, emb_svd3_unwt)
    had4_tr = svd_hadamard_features(train_pairs, emb_svd4_unwt)
    had4_te = svd_hadamard_features(test_pairs, emb_svd4_unwt)
    svd4_had3_tr = had4_tr[:, 3:4]
    svd4_had3_te = had4_te[:, 3:4]
    X_train_v26R = np.hstack([X_train_v26Q, svd3_cos_tr, svd4_had3_tr])
    X_test_v26R = np.hstack([X_test_v26Q, svd3_cos_te, svd4_had3_te])
    print(f"\n[v26R base] {X_train_v26R.shape[1]} features")

    # === v26U: NMF factorization ===
    print("\n[v26U] NMF factorization of the candidate graph adjacency")

    candidates_train = {}
    candidates_test = {}

    for k in [4, 8]:
        t1 = time.time()
        emb_nmf = fit_nmf_embedding(A_unwt, k=k, seed=SEED)
        nmf_time = time.time() - t1
        print(f"  NMF rank {k}: shape {emb_nmf.shape}  "
              f"nonzero per row mean {(emb_nmf > 0).sum(axis=1).mean():.2f}  "
              f"({nmf_time:.1f}s)")

        # Cosine
        _, _, cos_tr = nmf_pair_features(train_pairs, emb_nmf)
        _, _, cos_te = nmf_pair_features(test_pairs, emb_nmf)
        label = f"nmf{k}_cos"
        candidates_train[label] = cos_tr
        candidates_test[label] = cos_te
        p = cos_tr[y_train == 1].mean()
        n = cos_tr[y_train == 0].mean()
        print(f"    {label}  pos={p:.3f}  neg={n:.3f}  gap={p-n:+.3f}")

        # Hadamard (element-wise products)
        had_tr = nmf_hadamard_features(train_pairs, emb_nmf)
        had_te = nmf_hadamard_features(test_pairs, emb_nmf)
        for i in range(k):
            label = f"nmf{k}_had{i}"
            candidates_train[label] = had_tr[:, i:i+1]
            candidates_test[label] = had_te[:, i:i+1]
            p = had_tr[y_train == 1, i].mean()
            n = had_tr[y_train == 0, i].mean()
            print(f"    {label}  pos={p:.4f}  neg={n:.4f}  gap={p-n:+.4f}")

    # === Ablation ===
    print("\n[ablation] HGB+CatBoost blend OOF per candidate (base = v26R)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_h, base_c, base_b = blend_oof(X_train_v26R, y_train, cv, SEED)
    print(f"  v26R base   HGB={base_h:.5f}  Cat={base_c:.5f}  blend={base_b:.5f}")

    deltas = {}
    for label in candidates_train:
        X_tr = np.hstack([X_train_v26R, candidates_train[label]])
        h, c, b = blend_oof(X_tr, y_train, cv, SEED)
        delta = b - base_b
        deltas[label] = (h, c, b, delta)
        flag = "  *" if delta > 0 else ""
        print(f"  +{label:12s}   HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
              f"({delta:+.5f}){flag}")

    winning_sorted = sorted(
        [l for l in candidates_train if deltas[l][3] > 0],
        key=lambda l: -deltas[l][3],
    )
    if not winning_sorted:
        print("\n[v26U] no single candidate improved the blend. Stopping.")
        return
    print(f"\n[v26U individual winners sorted] {winning_sorted}")

    # Greedy forward selection
    print("\n[greedy forward selection]")
    cur_X_tr = X_train_v26R.copy()
    cur_h, cur_c, cur_b = blend_oof(cur_X_tr, y_train, cv, SEED)
    print(f"  start blend={cur_b:.5f}")
    selected = []
    for label in winning_sorted:
        candidate_tr = np.hstack([cur_X_tr, candidates_train[label]])
        h, c, b = blend_oof(candidate_tr, y_train, cv, SEED)
        if b > cur_b:
            print(f"  KEEP  +{label:12s}   blend={b:.5f}  ({b-cur_b:+.5f})")
            cur_X_tr = candidate_tr
            cur_b = b
            selected.append(label)
        else:
            print(f"  skip  +{label:12s}   blend={b:.5f}  ({b-cur_b:+.5f})")

    if not selected:
        print("\n[v26U] greedy kept nothing. Stopping.")
        return

    print(f"\n[v26U greedy final] {selected}")
    X_train_v26U = np.hstack([X_train_v26R] + [candidates_train[l] for l in selected])
    X_test_v26U = np.hstack([X_test_v26R] + [candidates_test[l] for l in selected])

    h, c, b = blend_oof(X_train_v26U, y_train, cv, SEED)
    print(f"  v26U HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  ({b - base_b:+.5f})")

    print("\n[final] 30-seed HGB+CatBoost on v26U winners")
    pred_h = predict_hgb(X_train_v26U, y_train, X_test_v26U, n_seeds=30)
    pred_c = predict_cat(X_train_v26U, y_train, X_test_v26U, n_seeds=30)
    pred_v26U = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)
    save_sub("submission_v26U.csv", pred_v26U)

    if Path("submission_v26R.csv").exists():
        v26R_known = pd.read_csv("submission_v26R.csv")["Predicted"].to_numpy()
        rR = rnk(v26R_known)
        corr, _ = spearmanr(rR, pred_v26U)
        print(f"\n  v26U vs submission_v26R.csv spearman = {corr:.5f}")
        save_sub("submission_v26U_blend_50.csv", 0.5 * pred_v26U + 0.5 * rR)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
