"""
v26Q — low-rank adjacency SVD embeddings on the candidate graph.

v26L extracts information from the candidate graph via community
detection (Louvain, both at a single resolution and across a
multi-resolution consensus) and spectral features (Laplacian
eigenmap). v26Q adds a fundamentally different global factorization:
the TRUNCATED SVD of the raw adjacency matrix itself.

Why this is different from v26b's spectral features:

  Laplacian spectral (v26b): factorize the normalized Laplacian
    I - D^{-1/2} A D^{-1/2}. This is optimized for *soft clustering* —
    nodes in the same cluster get similar eigenvectors. The k smallest
    non-trivial eigenvectors give cluster embeddings.

  Adjacency SVD (v26Q): factorize A = U Σ V^T at low rank k. The left
    and right singular vectors encode *role similarity* — nodes that
    connect to the same set of other nodes get similar embeddings.
    This is not clustering per se; it's a global factorization that
    captures "who-connects-to-whom" regularities regardless of whether
    those regularities form clusters.

The two can give orthogonal signal even on the same graph. Laplacian
says "A and B are in the same community", SVD says "A and B connect
to similar partners even if they're in different communities."

Candidate features (6 total):

  svd4_dot, svd4_l2, svd4_cos : dot, L2, cosine in rank-4 SVD embedding
  svd8_dot, svd8_l2, svd8_cos : dot, L2, cosine in rank-8 SVD embedding

Low rank is critical. Node2Vec / SVD at rank 64+ has been tried and
hurt (v25 "what didn't work" table). Rank 4-8 gives the model 1-3
scalar features total, which is low enough that overfitting is not a
concern — this is the same regime that let v26b's spectral features
work despite all other embedding methods failing.

Each of the 6 candidates is ablated individually on the HGB+CatBoost
rank-blend OOF on top of v26L. Ship only strict positive deltas.
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

import networkx as nx


def candidate_graph_adjacency(G, n_nodes):
    """Return a symmetric sparse adjacency matrix of G (networkx)."""
    rows, cols = [], []
    for u, v in G.edges():
        rows.append(u); cols.append(v)
        rows.append(v); cols.append(u)
    if not rows:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def truncated_svd_embedding(A, k, seed=SEED):
    """Compute a rank-k truncated SVD of A (sparse, symmetric).
    Returns node embeddings as U * sqrt(S) where A = U S V^T.
    """
    # svds returns singular values in ascending order; we want the top k
    # Use a random_state for reproducibility. For symmetric matrices,
    # left and right singular vectors are equal up to sign, so we take U.
    # k must be < min(A.shape), and the matrix should be non-empty
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(A.shape[0]).astype(np.float64)
    U, S, Vt = svds(A.astype(np.float64), k=k, v0=v0)
    # Reverse to descending singular value order
    order = np.argsort(-S)
    U = U[:, order]
    S = S[order]
    Vt = Vt[order]
    # Scale by sqrt(singular values) so dot product of embeddings
    # equals the rank-k approximation of A[u, v].
    emb = U * np.sqrt(S)
    return emb.astype(np.float32)


def svd_pair_features(pairs, emb):
    """Dot product, L2 distance, cosine similarity in the SVD embedding."""
    u = pairs[:, 0].astype(np.int64)
    v = pairs[:, 1].astype(np.int64)
    diff = emb[u] - emb[v]
    dot = np.einsum("ij,ij->i", emb[u], emb[v]).astype(np.float32)
    l2 = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
    norms_u = np.linalg.norm(emb[u], axis=1)
    norms_v = np.linalg.norm(emb[v], axis=1)
    cos = (dot / (norms_u * norms_v + EPS)).astype(np.float32)
    return dot.reshape(-1, 1), l2.reshape(-1, 1), cos.reshape(-1, 1)


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
    print(f"\n[v26L base] {X_train_v26L.shape[1]} features")

    # === v26Q: low-rank SVD of candidate-graph adjacency ===
    print("\n[v26Q] computing truncated SVD of candidate graph adjacency")
    A_cand = candidate_graph_adjacency(G_unwt, n_nodes)
    print(f"  |V|={A_cand.shape[0]}  |E|={A_cand.nnz // 2}")

    candidates_train = {}
    candidates_test = {}
    for k in [4, 8]:
        t1 = time.time()
        emb = truncated_svd_embedding(A_cand, k=k, seed=SEED)
        print(f"  svd k={k}:  emb shape={emb.shape}  built in {time.time()-t1:.1f}s")
        dot_tr, l2_tr, cos_tr = svd_pair_features(train_pairs, emb)
        dot_te, l2_te, cos_te = svd_pair_features(test_pairs, emb)
        for name, tr_arr, te_arr in [
            (f"svd{k}_dot", dot_tr, dot_te),
            (f"svd{k}_l2",  l2_tr,  l2_te),
            (f"svd{k}_cos", cos_tr, cos_te),
        ]:
            candidates_train[name] = tr_arr
            candidates_test[name] = te_arr
            p = tr_arr[y_train == 1].mean()
            n = tr_arr[y_train == 0].mean()
            print(f"    {name:12s}  pos={p:7.3f}  neg={n:7.3f}  gap={p-n:+.3f}")

    # Ablation
    print("\n[ablation] HGB+CatBoost blend OOF per candidate (base = v26L)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_h, base_c, base_b = blend_oof(X_train_v26L, y_train, cv, SEED)
    print(f"  v26L base   HGB={base_h:.5f}  Cat={base_c:.5f}  blend={base_b:.5f}")

    deltas = {}
    for label in candidates_train:
        X_tr = np.hstack([X_train_v26L, candidates_train[label]])
        h, c, b = blend_oof(X_tr, y_train, cv, SEED)
        delta = b - base_b
        deltas[label] = (h, c, b, delta)
        flag = "  *" if delta > 0 else ""
        print(f"  +{label:12s}   HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
              f"({delta:+.5f}){flag}")

    # All 6 together
    X_all = np.hstack([X_train_v26L] + list(candidates_train.values()))
    h, c, b = blend_oof(X_all, y_train, cv, SEED)
    delta = b - base_b
    flag = "  *" if delta > 0 else ""
    print(f"  +all 6 together          HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
          f"({delta:+.5f}){flag}")

    # Group by rank: rank-4 only, rank-8 only
    for prefix in ["svd4", "svd8"]:
        keys = [k for k in candidates_train if k.startswith(prefix)]
        X_tr = np.hstack([X_train_v26L] + [candidates_train[k] for k in keys])
        h, c, b = blend_oof(X_tr, y_train, cv, SEED)
        delta = b - base_b
        flag = "  *" if delta > 0 else ""
        print(f"  +{prefix} all 3           HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
              f"({delta:+.5f}){flag}")

    winning = [label for label in candidates_train if deltas[label][3] > 0]
    if not winning:
        print("\n[v26Q] no single candidate improved the blend. Stopping.")
        return

    print(f"\n[v26Q winners] {winning}")
    X_train_v26Q = np.hstack([X_train_v26L] + [candidates_train[l] for l in winning])
    X_test_v26Q = np.hstack([X_test_v26L] + [candidates_test[l] for l in winning])

    h, c, b = blend_oof(X_train_v26Q, y_train, cv, SEED)
    print(f"  v26Q HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  ({b - base_b:+.5f})")

    print("\n[final] 30-seed HGB+CatBoost on v26Q winners")
    pred_h = predict_hgb(X_train_v26Q, y_train, X_test_v26Q, n_seeds=30)
    pred_c = predict_cat(X_train_v26Q, y_train, X_test_v26Q, n_seeds=30)
    pred_v26Q = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)
    save_sub("submission_v26Q.csv", pred_v26Q)

    if Path("submission_v26L.csv").exists():
        v26L_known = pd.read_csv("submission_v26L.csv")["Predicted"].to_numpy()
        rL = rnk(v26L_known)
        corr, _ = spearmanr(rL, pred_v26Q)
        print(f"\n  v26Q vs submission_v26L.csv spearman = {corr:.5f}")
        save_sub("submission_v26Q_blend_50.csv", 0.5 * pred_v26Q + 0.5 * rL)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
