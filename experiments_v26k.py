"""
v26k — candidate-graph neighbor TEXT interaction features.

v19 has three features of the form
  neigh_text_uv = <node_tfidf[v], mean(node_tfidf[w] for w in N_real(u))>
that measure "does node v's text match u's graph neighbors' text?".
Those are computed on the *real* adjacency graph (train positives
+ pseudo-edges).

v26k asks the same question on the *candidate* graph neighborhoods:
for each pair (u, v), we average tfidf(w) over the candidate-graph
neighbors of u and inner-product with tfidf(v), and vice versa.
Candidate-graph neighbors include train pairs (regardless of label),
test pairs, and pseudo-edges — a different and denser neighborhood
than the real adjacency.

This is fundamentally different from v26j's candidate-graph LP
heuristics (CN, AA, RA, Katz): those are pure topology features,
v26k is a text-graph interaction. v26j showed that pure topology on
the candidate graph is fully subsumed by the consensus Louvain
feature, but text-graph interactions are a different signal — the
graph says "who is this actor connected to" and text says "what
keywords does this actor have", and their interaction (do your
graph neighbors look texturally like the query actor) is not
trivially captured by either on its own.

Three candidates, each ablated alone on top of v26h_pure:
  cand_ntext_uv : mean tfidf_cos between u and N_candidate(v)
  cand_ntext_vu : mean tfidf_cos between v and N_candidate(u)
  cand_ntext_nn : mean tfidf_cos between mean(N_candidate(u)) and
                   mean(N_candidate(v))   (two 1-hop aggregates)

Same strict ablation discipline as v26j.
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
from experiments_v26g import build_text_weighted_candidate_graph
from experiments_v26h import compute_consensus_same_community

import networkx as nx


def build_candidate_adj_sparse(G, n_nodes):
    """Build a sparse adjacency matrix (unweighted) from a networkx graph G."""
    rows, cols = [], []
    for u, v in G.edges():
        rows.append(u); cols.append(v)
        rows.append(v); cols.append(u)
    if not rows:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def compute_candidate_neigh_text(pairs, G, n_nodes, node_tfidf):
    """Compute three pair-level features based on candidate-graph
    neighborhoods and TF-IDF node texts.

    Returns (n, 3) float32: [cand_ntext_uv, cand_ntext_vu, cand_ntext_nn]
    """
    A = build_candidate_adj_sparse(G, n_nodes)
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_inv = np.zeros(n_nodes, dtype=np.float32)
    mask = deg > 0
    d_inv[mask] = 1.0 / deg[mask]
    # Mean TF-IDF of each node's candidate-graph neighbors
    neigh_tfidf = sparse.diags(d_inv) @ A @ node_tfidf  # (n_nodes, 932)

    u = pairs[:, 0].astype(np.int64)
    v = pairs[:, 1].astype(np.int64)

    # cand_ntext_uv : inner product between neigh_tfidf[u] and node_tfidf[v]
    uv = np.asarray(
        neigh_tfidf[u].multiply(node_tfidf[v]).sum(axis=1)
    ).ravel().astype(np.float32)

    # cand_ntext_vu : inner product between neigh_tfidf[v] and node_tfidf[u]
    vu = np.asarray(
        neigh_tfidf[v].multiply(node_tfidf[u]).sum(axis=1)
    ).ravel().astype(np.float32)

    # cand_ntext_nn : inner product between neigh_tfidf[u] and neigh_tfidf[v]
    nn = np.asarray(
        neigh_tfidf[u].multiply(neigh_tfidf[v]).sum(axis=1)
    ).ravel().astype(np.float32)

    return np.column_stack([uv, vu, nn])


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

    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v24_test = compute_pair_transductive_v24(test_pairs, train_partners, test_partners)
    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_test = compute_pair_transductive_v25(
        test_pairs, train_partners, test_partners, test_count, train_count, total_count)

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
    emb, emb_normed = compute_spectral_embedding(G_unwt, n_nodes, k=16, seed=SEED)
    spec_train = compute_spectral_features(train_pairs, emb, emb_normed)
    spec_test = compute_spectral_features(test_pairs, emb, emb_normed)
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

    # v26h_pure base: + consensus same_community
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        test_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)

    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])
    print(f"[v26h_pure base] {X_train_v26hp.shape[1]} features")

    # === v26k candidate features: neighbor text on the candidate graph ===
    print("\n[v26k] candidate-graph neighbor text features")
    t1 = time.time()
    cand_ntext_train = compute_candidate_neigh_text(
        train_pairs, G_unwt, n_nodes, node_tfidf)
    cand_ntext_test = compute_candidate_neigh_text(
        test_pairs, G_unwt, n_nodes, node_tfidf)
    print(f"  computed in {time.time()-t1:.1f}s")
    labels = ["cand_ntext_uv", "cand_ntext_vu", "cand_ntext_nn"]
    for j, label in enumerate(labels):
        p = cand_ntext_train[y_train == 1, j].mean()
        n = cand_ntext_train[y_train == 0, j].mean()
        print(f"  {label:15s}: pos={p:.4f}  neg={n:.4f}  gap={p-n:+.4f}")

    # Ablation
    print("\n[ablation] HGB+CatBoost blend OOF per feature")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    hp_h, hp_c, hp_b = blend_oof(X_train_v26hp, y_train, cv, SEED)
    print(f"  v26h_pure base   HGB={hp_h:.5f}  Cat={hp_c:.5f}  blend={hp_b:.5f}")

    deltas = {}
    for j, label in enumerate(labels):
        X_tr = np.hstack([X_train_v26hp, cand_ntext_train[:, j:j+1]])
        h, c, b = blend_oof(X_tr, y_train, cv, SEED)
        delta = b - hp_b
        deltas[label] = (h, c, b, delta)
        flag = "  *" if delta > 0 else ""
        print(f"  +{label:14s}   HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
              f"({delta:+.5f}){flag}")

    # Also try all 3 together
    X_all = np.hstack([X_train_v26hp, cand_ntext_train])
    h, c, b = blend_oof(X_all, y_train, cv, SEED)
    delta = b - hp_b
    flag = "  *" if delta > 0 else ""
    print(f"  +all 3           HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
          f"({delta:+.5f}){flag}")

    # Build winners
    winning = [j for j, label in enumerate(labels) if deltas[label][3] > 0]
    if not winning:
        print("\n[v26k] no single feature strictly improves. Stopping.")
        # Still save v26h_pure retrain as a reference just in case
        return

    print(f"\n[v26k winners] {[labels[j] for j in winning]}")
    X_train_v26k = np.hstack([X_train_v26hp, cand_ntext_train[:, winning]])
    X_test_v26k = np.hstack([X_test_v26hp, cand_ntext_test[:, winning]])

    print("\n[final] 30-seed HGB+CatBoost on v26k winners")
    pred_h = predict_hgb(X_train_v26k, y_train, X_test_v26k, n_seeds=30)
    pred_c = predict_cat(X_train_v26k, y_train, X_test_v26k, n_seeds=30)
    pred_v26k = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)
    save_sub("submission_v26k.csv", pred_v26k)

    if Path("submission_v26h_pure.csv").exists():
        v26hp = pd.read_csv("submission_v26h_pure.csv")["Predicted"].to_numpy()
        rhp = rnk(v26hp)
        corr, _ = spearmanr(rhp, pred_v26k)
        print(f"\n  v26k vs submission_v26h_pure.csv spearman = {corr:.5f}")
        save_sub("submission_v26k_blend_50.csv", 0.5 * pred_v26k + 0.5 * rhp)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
