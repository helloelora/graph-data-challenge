"""
v26g — v26d + text-weighted Louvain community features. Kaggle public: 0.88080 (1st place).

Previous best: v26d at 0.88050, built on v26b at 0.88038, both relying on
an *unweighted* label-free candidate graph clustered by Louvain. v26g
introduces a second Louvain partition on a *text-weighted* variant of the
same graph and adds its community features to the v26d feature set.

Why text weighting:
  v26d and its variants saturated near 0.88050 because every partition
  algorithm (Louvain, Leiden, different resolutions) kept carving up the
  unweighted candidate graph into essentially the same 16 clusters. The
  only way to get a structurally different partition was to change the
  *graph itself*, not the clustering algorithm.

  By weighting each candidate edge by its endpoints' TF-IDF cosine
  similarity, Louvain is pulled toward communities of actors with
  similar Wikipedia keywords, not just communities of actors that
  co-occur in the train/test pair list. ARI between the two partitions
  was 0.063 (ARI=1 means identical) — genuinely different structure.

  Ablation confirmed text community features gave the biggest
  single-family OOF jump of the session:
    +text community (same_comm + size_min + size_max) : +0.00073
    +text comm_cn                                      : +0.00003  (redundant)
    +text spectral                                     : -0.00065  (hurts)

  So v26g adds *only* the three text community columns on top of v26d.

Reference graphs (both built from the same label-free candidate edge list):

  Unweighted candidate graph (v26b):
    - All train pairs regardless of label                    w = 1.0
    - All test pairs as candidate edges                      w = 1.0
    - Self-training pseudo-edges (v25 predictions >= 0.95)   w = 1.0

  Text-weighted candidate graph (v26g):
    Same edge set but w = alpha + beta * tfidf_cosine(u, v)
    with alpha=1.0, beta=3.0 — the base 1.0 keeps Louvain sensitive to
    the original candidate-graph structure while the text term biases
    clustering toward text-similar neighbors.

Feature count progression:
  v26b : 62 features (v25 56 + 3 community + 3 spectral)
  v26d : 63 features (+ comm_cn)
  v26g : 66 features (+ 3 text community columns from the text-weighted
         Louvain partition)

Kaggle public leaderboard progression:
  v25  : 0.87208  (previous 1st)
  v26b : 0.88038  (+0.00830)
  v26d : 0.88050  (+0.00012)
  v26g : 0.88080  (+0.00030)  *current 1st*
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


def build_text_weighted_candidate_graph(
    train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf,
    alpha=1.0, beta=3.0,
):
    """Label-free candidate graph with edge weights biased by text similarity.

    Each candidate edge (u, v) gets weight `alpha + beta * tfidf_cos(u, v)`,
    where tfidf_cos is the dot product of the already-L2-normalized TF-IDF
    rows for u and v (so it lies in [0, 1] for non-negative features).

    - alpha keeps every candidate edge meaningfully present so Louvain
      still sees the raw candidate-graph topology,
    - beta amplifies the pull between text-similar endpoints,
    - the result is a partition of the same edge set that emphasizes
      text-coherent communities alongside transductive co-occurrence.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    def tfidf_cos(u, v):
        a = node_tfidf[u]
        b = node_tfidf[v]
        sim = float((a.multiply(b)).sum())
        return max(min(sim, 1.0), 0.0)

    for u, v in train_pairs:
        u, v = int(u), int(v)
        if u != v:
            G.add_edge(u, v, weight=alpha + beta * tfidf_cos(u, v))

    for u, v in test_pairs:
        u, v = int(u), int(v)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, weight=alpha + beta * tfidf_cos(u, v))

    if extra_edges is not None:
        for u, v in extra_edges:
            u, v = int(u), int(v)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v, weight=alpha + beta * tfidf_cos(u, v))

    return G


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

    # v19 / v25 transductive counts
    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in train_pairs:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in test_pairs:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    # Pair-level transductive features (v24 + v25)
    print("[v25] pair-level transductive features")
    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v24_test = compute_pair_transductive_v24(test_pairs, train_partners, test_partners)
    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_test = compute_pair_transductive_v25(
        test_pairs, train_partners, test_partners, test_count, train_count, total_count)

    # v19 pipeline with self-training (1 round at 0.95)
    print("\n[v19] self-training (st=0.95)")
    adj0, deg0, comp0 = build_graph(train_pairs, y_train, n_nodes)
    A0 = build_sparse_adj(adj0, n_nodes)
    d_inv0 = np.zeros(n_nodes, dtype=np.float32)
    d_inv0[deg0 > 0] = 1.0 / deg0[deg0 > 0]
    ntfidf0 = sparse.diags(d_inv0) @ A0 @ node_tfidf

    X_tr0 = build_features_v19(
        train_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count, A0, ntfidf0,
        y=y_train, remove_pos=True)
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

    # v26d base = v25 + community + spectral + comm_cn on unweighted candidate graph
    print("\n[v26d] unweighted candidate graph + Louvain + spectral + comm_cn")
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
    print(f"  v26d features: {X_train_v26d.shape[1]}")

    # v26g new: text-weighted Louvain community features (3 columns)
    print("\n[v26g] text-weighted candidate graph -> Louvain -> community features")
    G_text = build_text_weighted_candidate_graph(
        train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf,
        alpha=1.0, beta=3.0,
    )
    part_text = run_louvain(G_text, seed=SEED)
    n_text_comms = len(set(part_text.values()))
    print(f"  {n_text_comms} text-weighted communities")

    from sklearn.metrics import adjusted_rand_score
    nodes_list = sorted(part_unwt.keys())
    u_lab = [part_unwt[n] for n in nodes_list]
    t_lab = [part_text[n] for n in nodes_list]
    ari = adjusted_rand_score(u_lab, t_lab)
    print(f"  ARI(unweighted, text-weighted) = {ari:.4f}  "
          f"(lower = more orthogonal signal)")

    comm_text_train = compute_community_features(train_pairs, part_text, n_nodes)
    comm_text_test = compute_community_features(test_pairs, part_text, n_nodes)
    for j, label in enumerate(["text_same_comm", "text_size_min", "text_size_max"]):
        p_mean = comm_text_train[y_train == 1, j].mean()
        n_mean = comm_text_train[y_train == 0, j].mean()
        print(f"  {label:15s}: pos={p_mean:.3f}  neg={n_mean:.3f}  gap={p_mean-n_mean:+.3f}")

    X_train_v26g = np.hstack([X_train_v26d, comm_text_train])
    X_test_v26g = np.hstack([X_test_v26d, comm_text_test])
    print(f"  v26g features: {X_train_v26g.shape[1]} (+3)")

    # 5-fold OOF comparison v26d vs v26g (HGB only, single seed)
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v26d = np.zeros(len(y_train), dtype=np.float64)
    oof_v26g = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26g, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v26d[tr], y_train[tr])
        oof_v26d[va] = m1.predict_proba(X_train_v26d[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26g[tr], y_train[tr])
        oof_v26g[va] = m2.predict_proba(X_train_v26g[va])[:, 1]
    auc_v26d = roc_auc_score(y_train, oof_v26d)
    auc_v26g = roc_auc_score(y_train, oof_v26g)
    print(f"  v26d OOF = {auc_v26d:.5f}")
    print(f"  v26g OOF = {auc_v26g:.5f}  ({auc_v26g - auc_v26d:+.5f})")

    oof_v26g_cat = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26g, y_train), 1):
        mc = CatBoostClassifier(**CAT_PARAMS, random_seed=SEED + fold)
        mc.fit(X_train_v26g[tr], y_train[tr])
        oof_v26g_cat[va] = mc.predict_proba(X_train_v26g[va])[:, 1]
    print(f"  v26g CatBoost OOF = {roc_auc_score(y_train, oof_v26g_cat):.5f}")

    # Final 30-seed HGB + CatBoost rank blend
    print("\n[final] 30-seed HGB+CatBoost on v26g")
    pred_h = predict_hgb(X_train_v26g, y_train, X_test_v26g, n_seeds=30)
    pred_c = predict_cat(X_train_v26g, y_train, X_test_v26g, n_seeds=30)
    pred_v26g = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
    save_sub("submission_v26g.csv", pred_v26g)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
