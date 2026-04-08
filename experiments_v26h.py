"""
v26h — v26g + consensus Louvain features across 20 seeds.

v26g (0.88080) added a single text-weighted Louvain partition on top of
v26d, gaining +0.00030 Kaggle from 3 new columns. Ablation sanity
checks showed Louvain is extremely noisy across seeds on this graph:
the ARI between partitions at different seeds is ≈ 0.2, meaning each
single-seed Louvain run carves up the same 16-ish communities with
very different boundaries.

Consequence: v26g's `same_community` feature is a binary flag from a
single (essentially arbitrary) partition. If we run Louvain at 20
different seeds and look at the fraction of runs where u and v land
in the same community, we get a continuous "soft co-membership" score
that is strictly more informative than any single run's binary output.
Pairs with consensus > 0.9 are robustly clustered together; pairs with
consensus < 0.1 are robustly apart; the middle is where most of the
ambiguity lives.

Apply the same trick to both the unweighted and text-weighted candidate
graphs, giving two new features on top of v26g:

  consensus_same_comm_unwt : fraction of 20 Louvain seeds on the
                             unweighted candidate graph where u and v
                             are co-clustered
  consensus_same_comm_text : same for the text-weighted candidate graph
                             (edge weights 1.0 + 3.0 * tfidf_cos)

v26h strategy: strict minimal-risk single-family extension of v26g.
Two new columns, both from a mechanism (consensus Louvain) that is
demonstrably orthogonal to the single-seed partitions.

Reference:
  Lancichinetti & Fortunato - Consensus clustering in complex networks.
  Scientific Reports 2 (2012). Formalizes the idea that averaging many
  stochastic community detections gives a more robust partition than
  any single run.
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

try:
    from community import community_louvain
except (ImportError, AttributeError):
    import community as community_louvain
import networkx as nx


def compute_consensus_same_community(pairs, G, n_seeds=20, base_seed=SEED):
    """For each pair, fraction of n_seeds Louvain runs on G in which u and v
    land in the same community.
    """
    n = pairs.shape[0]
    counts = np.zeros(n, dtype=np.float32)

    for s in range(n_seeds):
        partition = community_louvain.best_partition(
            G, weight="weight", random_state=base_seed + s
        )
        for i in range(n):
            u, v = int(pairs[i, 0]), int(pairs[i, 1])
            if partition.get(u, -1) == partition.get(v, -2):
                counts[i] += 1.0

    return counts / float(n_seeds)


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
        total_count, train_count, test_count, A0, ntfidf0,
        y=y_train, remove_pos=True)
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

    # v26d base: v25 + community + spectral + comm_cn on unweighted candidate graph
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

    # v26g base: v26d + text-weighted Louvain community (3 columns)
    print("[v26g] text-weighted candidate graph + Louvain community")
    G_text = build_text_weighted_candidate_graph(
        train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf,
        alpha=1.0, beta=3.0,
    )
    part_text = run_louvain(G_text, seed=SEED)
    comm_text_train = compute_community_features(train_pairs, part_text, n_nodes)
    comm_text_test = compute_community_features(test_pairs, part_text, n_nodes)

    X_train_v26g = np.hstack([X_train_v26d, comm_text_train])
    X_test_v26g = np.hstack([X_test_v26d, comm_text_test])
    print(f"  v26g features: {X_train_v26g.shape[1]}")

    # === v26h NEW: consensus Louvain across 20 seeds ===
    N_SEEDS = 20
    print(f"\n[v26h] consensus Louvain across {N_SEEDS} seeds")

    t1 = time.time()
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=N_SEEDS, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        test_pairs, G_unwt, n_seeds=N_SEEDS, base_seed=SEED).reshape(-1, 1)
    print(f"  unwt consensus: {time.time()-t1:.1f}s")
    print(f"    pos={cons_unwt_train[y_train==1].mean():.3f}  "
          f"neg={cons_unwt_train[y_train==0].mean():.3f}  "
          f"gap={cons_unwt_train[y_train==1].mean() - cons_unwt_train[y_train==0].mean():+.3f}")

    t1 = time.time()
    cons_text_train = compute_consensus_same_community(
        train_pairs, G_text, n_seeds=N_SEEDS, base_seed=SEED).reshape(-1, 1)
    cons_text_test = compute_consensus_same_community(
        test_pairs, G_text, n_seeds=N_SEEDS, base_seed=SEED).reshape(-1, 1)
    print(f"  text consensus: {time.time()-t1:.1f}s")
    print(f"    pos={cons_text_train[y_train==1].mean():.3f}  "
          f"neg={cons_text_train[y_train==0].mean():.3f}  "
          f"gap={cons_text_train[y_train==1].mean() - cons_text_train[y_train==0].mean():+.3f}")

    X_train_v26h = np.hstack([X_train_v26g, cons_unwt_train, cons_text_train])
    X_test_v26h = np.hstack([X_test_v26g, cons_unwt_test, cons_text_test])
    print(f"  v26h features: {X_train_v26h.shape[1]} (+2)")

    # === CV comparison ===
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v26g = np.zeros(len(y_train), dtype=np.float64)
    oof_v26h = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26h, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v26g[tr], y_train[tr])
        oof_v26g[va] = m1.predict_proba(X_train_v26g[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26h[tr], y_train[tr])
        oof_v26h[va] = m2.predict_proba(X_train_v26h[va])[:, 1]
    auc_v26g = roc_auc_score(y_train, oof_v26g)
    auc_v26h = roc_auc_score(y_train, oof_v26h)
    print(f"  v26g OOF = {auc_v26g:.5f}")
    print(f"  v26h OOF = {auc_v26h:.5f}  ({auc_v26h - auc_v26g:+.5f})")

    # Ablations: unwt only vs text only vs both
    print("\n[ablation] per-feature OOF")
    for name, X_tr, X_te in [
        ("+ cons_unwt only", np.hstack([X_train_v26g, cons_unwt_train]),
                              np.hstack([X_test_v26g, cons_unwt_test])),
        ("+ cons_text only", np.hstack([X_train_v26g, cons_text_train]),
                              np.hstack([X_test_v26g, cons_text_test])),
    ]:
        oof_a = np.zeros(len(y_train), dtype=np.float64)
        for fold, (tr, va) in enumerate(cv.split(X_tr, y_train), 1):
            m = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
            m.fit(X_tr[tr], y_train[tr])
            oof_a[va] = m.predict_proba(X_tr[va])[:, 1]
        auc = roc_auc_score(y_train, oof_a)
        print(f"  {name:22s} OOF = {auc:.5f}  ({auc - auc_v26g:+.5f})")

    oof_v26h_cat = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26h, y_train), 1):
        mc = CatBoostClassifier(**CAT_PARAMS, random_seed=SEED + fold)
        mc.fit(X_train_v26h[tr], y_train[tr])
        oof_v26h_cat[va] = mc.predict_proba(X_train_v26h[va])[:, 1]
    print(f"  v26h CatBoost OOF = {roc_auc_score(y_train, oof_v26h_cat):.5f}")

    # Final 30-seed HGB+CatBoost
    print("\n[final] 30-seed HGB+CatBoost on v26h")
    pred_h = predict_hgb(X_train_v26h, y_train, X_test_v26h, n_seeds=30)
    pred_c = predict_cat(X_train_v26h, y_train, X_test_v26h, n_seeds=30)
    pred_v26h = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
    save_sub("submission_v26h.csv", pred_v26h)

    v26g_known = pd.read_csv("submission_v26g.csv")["Predicted"].to_numpy() \
        if Path("submission_v26g.csv").exists() else None
    if v26g_known is not None:
        rg = normalize_rank(v26g_known)
        corr, _ = spearmanr(rg, pred_v26h)
        print(f"\n  v26h vs submission_v26g.csv spearman = {corr:.5f}")
        save_sub("submission_v26h_blend_50v26g.csv", 0.5 * pred_v26h + 0.5 * rg)
        save_sub("submission_v26h_blend_70v26h.csv", 0.7 * pred_v26h + 0.3 * rg)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
