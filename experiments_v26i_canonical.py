"""
v26i_canonical — v26h_pure + Lancichinetti canonical partition community features.

v26i ablation tested seven consensus-derived candidates on top of
v26h_pure (OOF 0.90936). Clear winner: the Lancichinetti canonical
partition (Louvain run on the consensus matrix of the 20 unweighted
Louvain seeds, threshold 0.5) gave +0.00071 OOF from three columns:

  canon_same_community, canon_community_size_min, canon_community_size_max

Losers (hurt or dead on top of v26h_pure):
  - cons_comm_cn         : +0.00000  (count feature already has low per-seed variance)
  - node_entropy(min,max): -0.00046  (model handled the noise via cons_unwt already)
  - canonical comm_cn    : -0.00027  (consensus threshold 0.5 drops too many neighbors)

Also marginal:
  - cons_size_min        : +0.00032  alone
  - cons_size_max        : +0.00015  alone
  - cons_sizes both      : +0.00017  (sub-additive, suggests size_max
                                       cancels some size_min signal)

v26i_canonical keeps only the clear winner:
  v26h_pure (67 features) + canonical same_community (3 cols) = 70 features

Canonical partition has 30 communities vs. 16 from a single-seed
Louvain run — the extra granularity comes from the consensus matrix
filter at threshold 0.5, which only keeps node pairs that co-cluster
in at least 10/20 seeds. Its same_community stats:

  pos = 0.682   neg = 0.583   gap = +0.099

which is a slightly larger gap than the single-seed v26g same_community
(pos 0.560 / neg 0.460 / gap +0.100) — roughly the same strength, but
from a fundamentally different (deterministic, stable) source.

Reference:
  Lancichinetti & Fortunato - Consensus clustering in complex networks.
  Scientific Reports 2 (2012).
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
from experiments_v26i import (
    compute_all_partitions,
    build_canonical_partition,
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
    print(f"self-train: +{len(extra_edges)} pseudo-edges")

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
        train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf,
        alpha=1.0, beta=3.0,
    )
    part_text = run_louvain(G_text, seed=SEED)
    comm_text_train = compute_community_features(train_pairs, part_text, n_nodes)
    comm_text_test = compute_community_features(test_pairs, part_text, n_nodes)

    X_train_v26g = np.hstack([X_train_v26d, comm_text_train])
    X_test_v26g = np.hstack([X_test_v26d, comm_text_test])

    # v26h_pure: add consensus Louvain across 20 seeds (unweighted graph)
    print("\n[v26h_pure] consensus Louvain across 20 seeds")
    N_SEEDS = 20
    partitions_unwt = compute_all_partitions(G_unwt, n_seeds=N_SEEDS, base_seed=SEED)
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=N_SEEDS, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        test_pairs, G_unwt, n_seeds=N_SEEDS, base_seed=SEED).reshape(-1, 1)

    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])
    print(f"  v26h_pure features: {X_train_v26hp.shape[1]}")

    # === v26i_canonical NEW: add Lancichinetti canonical partition features ===
    print("\n[v26i_canonical] building canonical consensus partition")
    canonical = build_canonical_partition(
        partitions_unwt, n_nodes, threshold=0.5, seed=SEED
    )
    n_canonical = len(set(canonical.values()))
    print(f"  {n_canonical} canonical communities")

    canon_comm_train = compute_community_features(train_pairs, canonical, n_nodes)
    canon_comm_test = compute_community_features(test_pairs, canonical, n_nodes)
    print(f"  canon same_comm: "
          f"pos={canon_comm_train[y_train==1, 0].mean():.3f}  "
          f"neg={canon_comm_train[y_train==0, 0].mean():.3f}  "
          f"gap={canon_comm_train[y_train==1, 0].mean() - canon_comm_train[y_train==0, 0].mean():+.3f}")

    X_train_v26ic = np.hstack([X_train_v26hp, canon_comm_train])
    X_test_v26ic = np.hstack([X_test_v26hp, canon_comm_test])
    print(f"  v26i_canonical features: {X_train_v26ic.shape[1]} (+3)")

    # CV comparison
    print("\n[CV] OOF AUC")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v26hp = np.zeros(len(y_train), dtype=np.float64)
    oof_v26ic = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26ic, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v26hp[tr], y_train[tr])
        oof_v26hp[va] = m1.predict_proba(X_train_v26hp[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26ic[tr], y_train[tr])
        oof_v26ic[va] = m2.predict_proba(X_train_v26ic[va])[:, 1]
    auc_hp = roc_auc_score(y_train, oof_v26hp)
    auc_ic = roc_auc_score(y_train, oof_v26ic)
    print(f"  v26h_pure      HGB OOF = {auc_hp:.5f}")
    print(f"  v26i_canonical HGB OOF = {auc_ic:.5f}  ({auc_ic - auc_hp:+.5f})")

    oof_v26ic_cat = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26ic, y_train), 1):
        mc = CatBoostClassifier(**CAT_PARAMS, random_seed=SEED + fold)
        mc.fit(X_train_v26ic[tr], y_train[tr])
        oof_v26ic_cat[va] = mc.predict_proba(X_train_v26ic[va])[:, 1]
    print(f"  v26i_canonical CatBoost OOF = {roc_auc_score(y_train, oof_v26ic_cat):.5f}")

    # Final 30-seed ensemble
    print("\n[final] 30-seed HGB+CatBoost on v26i_canonical")
    pred_h = predict_hgb(X_train_v26ic, y_train, X_test_v26ic, n_seeds=30)
    pred_c = predict_cat(X_train_v26ic, y_train, X_test_v26ic, n_seeds=30)
    pred_v26ic = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
    save_sub("submission_v26i_canonical.csv", pred_v26ic)

    if Path("submission_v26h_pure.csv").exists():
        v26hp = pd.read_csv("submission_v26h_pure.csv")["Predicted"].to_numpy()
        rhp = normalize_rank(v26hp)
        corr, _ = spearmanr(rhp, pred_v26ic)
        print(f"\n  v26i_canonical vs v26h_pure spearman = {corr:.5f}")
        save_sub("submission_v26i_canonical_blend_50.csv", 0.5 * pred_v26ic + 0.5 * rhp)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
