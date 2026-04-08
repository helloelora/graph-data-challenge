"""
v26h_pure — v26g + ONLY unweighted consensus Louvain (single new feature).

v26h (in experiments_v26h.py) added two consensus features:

  cons_unwt : fraction of 20 Louvain seeds on the unweighted candidate
              graph where u and v are co-clustered
  cons_text : same on the text-weighted candidate graph

OOF ablation (5-fold HGB single seed):

  v26g              = 0.90794
  + cons_unwt only  = 0.90936  (+0.00142)
  + cons_text only  = 0.90752  (-0.00042)  ← HURTS alone
  + both (v26h)     = 0.90945  (+0.00152)

Same pattern as v26c (marginal int_dens columns that flipped Kaggle
negative) and v26f (text_spectral that hurt alone and dragged the
bundle down). The +0.00010 OOF difference between "both" and
"cons_unwt only" is within CV noise, and the cons_text ablation shows
clear negative signal.

Minimal-risk recipe: v26g + cons_unwt only. One new column, strong
OOF gain, no hurting features bundled in.
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
    print(f"v26g features: {X_train_v26g.shape[1]}")

    # === v26h_pure: v26g + consensus Louvain on UNWEIGHTED graph ONLY ===
    print("\n[v26h_pure] consensus Louvain across 20 seeds (unweighted graph)")
    N_SEEDS = 20
    t1 = time.time()
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=N_SEEDS, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        test_pairs, G_unwt, n_seeds=N_SEEDS, base_seed=SEED).reshape(-1, 1)
    print(f"  consensus compute time: {time.time()-t1:.1f}s")
    p_mean = cons_unwt_train[y_train == 1].mean()
    n_mean = cons_unwt_train[y_train == 0].mean()
    print(f"  cons_unwt: pos={p_mean:.3f}  neg={n_mean:.3f}  gap={p_mean-n_mean:+.3f}")

    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])
    print(f"  v26h_pure features: {X_train_v26hp.shape[1]} (+1)")

    # CV
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v26g = np.zeros(len(y_train), dtype=np.float64)
    oof_v26hp = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26hp, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v26g[tr], y_train[tr])
        oof_v26g[va] = m1.predict_proba(X_train_v26g[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26hp[tr], y_train[tr])
        oof_v26hp[va] = m2.predict_proba(X_train_v26hp[va])[:, 1]
    auc_v26g = roc_auc_score(y_train, oof_v26g)
    auc_v26hp = roc_auc_score(y_train, oof_v26hp)
    print(f"  v26g      OOF = {auc_v26g:.5f}")
    print(f"  v26h_pure OOF = {auc_v26hp:.5f}  ({auc_v26hp - auc_v26g:+.5f})")

    oof_v26hp_cat = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26hp, y_train), 1):
        mc = CatBoostClassifier(**CAT_PARAMS, random_seed=SEED + fold)
        mc.fit(X_train_v26hp[tr], y_train[tr])
        oof_v26hp_cat[va] = mc.predict_proba(X_train_v26hp[va])[:, 1]
    print(f"  v26h_pure CatBoost OOF = {roc_auc_score(y_train, oof_v26hp_cat):.5f}")

    # Final 30-seed
    print("\n[final] 30-seed HGB+CatBoost on v26h_pure")
    pred_h = predict_hgb(X_train_v26hp, y_train, X_test_v26hp, n_seeds=30)
    pred_c = predict_cat(X_train_v26hp, y_train, X_test_v26hp, n_seeds=30)
    pred_v26hp = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
    save_sub("submission_v26h_pure.csv", pred_v26hp)

    if Path("submission_v26g.csv").exists():
        v26g_known = pd.read_csv("submission_v26g.csv")["Predicted"].to_numpy()
        rg = normalize_rank(v26g_known)
        corr, _ = spearmanr(rg, pred_v26hp)
        print(f"\n  v26h_pure vs submission_v26g.csv spearman = {corr:.5f}")
        save_sub("submission_v26h_pure_blend_50v26g.csv", 0.5 * pred_v26hp + 0.5 * rg)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
