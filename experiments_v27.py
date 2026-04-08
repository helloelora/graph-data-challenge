"""
v27 — v26b (the 0.88038 winner) rank-blended with TabPFN.

TabPFN was trained on v25 features on Colab A100 (5-fold CV x 5 seeds,
+ 5-seed final test ensemble). We have:
  - tabpfn_oof.npy       : OOF predictions on training set
  - tabpfn_test_pred.npy : test predictions

v26b base is:
  - v25 (56 features) + 6 community/spectral features on label-free candidate
    graph. Local OOF 0.90686, Kaggle 0.88038.

This script:
  1. Rebuilds v26b features + runs 5-fold HGB+CatBoost CV to get OOF preds
     (we didn't save these earlier — have to recompute)
  2. Loads TabPFN OOF + test preds from disk
  3. Sweeps blend weights on OOF to find the optimum
  4. Produces candidate submissions at the best weight + nearby weights
     (Kaggle OOF → leaderboard correlation is noisy here, so we submit a
     few close alternatives)
  5. Trains the final HGB+CatBoost v26b ensemble on full training data
     and rank-blends with TabPFN test preds at the selected weight
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


def rnk(a):
    r = rankdata(a)
    return (r - r.min()) / (r.max() - r.min() + EPS)


def main():
    t0 = time.time()
    np.random.seed(SEED)

    # === Load TabPFN outputs (from Colab) ===
    tabpfn_dir = Path("tabpfn_out/tabpfn_out")
    assert tabpfn_dir.exists(), f"Missing {tabpfn_dir}"
    tabpfn_oof = np.load(tabpfn_dir / "tabpfn_oof.npy")
    tabpfn_test = np.load(tabpfn_dir / "tabpfn_test_pred.npy")
    print(f"[load] TabPFN oof={tabpfn_oof.shape} test={tabpfn_test.shape}")

    # === Rebuild v26b features ===
    print("\n[rebuild] v26b features")
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

    # Community + spectral features
    G_cand = build_candidate_graph(train_pairs, test_pairs, extra_edges, n_nodes)
    partition = run_louvain(G_cand, seed=SEED)
    comm_train = compute_community_features(train_pairs, partition, n_nodes)
    comm_test = compute_community_features(test_pairs, partition, n_nodes)
    emb, emb_normed = compute_spectral_embedding(G_cand, n_nodes, k=16, seed=SEED)
    spec_train = compute_spectral_features(train_pairs, emb, emb_normed)
    spec_test = compute_spectral_features(test_pairs, emb, emb_normed)

    X_train_v26b = np.hstack([X_train_v25, comm_train, spec_train])
    X_test_v26b = np.hstack([X_test_v25, comm_test, spec_test])
    print(f"  v26b features: {X_train_v26b.shape[1]}")

    # === 5-fold OOF with HGB+CatBoost rank blend on v26b ===
    print("\n[oof] 5-fold HGB+CatBoost rank blend on v26b")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_hgb = np.zeros(len(y_train), dtype=np.float64)
    oof_cat = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26b, y_train), 1):
        # 3 seeds each (tradeoff: enough stability, fast enough to iterate)
        for s in range(3):
            m = HistGradientBoostingClassifier(
                **HGB_PARAMS, random_state=SEED + fold * 13 + s * 7)
            m.fit(X_train_v26b[tr], y_train[tr])
            oof_hgb[va] += m.predict_proba(X_train_v26b[va])[:, 1] / 3

            mc = CatBoostClassifier(
                **CAT_PARAMS, random_seed=SEED + fold * 13 + s * 7)
            mc.fit(X_train_v26b[tr], y_train[tr])
            oof_cat[va] += mc.predict_proba(X_train_v26b[va])[:, 1] / 3
        print(f"  fold {fold}: HGB OOF {roc_auc_score(y_train[va], oof_hgb[va]):.5f}  "
              f"Cat OOF {roc_auc_score(y_train[va], oof_cat[va]):.5f}")

    oof_v26b_blend = 0.5 * rnk(oof_hgb) + 0.5 * rnk(oof_cat)
    auc_v26b_blend = roc_auc_score(y_train, oof_v26b_blend)
    auc_tabpfn = roc_auc_score(y_train, tabpfn_oof)
    print(f"\nv26b HGB+Cat rank blend OOF = {auc_v26b_blend:.5f}")
    print(f"TabPFN OOF (from Colab)    = {auc_tabpfn:.5f}")
    spearman = spearmanr(oof_v26b_blend, tabpfn_oof)[0]
    print(f"OOF rank correlation       = {spearman:.5f}")

    # === Sweep blend weights on OOF ===
    print("\n[sweep] blend weight w_tabpfn sweep (OOF)")
    weights = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    best_w, best_auc = 0.0, auc_v26b_blend
    for w in weights:
        blend = (1 - w) * rnk(oof_v26b_blend) + w * rnk(tabpfn_oof)
        auc = roc_auc_score(y_train, blend)
        marker = ""
        if auc > best_auc:
            best_auc = auc
            best_w = w
            marker = "  *"
        print(f"  w_tabpfn={w:.2f}  OOF={auc:.5f}{marker}")
    print(f"\nbest weight = {best_w:.2f}  OOF = {best_auc:.5f}  "
          f"(gain over pure v26b: {best_auc - auc_v26b_blend:+.5f})")

    # === Train final 30-seed v26b on all data, then blend with TabPFN ===
    print("\n[final] 30-seed HGB+CatBoost on full v26b train set")
    pred_h = predict_hgb(X_train_v26b, y_train, X_test_v26b, n_seeds=30)
    pred_c = predict_cat(X_train_v26b, y_train, X_test_v26b, n_seeds=30)
    pred_v26b_final = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)

    r_v26b = rnk(pred_v26b_final)
    r_tab = rnk(tabpfn_test)

    # Save at best weight + several alternatives for safety
    print("\n[save] candidate submissions")
    save_candidates = [best_w]
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        if w not in save_candidates:
            save_candidates.append(w)
    save_candidates = sorted(set(save_candidates))

    for w in save_candidates:
        blend = (1 - w) * r_v26b + w * r_tab
        tag = f"{int(w * 100):02d}"
        fname = f"submission_v27_tab{tag}.csv"
        save_sub(fname, blend)

    # And the "winner" as v27.csv
    best_blend = (1 - best_w) * r_v26b + best_w * r_tab
    save_sub("submission_v27.csv", best_blend)
    print(f"\n  submission_v27.csv = {1-best_w:.2f}*v26b + {best_w:.2f}*TabPFN (OOF best)")

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
