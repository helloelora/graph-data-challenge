"""
v26N — even more extreme resolutions on the unweighted candidate graph.

v26L's multi-resolution consensus Louvain scored 0.88491 Kaggle with
winners at resolutions {0.7, 1.3, 2.0}. The pattern was bimodal: the
two extremes (0.7 and 2.0) contributed the most, the middle values
(1.0, 1.6) either duplicated the base or were noise.

v26M tried to push the same idea on weighted graphs (text-weighted and
test-partner-weighted) and all six candidates regressed. The weighting
schemes collapse the community granularity spectrum and produce
partitions that are correlated with features the model already has.

v26N pushes the OTHER lever — deeper into the unweighted graph's
resolution extremes. The v26L resolutions {0.7, 1.3, 2.0} were all
within 2x of the default 1.0. This file tests values further out in
both directions:

  res=0.3  (very fine — python-louvain's "low resolution" gives many
            micro-clusters because the null-model term dominates)
  res=0.5  (still very fine — v26c showed this gives ~1206 communities)
  res=3.0  (double the highest v26L value — likely ~80-120 communities)
  res=4.0  (more extreme still — likely ~130-180 communities)

Each is ablated individually on the HGB+CatBoost rank-blend OOF on
top of v26L (the current shipping best). Strict discipline, ship only
strictly positive deltas.

Hypothesis: the further a resolution is from 1.0 in either direction,
the more its partition captures structure orthogonal to the base. If
this holds, 0.3/0.5/3.0/4.0 should give positive deltas on par with
or better than v26L's 0.7/2.0.

Risk: at extreme resolutions the partitions may become degenerate
(near-singletons or near-connected-components), in which case the
feature collapses to near-constant and ablation catches it. No
commitment until the ablation speaks.
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
from experiments_v26L import compute_consensus_at_resolution


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

    # v26h_pure base: + cons_unwt
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        test_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])

    # v26L base: + multi-res consensus winners
    print("\n[v26L base] multi-resolution consensus winners")
    for r in [0.7, 1.3, 2.0]:
        tr_arr, _ = compute_consensus_at_resolution(
            train_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        te_arr, _ = compute_consensus_at_resolution(
            test_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        X_train_v26hp = np.hstack([X_train_v26hp, tr_arr.reshape(-1, 1)])
        X_test_v26hp = np.hstack([X_test_v26hp, te_arr.reshape(-1, 1)])
    X_train_v26L = X_train_v26hp
    X_test_v26L = X_test_v26hp
    print(f"  v26L base: {X_train_v26L.shape[1]} features")

    # === v26N: more extreme resolutions on unweighted graph ===
    extreme_resolutions = [0.3, 0.5, 3.0, 4.0]
    print(f"\n[v26N] running consensus at {len(extreme_resolutions)} more extreme resolutions")

    candidates_train = {}
    candidates_test = {}
    for r in extreme_resolutions:
        label = f"cons_res{int(r*10):02d}"
        t1 = time.time()
        tr_arr, n_comms_tr = compute_consensus_at_resolution(
            train_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        te_arr, _ = compute_consensus_at_resolution(
            test_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        candidates_train[label] = tr_arr.reshape(-1, 1)
        candidates_test[label] = te_arr.reshape(-1, 1)
        n_comms_avg = np.mean(n_comms_tr)
        p = tr_arr[y_train == 1].mean()
        n = tr_arr[y_train == 0].mean()
        print(f"  {label:12s}  (res={r:.1f}, ~{n_comms_avg:.0f} communities)  "
              f"pos={p:.3f}  neg={n:.3f}  gap={p-n:+.3f}  ({time.time()-t1:.1f}s)")

    # Ablation on HGB+CatBoost blend OOF
    print("\n[ablation] blend OOF per candidate (base = v26L)")
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

    # Try all 4 at once
    X_all = np.hstack([X_train_v26L] + list(candidates_train.values()))
    h, c, b = blend_oof(X_all, y_train, cv, SEED)
    delta = b - base_b
    flag = "  *" if delta > 0 else ""
    print(f"  +all 4 together         HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
          f"({delta:+.5f}){flag}")

    winning = [label for label in candidates_train if deltas[label][3] > 0]
    if not winning:
        print("\n[v26N] no feature strictly improved. Stopping.")
        return

    print(f"\n[v26N winners only] {winning}")
    X_train_v26N = np.hstack([X_train_v26L] + [candidates_train[l] for l in winning])
    X_test_v26N = np.hstack([X_test_v26L] + [candidates_test[l] for l in winning])

    h, c, b = blend_oof(X_train_v26N, y_train, cv, SEED)
    print(f"  v26N HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  ({b - base_b:+.5f})")

    # Final ensemble
    print("\n[final] 30-seed HGB+CatBoost on v26N winners")
    pred_h = predict_hgb(X_train_v26N, y_train, X_test_v26N, n_seeds=30)
    pred_c = predict_cat(X_train_v26N, y_train, X_test_v26N, n_seeds=30)
    pred_v26N = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)
    save_sub("submission_v26N.csv", pred_v26N)

    if Path("submission_v26L.csv").exists():
        v26L_known = pd.read_csv("submission_v26L.csv")["Predicted"].to_numpy()
        rL = rnk(v26L_known)
        corr, _ = spearmanr(rL, pred_v26N)
        print(f"\n  v26N vs submission_v26L.csv spearman = {corr:.5f}")
        save_sub("submission_v26N_blend_50.csv", 0.5 * pred_v26N + 0.5 * rL)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
