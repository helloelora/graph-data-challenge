"""
v26P — multi-round self-training with v26L as the base model.

Current v26L pipeline:
  1. Train 5-seed HGB on initial v19 features (no pseudo-edges yet)
  2. Take test pairs with prediction >= 0.95 → ~332 pseudo-edges
  3. Rebuild real adjacency WITH these pseudo-edges
  4. Compute all v19/v25/v26b/g/h_pure/L features on the enriched adjacency
  5. Train 30-seed HGB+CatBoost ensemble → final predictions

The round-1 pseudo-edges come from a WEAK model (5-seed HGB on v19 features
alone, without any community/consensus signal). v26L itself is much
stronger: blend OOF ~0.9116, single-seed Cat ~0.911. Using v26L's own
predictions to redefine the pseudo-edge set should give a cleaner, and
potentially denser, pseudo-edge set — and feeding that back into the
feature computation pipeline is a classic multi-round self-training loop.

v26P does exactly one additional round:

  Round 1 (same as v26L):
    extra_edges_r1 = v19-warmup predictions >= 0.95
    features_r1 = build v26L features with extra_edges_r1
    pred_r1 = 30-seed HGB+CatBoost on features_r1

  Round 2 (new):
    extra_edges_r2 = extra_edges_r1 ∪ { test pair with pred_r1 >= 0.97 }
    features_r2 = build v26L features with extra_edges_r2 (denser candidate graph)
    pred_r2 = 30-seed HGB+CatBoost on features_r2

We report the HGB+CatBoost blend OOF on each round and keep round 2 only
if it strictly improves on round 1.

Why this might help on v26L specifically: v19's "dual self-training" trick
was tried earlier (v21) and HURT. But that was without the consensus
Louvain feature. With v26L's denoised features, the round-2 pseudo-edge
set should be much cleaner (v26L's 0.97-threshold precision is higher
than the 5-seed warmup's 0.95-threshold precision), and the enriched
candidate graph could give the consensus Louvain a cleaner input without
inducing the errors that hurt v21.

Risk: multi-round self-training can overfit to the model's own biases.
We use threshold=0.97 (more conservative than round-1's 0.95) as a guard
and keep only the union, not a replacement, so we never shrink the
pseudo-edge set.
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


def build_v26L_feature_matrices(
    train_pairs, y_train, test_pairs, node_features, node_tfidf,
    extra_edges,
    pair_v24_train, pair_v24_test,
    pair_v25_train, pair_v25_test,
    total_count, train_count, test_count,
    n_nodes,
):
    """Build the full 70-feature v26L matrices for a given `extra_edges`
    pseudo-edge set. Pair-transductive features (v24/v25) don't depend
    on extra_edges, so they're passed in precomputed.
    """
    # Real adjacency with pseudo-edges
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

    # v26b community + spectral + comm_cn on unweighted candidate graph
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

    # v26g text-weighted community
    G_text = build_text_weighted_candidate_graph(
        train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf, alpha=1.0, beta=3.0)
    part_text = run_louvain(G_text, seed=SEED)
    comm_text_train = compute_community_features(train_pairs, part_text, n_nodes)
    comm_text_test = compute_community_features(test_pairs, part_text, n_nodes)

    X_train_v26g = np.hstack([X_train_v26d, comm_text_train])
    X_test_v26g = np.hstack([X_test_v26d, comm_text_test])

    # v26h_pure: 20-seed consensus at res 1.0
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        test_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)

    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])

    # v26L: multi-res consensus winners (res 0.7, 1.3, 2.0)
    for r in [0.7, 1.3, 2.0]:
        tr_arr, _ = compute_consensus_at_resolution(
            train_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        te_arr, _ = compute_consensus_at_resolution(
            test_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        X_train_v26hp = np.hstack([X_train_v26hp, tr_arr.reshape(-1, 1)])
        X_test_v26hp = np.hstack([X_test_v26hp, te_arr.reshape(-1, 1)])

    return X_train_v26hp, X_test_v26hp


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

    # Transductive counts (don't depend on extra_edges)
    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in train_pairs:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in test_pairs:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    # Pair transductive (don't depend on extra_edges)
    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v24_test = compute_pair_transductive_v24(test_pairs, train_partners, test_partners)
    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_test = compute_pair_transductive_v25(
        test_pairs, train_partners, test_partners, test_count, train_count, total_count)

    # === Round 1: v19 warmup -> initial pseudo-edges ===
    print("[round 1] v19 warmup self-training")
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
    extra_edges_r1 = test_pairs[pred_init >= 0.95]
    print(f"  warmup pseudo-edges: {len(extra_edges_r1)}")

    # === Round 1: build v26L features with warmup pseudo-edges ===
    print("\n[round 1] building v26L features with warmup pseudo-edges")
    X_train_r1, X_test_r1 = build_v26L_feature_matrices(
        train_pairs, y_train, test_pairs, node_features, node_tfidf,
        extra_edges_r1,
        pair_v24_train, pair_v24_test, pair_v25_train, pair_v25_test,
        total_count, train_count, test_count, n_nodes,
    )
    print(f"  round 1 features: {X_train_r1.shape[1]}")

    # Round 1 blend OOF (for comparison)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    r1_h, r1_c, r1_b = blend_oof(X_train_r1, y_train, cv, SEED)
    print(f"\n[round 1] blend OOF  HGB={r1_h:.5f}  Cat={r1_c:.5f}  blend={r1_b:.5f}")

    # Round 1 full 30-seed test predictions -> input for round 2 pseudo-edges
    print("\n[round 1] 30-seed HGB+CatBoost test predictions")
    r1_pred_h = predict_hgb(X_train_r1, y_train, X_test_r1, n_seeds=30)
    r1_pred_c = predict_cat(X_train_r1, y_train, X_test_r1, n_seeds=30)
    r1_pred = 0.5 * rnk(r1_pred_h) + 0.5 * rnk(r1_pred_c)

    # Round 2 pseudo-edges: union of round 1 + v26L-confident predictions
    # We use a raw-probability threshold on the average of the two models
    # (not the rank-blend, because the rank-blend is in [0,1] but not a
    # probability). The average of HGB and Cat raw probabilities is a
    # reasonable surrogate.
    raw_pred = 0.5 * r1_pred_h + 0.5 * r1_pred_c
    mask_97 = raw_pred >= 0.97
    mask_95 = raw_pred >= 0.95
    print(f"\n[round 2 candidates]")
    print(f"  r1 warmup pseudo-edges: {len(extra_edges_r1)}")
    print(f"  v26L round-1 predictions >= 0.95: {int(mask_95.sum())}")
    print(f"  v26L round-1 predictions >= 0.97: {int(mask_97.sum())}")

    # Round-2 pseudo-edge sets: union of r1 edges with v26L-confident edges
    r2_new_edges_97 = test_pairs[mask_97]
    r2_new_edges_95 = test_pairs[mask_95]

    # Stack r1 edges with new edges; build_candidate_graph and build_graph
    # use set semantics internally so duplicates don't matter
    r2_97 = np.vstack([extra_edges_r1, r2_new_edges_97]) if len(r2_new_edges_97) > 0 else extra_edges_r1
    r2_95 = np.vstack([extra_edges_r1, r2_new_edges_95]) if len(r2_new_edges_95) > 0 else extra_edges_r1

    # Deduplicate
    r2_97 = np.unique(r2_97, axis=0)
    r2_95 = np.unique(r2_95, axis=0)
    print(f"  union r1 + >=0.97 deduplicated: {len(r2_97)}")
    print(f"  union r1 + >=0.95 deduplicated: {len(r2_95)}")

    # === Round 2 variant A: union at threshold 0.97 (conservative) ===
    print("\n[round 2A] build v26L features with union(r1, >=0.97)")
    X_train_r2a, X_test_r2a = build_v26L_feature_matrices(
        train_pairs, y_train, test_pairs, node_features, node_tfidf,
        r2_97,
        pair_v24_train, pair_v24_test, pair_v25_train, pair_v25_test,
        total_count, train_count, test_count, n_nodes,
    )
    r2a_h, r2a_c, r2a_b = blend_oof(X_train_r2a, y_train, cv, SEED)
    print(f"  blend OOF  HGB={r2a_h:.5f}  Cat={r2a_c:.5f}  blend={r2a_b:.5f}  "
          f"({r2a_b - r1_b:+.5f} vs r1)")

    # === Round 2 variant B: union at threshold 0.95 (more aggressive) ===
    print("\n[round 2B] build v26L features with union(r1, >=0.95)")
    X_train_r2b, X_test_r2b = build_v26L_feature_matrices(
        train_pairs, y_train, test_pairs, node_features, node_tfidf,
        r2_95,
        pair_v24_train, pair_v24_test, pair_v25_train, pair_v25_test,
        total_count, train_count, test_count, n_nodes,
    )
    r2b_h, r2b_c, r2b_b = blend_oof(X_train_r2b, y_train, cv, SEED)
    print(f"  blend OOF  HGB={r2b_h:.5f}  Cat={r2b_c:.5f}  blend={r2b_b:.5f}  "
          f"({r2b_b - r1_b:+.5f} vs r1)")

    # Pick the best variant
    results = [
        ("r1 (no round 2)", r1_b, X_train_r1, X_test_r1),
        ("r2a union 0.97", r2a_b, X_train_r2a, X_test_r2a),
        ("r2b union 0.95", r2b_b, X_train_r2b, X_test_r2b),
    ]
    results.sort(key=lambda x: -x[1])
    print(f"\n[best] {results[0][0]}  blend={results[0][1]:.5f}")

    if results[0][0] == "r1 (no round 2)":
        print("\n[v26P] round 2 did not strictly improve. Saving r1 reference only.")
        return

    best_name, best_b, best_X_train, best_X_test = results[0]

    # Final 30-seed on the winner
    print(f"\n[final] 30-seed HGB+CatBoost on {best_name}")
    pred_h = predict_hgb(best_X_train, y_train, best_X_test, n_seeds=30)
    pred_c = predict_cat(best_X_train, y_train, best_X_test, n_seeds=30)
    pred_v26P = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)
    save_sub("submission_v26P.csv", pred_v26P)

    if Path("submission_v26L.csv").exists():
        v26L_known = pd.read_csv("submission_v26L.csv")["Predicted"].to_numpy()
        rL = rnk(v26L_known)
        corr, _ = spearmanr(rL, pred_v26P)
        print(f"\n  v26P vs submission_v26L.csv spearman = {corr:.5f}")
        save_sub("submission_v26P_blend_50.csv", 0.5 * pred_v26P + 0.5 * rL)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
