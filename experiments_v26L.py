"""
v26L — multi-resolution consensus Louvain, compounding the v26h_pure insight.

v26h_pure (0.88432 Kaggle, +0.00352 over v26g) proved that averaging the
`same_community` flag across 20 stochastic Louvain runs on the candidate
graph de-noises the feature dramatically: single-seed Louvain ARI on this
graph is only ~0.2, so any individual run's binary flag is nearly a coin
flip for boundary pairs. The continuous 20-seed consensus fraction fixed
that.

Then we hit a wall:
  - v26i (canonical Lancichinetti partition + cons_sizes): regressed Kaggle
  - v26j (candidate-graph LP heuristics):                   all regressed
  - v26k (candidate-graph neighbor text):                   all regressed
  - SEAL radius-1 GNN:                                      OOF only 0.701

What's shared across those failures: they all try to extract new signal from
places that are downstream of what we already have. The only direction that
*added genuinely new information* was "more samples from a stochastic
clustering algorithm". v26L compounds exactly that, in two orthogonal ways:

  More *samples* at the same resolution: already exploited in v26h_pure
    (20 seeds at resolution=1.0).

  More *resolutions*: unexplored. Louvain's resolution parameter controls
    the granularity of the partition. Different resolutions produce
    *structurally different* partitions: at low resolution we see a few
    big coarse clusters, at high resolution many small fine-grained ones.
    A pair that is co-clustered at multiple resolutions is much more
    robustly "together" than one that's co-clustered only at a single
    resolution.

v26L runs 20 seeds at each of 5 resolutions on the unweighted candidate
graph. For each pair, each resolution produces one scalar feature: the
fraction of 20 seeds in which u and v land in the same community at that
resolution.

Five candidate features on top of v26h_pure:
  cons_res07  : consensus at Louvain resolution 0.7 (finer clusters)
  cons_res10  : consensus at Louvain resolution 1.0 (same as v26h_pure — sanity)
  cons_res13  : consensus at Louvain resolution 1.3
  cons_res16  : consensus at Louvain resolution 1.6
  cons_res20  : consensus at Louvain resolution 2.0 (coarser clusters)

Note: the resolution semantics in python-louvain are non-standard. In v26c
we observed resolution=0.5 → 1206 communities (tiny) while resolution=2.0
→ 55 communities (bigger). We avoid the extreme 0.5 case here and pick 5
values that produce meaningful partitions without degenerating into
singletons.

Each resolution's feature is ablated individually on top of v26h_pure, on
the HGB+CatBoost rank-blend OOF (the metric that matches Kaggle — never
again HGB-alone like v26i). Ship only the features with strictly positive
blend delta. v26h_pure's cons_res10 (the 1.0 resolution feature) is
already in the base, so we expect it to ablate near zero — the test is
whether *different* resolutions add orthogonal signal.
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

try:
    from community import community_louvain
except (ImportError, AttributeError):
    import community as community_louvain
import networkx as nx


def compute_consensus_at_resolution(pairs, G, n_seeds, base_seed, resolution):
    """Fraction of n_seeds Louvain runs (at the given resolution) in which
    u and v are in the same community.
    """
    n = pairs.shape[0]
    counts = np.zeros(n, dtype=np.float32)

    n_comms_by_seed = []
    for s in range(n_seeds):
        partition = community_louvain.best_partition(
            G, weight="weight", random_state=base_seed + s,
            resolution=resolution,
        )
        n_comms_by_seed.append(len(set(partition.values())))
        for i in range(n):
            u, v = int(pairs[i, 0]), int(pairs[i, 1])
            if partition.get(u, -1) == partition.get(v, -2):
                counts[i] += 1.0

    return counts / float(n_seeds), n_comms_by_seed


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

    # v26h_pure base: + cons_unwt (consensus at resolution 1.0, 20 seeds)
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        test_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)

    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])
    print(f"\n[v26h_pure base] {X_train_v26hp.shape[1]} features")

    # === v26L: multi-resolution consensus Louvain ===
    resolutions = [0.7, 1.0, 1.3, 1.6, 2.0]
    labels = [f"cons_res{int(r*10):02d}" for r in resolutions]
    print(f"\n[v26L] computing consensus at {len(resolutions)} resolutions (20 seeds each)")

    cons_train_by_res = {}
    cons_test_by_res = {}
    for r, label in zip(resolutions, labels):
        t1 = time.time()
        cons_tr, n_comms_tr = compute_consensus_at_resolution(
            train_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r
        )
        cons_te, _ = compute_consensus_at_resolution(
            test_pairs, G_unwt, n_seeds=20, base_seed=SEED, resolution=r
        )
        cons_train_by_res[label] = cons_tr.reshape(-1, 1)
        cons_test_by_res[label] = cons_te.reshape(-1, 1)
        n_comms_avg = np.mean(n_comms_tr)
        p = cons_tr[y_train == 1].mean()
        n = cons_tr[y_train == 0].mean()
        print(f"  {label}  (res={r:.1f}, ~{n_comms_avg:.0f} communities)  "
              f"pos={p:.3f}  neg={n:.3f}  gap={p-n:+.3f}  ({time.time()-t1:.1f}s)")

    # Ablation: each resolution alone on top of v26h_pure
    print("\n[ablation] HGB+CatBoost blend OOF per resolution")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    hp_h, hp_c, hp_b = blend_oof(X_train_v26hp, y_train, cv, SEED)
    print(f"  v26h_pure base   HGB={hp_h:.5f}  Cat={hp_c:.5f}  blend={hp_b:.5f}")

    deltas = {}
    for label in labels:
        X_tr = np.hstack([X_train_v26hp, cons_train_by_res[label]])
        h, c, b = blend_oof(X_tr, y_train, cv, SEED)
        delta = b - hp_b
        deltas[label] = (h, c, b, delta)
        flag = "  *" if delta > 0 else ""
        print(f"  +{label:11s}   HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
              f"({delta:+.5f}){flag}")

    # Try all 5 together (should be better than any single if they are diverse)
    X_all = np.hstack([X_train_v26hp] + [cons_train_by_res[l] for l in labels])
    h, c, b = blend_oof(X_all, y_train, cv, SEED)
    delta = b - hp_b
    flag = "  *" if delta > 0 else ""
    print(f"  +all 5 together  HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
          f"({delta:+.5f}){flag}")

    # Winners only: strict discipline — each feature must be strictly positive alone
    winning = [l for l in labels if deltas[l][3] > 0]
    if not winning:
        print("\n[v26L] no feature strictly improved. Stopping.")
        return

    print(f"\n[v26L winners alone] {winning}")
    X_train_v26L = np.hstack([X_train_v26hp] + [cons_train_by_res[l] for l in winning])
    X_test_v26L = np.hstack([X_test_v26hp] + [cons_test_by_res[l] for l in winning])

    h, c, b = blend_oof(X_train_v26L, y_train, cv, SEED)
    print(f"  v26L HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  ({b - hp_b:+.5f})")

    # Final 30-seed ensemble
    print("\n[final] 30-seed HGB+CatBoost on v26L winners")
    pred_h = predict_hgb(X_train_v26L, y_train, X_test_v26L, n_seeds=30)
    pred_c = predict_cat(X_train_v26L, y_train, X_test_v26L, n_seeds=30)
    pred_v26L = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)
    save_sub("submission_v26L.csv", pred_v26L)

    if Path("submission_v26h_pure.csv").exists():
        v26hp = pd.read_csv("submission_v26h_pure.csv")["Predicted"].to_numpy()
        rhp = rnk(v26hp)
        corr, _ = spearmanr(rhp, pred_v26L)
        print(f"\n  v26L vs submission_v26h_pure.csv spearman = {corr:.5f}")
        save_sub("submission_v26L_blend_50.csv", 0.5 * pred_v26L + 0.5 * rhp)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
