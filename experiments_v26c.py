"""
v26c — v26b but with an expanded community feature family.

v26b (community + spectral) hit 0.88038 Kaggle (+0.00830 over v25 at 0.87208).
Community alone was slightly better than community+spectral on OOF, so we
drop spectral and instead deepen the community feature family.

New community features (on the same label-free candidate graph as v26b):
  Basic (carried from v26b):
    1. same_community
    2. community_size_min, community_size_max
  New:
    3. community_cn        : # of w in partition(u) s.t. w is a neighbor of
                             both u and v in the candidate graph
    4. community_frac_u    : deg_u inside C(u) divided by deg_u total
    5. community_frac_v    : same for v
    6. internal_density_u  : edge density of u's community
    7. internal_density_v  : edge density of v's community
    8. same_comm_hi_res    : same-community flag at a HIGHER Louvain resolution
                             (finer-grained clustering)
    9. same_comm_lo_res    : same-community flag at a LOWER resolution
                             (coarser clustering)

Rationale:
  - same_community alone is binary; we want gradations of "how tightly
    are u and v bound in their shared cluster"
  - community_cn is like classical CN but restricted to the same community;
    captures second-order proximity inside clusters
  - community_frac_* tells the model whether a node is mostly connected
    inside or outside its community (core vs. periphery)
  - multi-resolution Louvain captures both tight micro-clusters and loose
    macro-clusters; positive pairs tend to be in the same community at
    MULTIPLE resolutions, negatives at FEWER
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
    compute_community_features as compute_comm_basic,
)

import community as community_louvain
import networkx as nx


def compute_extended_community_features(
    pairs, G, partition_default, partition_hi, partition_lo
):
    """Extended community features on the (label-free) candidate graph G.

    Returns columns:
      [same_comm, size_min, size_max,     # 3 from v26b
       comm_cn, frac_u, frac_v,
       int_dens_u, int_dens_v,
       same_hi, same_lo]                  # 6 new  (total 9)
    """
    # Precompute per-community stats for the default partition
    comms = {}
    for node, cid in partition_default.items():
        comms.setdefault(cid, set()).add(node)
    comm_sizes = {cid: len(nodes) for cid, nodes in comms.items()}

    # Edge density per community (within-community edges / possible)
    comm_density = {}
    for cid, members in comms.items():
        m = len(members)
        if m < 2:
            comm_density[cid] = 0.0
            continue
        e_in = 0
        for u in members:
            for v in G.neighbors(u):
                if v in members and v > u:
                    e_in += 1
        comm_density[cid] = (2.0 * e_in) / (m * (m - 1))

    # Degree inside community per node
    intra_deg = {}
    for cid, members in comms.items():
        for u in members:
            intra = sum(1 for v in G.neighbors(u) if v in members)
            intra_deg[u] = intra

    n = pairs.shape[0]
    out = np.zeros((n, 9), dtype=np.float32)

    for i in range(n):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        cu = partition_default.get(u, -1)
        cv = partition_default.get(v, -2)

        out[i, 0] = 1.0 if cu == cv else 0.0
        out[i, 1] = min(comm_sizes.get(cu, 1), comm_sizes.get(cv, 1))
        out[i, 2] = max(comm_sizes.get(cu, 1), comm_sizes.get(cv, 1))

        # community_cn: common neighbors of u,v that are in u's community
        # (we could use either u's or v's community; if same community, it's
        # the same set; if different, use union)
        cn_in_comm = 0
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        members_u = comms.get(cu, set())
        members_v = comms.get(cv, set())
        cn_set = Nu & Nv
        for w in cn_set:
            if w in members_u or w in members_v:
                cn_in_comm += 1
        out[i, 3] = cn_in_comm

        # fraction of node's degree that is inside its own community
        deg_u = len(Nu)
        deg_v = len(Nv)
        out[i, 4] = intra_deg.get(u, 0) / max(deg_u, 1)
        out[i, 5] = intra_deg.get(v, 0) / max(deg_v, 1)

        out[i, 6] = comm_density.get(cu, 0.0)
        out[i, 7] = comm_density.get(cv, 0.0)

        # Multi-resolution same-community flags
        out[i, 8] = 1.0 if partition_hi.get(u, -1) == partition_hi.get(v, -2) else 0.0
        # reuse the same column for "low res" by stacking later

    # Low-res column as a separate 10th
    lo_col = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        lo_col[i, 0] = 1.0 if partition_lo.get(u, -1) == partition_lo.get(v, -2) else 0.0

    return np.hstack([out, lo_col])


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

    # Pair transductive (v24 + v25)
    print("[v25] pair-level transductive features")
    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v24_test = compute_pair_transductive_v24(test_pairs, train_partners, test_partners)
    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_test = compute_pair_transductive_v25(
        test_pairs, train_partners, test_partners, test_count, train_count, total_count)

    # v19 self-training
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
    print(f"  v25 features: {X_train_v25.shape[1]}")

    # === Candidate graph (label-free) ===
    print("\n[v26c] building candidate graph")
    G_cand = build_candidate_graph(train_pairs, test_pairs, extra_edges, n_nodes)
    print(f"  |V|={G_cand.number_of_nodes()}  |E|={G_cand.number_of_edges()}  "
          f"components={nx.number_connected_components(G_cand)}")

    # Three Louvain resolutions
    print("[v26c] Louvain at 3 resolutions")
    partition_default = community_louvain.best_partition(G_cand, weight="weight", random_state=SEED, resolution=1.0)
    partition_hi = community_louvain.best_partition(G_cand, weight="weight", random_state=SEED, resolution=2.0)
    partition_lo = community_louvain.best_partition(G_cand, weight="weight", random_state=SEED, resolution=0.5)
    print(f"  resolution=1.0: {len(set(partition_default.values()))} communities")
    print(f"  resolution=2.0: {len(set(partition_hi.values()))} communities")
    print(f"  resolution=0.5: {len(set(partition_lo.values()))} communities")

    print("[v26c] computing extended community features")
    comm_train = compute_extended_community_features(
        train_pairs, G_cand, partition_default, partition_hi, partition_lo)
    comm_test = compute_extended_community_features(
        test_pairs, G_cand, partition_default, partition_hi, partition_lo)

    labels = [
        "same_comm", "size_min", "size_max",
        "comm_cn", "frac_u", "frac_v",
        "int_dens_u", "int_dens_v",
        "same_hi", "same_lo",
    ]
    print(f"  new features: {comm_train.shape[1]} ({', '.join(labels)})")
    for j, label in enumerate(labels):
        p_mean = comm_train[y_train == 1, j].mean()
        n_mean = comm_train[y_train == 0, j].mean()
        print(f"    {label:12s}: pos={p_mean:.3f}  neg={n_mean:.3f}  "
              f"gap={p_mean - n_mean:+.3f}")

    X_train_v26c = np.hstack([X_train_v25, comm_train])
    X_test_v26c = np.hstack([X_test_v25, comm_test])
    print(f"  v26c features: {X_train_v26c.shape[1]} (+{X_train_v26c.shape[1] - X_train_v25.shape[1]})")

    # === CV comparison ===
    print("\n[CV] OOF AUC")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v25 = np.zeros(len(y_train), dtype=np.float64)
    oof_v26c = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26c, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v25[tr], y_train[tr])
        oof_v25[va] = m1.predict_proba(X_train_v25[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26c[tr], y_train[tr])
        oof_v26c[va] = m2.predict_proba(X_train_v26c[va])[:, 1]
    auc_v25 = roc_auc_score(y_train, oof_v25)
    auc_v26c = roc_auc_score(y_train, oof_v26c)
    print(f"  v25  OOF = {auc_v25:.5f}")
    print(f"  v26c OOF = {auc_v26c:.5f}  ({auc_v26c - auc_v25:+.5f})")

    # === Final 30-seed HGB+CatBoost ===
    print("\n[final] 30-seed HGB+CatBoost on v26c")
    pred_h = predict_hgb(X_train_v26c, y_train, X_test_v26c, n_seeds=30)
    pred_c = predict_cat(X_train_v26c, y_train, X_test_v26c, n_seeds=30)
    pred_v26c = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
    save_sub("submission_v26c.csv", pred_v26c)

    # Blend with v26b (we have its submission file on disk from the earlier run)
    v26b_path = Path("submission_v26b.csv")
    if v26b_path.exists():
        v26b = pd.read_csv(v26b_path)["Predicted"].to_numpy()
        rv26b = normalize_rank(v26b)
        save_sub("submission_v26c_blend_50v26b.csv", 0.5 * pred_v26c + 0.5 * rv26b)
        save_sub("submission_v26c_blend_70v26c.csv", 0.7 * pred_v26c + 0.3 * rv26b)
        corr, _ = spearmanr(rv26b, pred_v26c)
        print(f"\n  v26c vs v26b spearman = {corr:.5f}")

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
