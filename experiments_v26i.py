"""
v26i — compound the v26h_pure consensus insight onto more features.

v26h_pure scored 0.88432 Kaggle (+0.00352 over v26g) by denoising the
single-seed Louvain `same_community` feature via 20-seed averaging.
Single-seed Louvain ARI on this graph is ~0.2, meaning every other
feature derived from a single-seed partition (comm_cn, community_size,
...) is equally noisy and should benefit from the same trick.

v26i tests four candidate features, all derived from the SAME 20
unweighted Louvain partitions that v26h_pure already computes:

  Denoising the existing features (averaging across 20 seeds):
    1. cons_comm_cn  : mean of comm_cn across 20 partitions
    2. cons_size_min : mean of community_size_min across 20 partitions
    3. cons_size_max : mean of community_size_max across 20 partitions

  Node stability (new direction):
    4. node_entropy_max : max Shannon entropy of u's and v's community-ID
                           distribution across the 20 runs. High entropy =
                           "boundary" node whose assignment flips between
                           runs; low entropy = "core" node that Louvain
                           always places in the same community. Tells the
                           booster how much to trust the other community
                           features for this specific pair.
    5. node_entropy_min : same, but min

  Canonical consensus partition (Lancichinetti & Fortunato 2012):
    6. canonical_same_comm : run Louvain on the consensus matrix
                              (edge weight = fraction of 20 runs where u,v
                              are in same community), use the resulting
                              partition as a stable canonical view
    7. canonical_comm_cn   : recompute comm_cn from the canonical
                              partition

Reference:
  Lancichinetti & Fortunato — Consensus clustering in complex networks.
  Scientific Reports 2 (2012). Building a new graph from the consensus
  matrix and reclustering it gives a provably stable partition.

Ablation strategy: each new feature is evaluated individually on top of
v26h_pure, then the winning combination is retrained as the final
submission. Same minimal-risk single-family discipline that saved us
in v26d (vs v26c) and v26g (vs v26f).
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


def compute_all_partitions(G, n_seeds=20, base_seed=SEED):
    """Return a list of n_seeds partitions (dict node -> community id)."""
    partitions = []
    for s in range(n_seeds):
        part = community_louvain.best_partition(
            G, weight="weight", random_state=base_seed + s
        )
        partitions.append(part)
    return partitions


def consensus_same_community_from_partitions(pairs, partitions):
    n = pairs.shape[0]
    counts = np.zeros(n, dtype=np.float32)
    for part in partitions:
        for i in range(n):
            u, v = int(pairs[i, 0]), int(pairs[i, 1])
            if part.get(u, -1) == part.get(v, -2):
                counts[i] += 1.0
    return counts / float(len(partitions))


def consensus_comm_cn(pairs, G, partitions):
    """Average comm_cn across multiple partitions of the same graph."""
    n = pairs.shape[0]
    total = np.zeros(n, dtype=np.float64)
    for part in partitions:
        vals = compute_comm_cn(pairs, G, part).ravel()
        total += vals
    return (total / float(len(partitions))).astype(np.float32).reshape(-1, 1)


def consensus_community_sizes(pairs, partitions):
    """Average community_size_min / community_size_max across partitions."""
    n = pairs.shape[0]
    total_min = np.zeros(n, dtype=np.float64)
    total_max = np.zeros(n, dtype=np.float64)
    for part in partitions:
        sizes = {}
        for node, cid in part.items():
            sizes[cid] = sizes.get(cid, 0) + 1
        for i in range(n):
            u, v = int(pairs[i, 0]), int(pairs[i, 1])
            su = sizes.get(part.get(u, -1), 1)
            sv = sizes.get(part.get(v, -1), 1)
            total_min[i] += min(su, sv)
            total_max[i] += max(su, sv)
    k = float(len(partitions))
    smin = (total_min / k).astype(np.float32).reshape(-1, 1)
    smax = (total_max / k).astype(np.float32).reshape(-1, 1)
    return smin, smax


def compute_node_entropy(partitions, n_nodes):
    """For each node, Shannon entropy (in bits) of its community-ID
    distribution across all partitions.

    High entropy = unstable assignment = boundary node.
    """
    from math import log2
    counts = [{} for _ in range(n_nodes)]
    for part in partitions:
        for node, cid in part.items():
            counts[node][cid] = counts[node].get(cid, 0) + 1
    k = float(len(partitions))
    entropy = np.zeros(n_nodes, dtype=np.float32)
    for u in range(n_nodes):
        h = 0.0
        for cnt in counts[u].values():
            p = cnt / k
            if p > 0:
                h -= p * log2(p)
        entropy[u] = h
    return entropy


def node_entropy_pair_features(pairs, node_entropy):
    u = pairs[:, 0].astype(np.int64)
    v = pairs[:, 1].astype(np.int64)
    hu = node_entropy[u]
    hv = node_entropy[v]
    return np.column_stack([
        np.minimum(hu, hv),
        np.maximum(hu, hv),
    ]).astype(np.float32)


def build_canonical_partition(partitions, n_nodes, threshold=0.5, seed=SEED):
    """Lancichinetti & Fortunato consensus clustering.

    1. Build a consensus graph where edge weight between u and v is the
       fraction of `partitions` in which they land in the same community
    2. Keep only edges with weight >= threshold (drops noisy boundary ones)
    3. Run Louvain on the consensus graph to get a canonical partition

    Returns a dict node -> canonical community id.
    """
    k = float(len(partitions))
    # Build a co-occurrence count dict so we only visit edges that actually
    # appear in at least one partition (the graph is ~14k edges so this
    # is tiny vs the full n_nodes^2 loop)
    co_counts = {}
    for part in partitions:
        # Group nodes by community
        groups = {}
        for node, cid in part.items():
            groups.setdefault(cid, []).append(node)
        for members in groups.values():
            members_sorted = sorted(members)
            for i in range(len(members_sorted)):
                a = members_sorted[i]
                for j in range(i + 1, len(members_sorted)):
                    b = members_sorted[j]
                    key = (a, b)
                    co_counts[key] = co_counts.get(key, 0) + 1

    # Build consensus graph
    G_cons = nx.Graph()
    G_cons.add_nodes_from(range(n_nodes))
    for (a, b), cnt in co_counts.items():
        w = cnt / k
        if w >= threshold:
            G_cons.add_edge(a, b, weight=w)

    # Louvain on the consensus graph (deterministic-ish since the input
    # already aggregates 20 stochastic runs)
    canonical = community_louvain.best_partition(
        G_cons, weight="weight", random_state=seed
    )
    return canonical


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

    # === Build base feature sets up through v26h_pure ===
    print("\n[base] v26d + v26g + v26h_pure")
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

    G_text = build_text_weighted_candidate_graph(
        train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf,
        alpha=1.0, beta=3.0,
    )
    part_text = run_louvain(G_text, seed=SEED)
    comm_text_train = compute_community_features(train_pairs, part_text, n_nodes)
    comm_text_test = compute_community_features(test_pairs, part_text, n_nodes)

    X_train_v26g = np.hstack([X_train_v26d, comm_text_train])
    X_test_v26g = np.hstack([X_test_v26d, comm_text_test])

    # === v26h_pure base: add cons_unwt across 20 seeds ===
    N_SEEDS = 20
    print(f"[consensus] running {N_SEEDS} Louvain partitions on unweighted graph")
    t1 = time.time()
    partitions_unwt = compute_all_partitions(G_unwt, n_seeds=N_SEEDS, base_seed=SEED)
    print(f"  {N_SEEDS} partitions computed in {time.time()-t1:.1f}s")

    cons_unwt_train = consensus_same_community_from_partitions(
        train_pairs, partitions_unwt).reshape(-1, 1)
    cons_unwt_test = consensus_same_community_from_partitions(
        test_pairs, partitions_unwt).reshape(-1, 1)

    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])
    print(f"  v26h_pure features: {X_train_v26hp.shape[1]}")

    # === v26i candidate features ===
    print(f"\n[v26i] candidate features from the same {N_SEEDS} partitions")

    # 1. cons_comm_cn
    t1 = time.time()
    cons_ccn_train = consensus_comm_cn(train_pairs, G_unwt, partitions_unwt)
    cons_ccn_test = consensus_comm_cn(test_pairs, G_unwt, partitions_unwt)
    print(f"  cons_comm_cn computed in {time.time()-t1:.1f}s  "
          f"pos={cons_ccn_train[y_train==1].mean():.3f}  "
          f"neg={cons_ccn_train[y_train==0].mean():.3f}")

    # 2-3. cons_size_min / cons_size_max
    t1 = time.time()
    cons_smin_train, cons_smax_train = consensus_community_sizes(train_pairs, partitions_unwt)
    cons_smin_test, cons_smax_test = consensus_community_sizes(test_pairs, partitions_unwt)
    print(f"  cons_sizes computed in {time.time()-t1:.1f}s  "
          f"smin pos={cons_smin_train[y_train==1].mean():.1f}  "
          f"neg={cons_smin_train[y_train==0].mean():.1f}")

    # 4-5. node entropy (max/min over the pair)
    t1 = time.time()
    node_ent = compute_node_entropy(partitions_unwt, n_nodes)
    node_ent_train = node_entropy_pair_features(train_pairs, node_ent)
    node_ent_test = node_entropy_pair_features(test_pairs, node_ent)
    print(f"  node_entropy computed in {time.time()-t1:.1f}s  "
          f"mean={node_ent.mean():.3f}  max={node_ent.max():.3f}")
    print(f"    ent_min pos={node_ent_train[y_train==1, 0].mean():.3f}  "
          f"neg={node_ent_train[y_train==0, 0].mean():.3f}")
    print(f"    ent_max pos={node_ent_train[y_train==1, 1].mean():.3f}  "
          f"neg={node_ent_train[y_train==0, 1].mean():.3f}")

    # 6-7. canonical partition via Lancichinetti consensus clustering
    t1 = time.time()
    canonical = build_canonical_partition(
        partitions_unwt, n_nodes, threshold=0.5, seed=SEED
    )
    n_canonical = len(set(canonical.values()))
    print(f"  canonical partition: {n_canonical} communities  "
          f"({time.time()-t1:.1f}s)")

    # Build a graph-like object so compute_comm_cn/community_features work:
    # canonical is just a dict node -> cid, which is what those functions want.
    canon_comm_train = compute_community_features(train_pairs, canonical, n_nodes)
    canon_comm_test = compute_community_features(test_pairs, canonical, n_nodes)

    canon_cn_train = compute_comm_cn(train_pairs, G_unwt, canonical)
    canon_cn_test = compute_comm_cn(test_pairs, G_unwt, canonical)
    print(f"  canon same_comm: pos={canon_comm_train[y_train==1, 0].mean():.3f}  "
          f"neg={canon_comm_train[y_train==0, 0].mean():.3f}")
    print(f"  canon comm_cn:   pos={canon_cn_train[y_train==1].mean():.3f}  "
          f"neg={canon_cn_train[y_train==0].mean():.3f}")

    # === Ablation: each new feature alone on top of v26h_pure ===
    print("\n[ablation] each candidate on top of v26h_pure")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def oof_hgb(X_tr):
        oof = np.zeros(len(y_train), dtype=np.float64)
        for fold, (tr, va) in enumerate(cv.split(X_tr, y_train), 1):
            m = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
            m.fit(X_tr[tr], y_train[tr])
            oof[va] = m.predict_proba(X_tr[va])[:, 1]
        return roc_auc_score(y_train, oof)

    auc_hp = oof_hgb(X_train_v26hp)
    print(f"  v26h_pure base           OOF = {auc_hp:.5f}")

    candidates = [
        ("+cons_comm_cn",      cons_ccn_train,   cons_ccn_test),
        ("+cons_size_min",     cons_smin_train,  cons_smin_test),
        ("+cons_size_max",     cons_smax_train,  cons_smax_test),
        ("+cons_sizes(both)",  np.hstack([cons_smin_train, cons_smax_train]),
                                np.hstack([cons_smin_test, cons_smax_test])),
        ("+node_entropy(min,max)", node_ent_train, node_ent_test),
        ("+canonical same_comm (3)",  canon_comm_train, canon_comm_test),
        ("+canonical comm_cn",        canon_cn_train,   canon_cn_test),
    ]

    results = []
    for name, ftr_tr, ftr_te in candidates:
        X_tr = np.hstack([X_train_v26hp, ftr_tr])
        auc = oof_hgb(X_tr)
        delta = auc - auc_hp
        results.append((name, auc, delta, ftr_tr, ftr_te))
        flag = "  *" if delta > 0 else ""
        print(f"  {name:32s} OOF = {auc:.5f}  ({delta:+.5f}){flag}")

    # === Winners-only combination ===
    winners_tr = [X_train_v26hp]
    winners_te = [X_test_v26hp]
    winner_names = []
    for name, auc, delta, ftr_tr, ftr_te in results:
        if delta > 0:
            winners_tr.append(ftr_tr)
            winners_te.append(ftr_te)
            winner_names.append(name)

    if winner_names:
        X_train_v26i = np.hstack(winners_tr)
        X_test_v26i = np.hstack(winners_te)
        print(f"\n[v26i] winners-only combo: {winner_names}")
        print(f"  v26i features: {X_train_v26i.shape[1]}")
        auc_v26i = oof_hgb(X_train_v26i)
        print(f"  v26i OOF = {auc_v26i:.5f}  ({auc_v26i - auc_hp:+.5f})")

        oof_v26i_cat = np.zeros(len(y_train), dtype=np.float64)
        for fold, (tr, va) in enumerate(cv.split(X_train_v26i, y_train), 1):
            mc = CatBoostClassifier(**CAT_PARAMS, random_seed=SEED + fold)
            mc.fit(X_train_v26i[tr], y_train[tr])
            oof_v26i_cat[va] = mc.predict_proba(X_train_v26i[va])[:, 1]
        print(f"  v26i CatBoost OOF = {roc_auc_score(y_train, oof_v26i_cat):.5f}")

        # Final 30-seed ensemble on winners-only
        print("\n[final] 30-seed HGB+CatBoost on v26i (winners only)")
        pred_h = predict_hgb(X_train_v26i, y_train, X_test_v26i, n_seeds=30)
        pred_c = predict_cat(X_train_v26i, y_train, X_test_v26i, n_seeds=30)
        pred_v26i = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
        save_sub("submission_v26i.csv", pred_v26i)

        if Path("submission_v26h_pure.csv").exists():
            v26hp = pd.read_csv("submission_v26h_pure.csv")["Predicted"].to_numpy()
            rhp = normalize_rank(v26hp)
            corr, _ = spearmanr(rhp, pred_v26i)
            print(f"\n  v26i vs submission_v26h_pure.csv spearman = {corr:.5f}")
            save_sub("submission_v26i_blend_50v26hp.csv", 0.5 * pred_v26i + 0.5 * rhp)
    else:
        print("\n[v26i] no winners — nothing improved on v26h_pure alone.")

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
