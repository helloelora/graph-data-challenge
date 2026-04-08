"""
v26j — classical link-prediction heuristics on the CANDIDATE graph.

All of v19/v25's graph heuristics (CN, Jaccard, Adamic-Adar, Resource
Allocation, Sorensen, Katz, paths3, neighborhood TF-IDF) are computed
on the *real* adjacency graph: train positives + v25 self-training
pseudo-edges. That graph is very sparse (mean deg ~2.9) and 91.5% of
test pairs have CN = 0 on it.

The *candidate graph* built in v26b contains every pair in `train.txt`
(regardless of label) plus every pair in `test.txt` plus the pseudo-
edges. It is much denser (mean deg ~7.75) and the edges carry a
fundamentally different kind of information: "did the competition
organizers ask us about this pair", not "are they connected in the
original graph". We already use this graph for Louvain and spectral
features (v26b/d/g/h) but we have never computed the classical LP
heuristics on it.

Recomputing CN, Jaccard, AA, RA, Sorensen, LHN, Katz on the candidate
graph gives a completely new feature family: same mathematical forms
v19 uses, different input graph, different signal. The candidate graph
is leakage-free for training pairs (train labels are never used to
decide which edges exist in it).

Seven new candidate features on top of v26h_pure:
  cand_cn         common neighbors in candidate graph
  cand_jaccard    |N(u) ∩ N(v)| / |N(u) ∪ N(v)| in candidate graph
  cand_aa         Adamic-Adar in candidate graph
  cand_ra         resource allocation in candidate graph
  cand_sorensen   2|N∩|/(|N(u)|+|N(v)|) in candidate graph
  cand_lhn        Leicht-Holme-Newman: |N∩| / (|N(u)|*|N(v)|)
  cand_katz       Katz index approximation at beta=0.005

Each is ablated individually on top of v26h_pure (the Kaggle best at
0.88432). Only features that strictly improve the HGB+CatBoost
rank-blend OOF are kept. No bundling of marginal or hurting features
(v26c/v26f/v26i failure mode).

Reference:
  Liben-Nowell & Kleinberg - The Link Prediction Problem for Social
  Networks (JASIST 2007). Catalogs the classical heuristics and
  benchmarks their transfer across different graph constructions.
"""

import math
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

import networkx as nx


def compute_candidate_heuristics(pairs, G, n_nodes):
    """Compute classical LP heuristics on the candidate graph G for every
    pair in `pairs`. Returns an (n, 7) float32 array with columns:
    [cn, jaccard, aa, ra, sorensen, lhn, katz]
    """
    # Precompute neighbor sets and degrees once per node
    neighbors = [set(G.neighbors(u)) for u in range(n_nodes)]
    deg = np.array([len(neighbors[u]) for u in range(n_nodes)], dtype=np.float32)

    n = pairs.shape[0]
    out = np.zeros((n, 7), dtype=np.float32)

    for i in range(n):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        Nu = neighbors[u]
        Nv = neighbors[v]
        if len(Nu) <= len(Nv):
            common = Nu & Nv
        else:
            common = Nv & Nu

        cn = float(len(common))

        union_size = len(Nu | Nv)
        jaccard = cn / max(union_size, 1)

        deg_sum = deg[u] + deg[v]
        sorensen = 2.0 * cn / max(deg_sum, 1.0)

        lhn = cn / max(deg[u] * deg[v], 1.0)

        aa = 0.0
        ra = 0.0
        for w in common:
            dw = deg[w]
            if dw > 1:
                aa += 1.0 / math.log(dw)
            if dw > 0:
                ra += 1.0 / dw

        # Katz approximation: beta^2 * CN + beta^3 * paths3
        # (paths of length 3 through the candidate graph)
        beta = 0.005
        paths3 = 0.0
        for w in Nu:
            # paths of length 3 from u through w: w's neighbors that
            # are adjacent to v
            for x in neighbors[w]:
                if x == u or x == v:
                    continue
                if v in neighbors[x]:
                    paths3 += 1.0
        katz = (beta ** 2) * cn + (beta ** 3) * paths3

        out[i, 0] = cn
        out[i, 1] = jaccard
        out[i, 2] = aa
        out[i, 3] = ra
        out[i, 4] = sorensen
        out[i, 5] = lhn
        out[i, 6] = katz

    return out


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

    # Pair-level transductive (v24+v25)
    print("[v25] pair-level transductive features")
    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v24_test = compute_pair_transductive_v24(test_pairs, train_partners, test_partners)
    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_test = compute_pair_transductive_v25(
        test_pairs, train_partners, test_partners, test_count, train_count, total_count)

    # v19 self-training + features
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

    # v26h_pure base: + consensus same_community from 20 Louvain seeds
    cons_unwt_train = compute_consensus_same_community(
        train_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        test_pairs, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)

    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])
    print(f"\n[v26h_pure base] {X_train_v26hp.shape[1]} features")

    # === v26j NEW: candidate-graph heuristics ===
    print("\n[v26j] computing candidate-graph LP heuristics")
    t1 = time.time()
    cand_train = compute_candidate_heuristics(train_pairs, G_unwt, n_nodes)
    cand_test = compute_candidate_heuristics(test_pairs, G_unwt, n_nodes)
    print(f"  computed in {time.time()-t1:.1f}s")

    labels = ["cand_cn", "cand_jaccard", "cand_aa", "cand_ra",
              "cand_sorensen", "cand_lhn", "cand_katz"]
    for j, label in enumerate(labels):
        p = cand_train[y_train == 1, j].mean()
        n = cand_train[y_train == 0, j].mean()
        print(f"  {label:14s}: pos={p:8.3f}  neg={n:8.3f}  gap={p-n:+.3f}")

    # Ablation: each feature alone on the HGB+CatBoost blend
    print("\n[ablation] HGB+CatBoost blend OOF per feature")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    hp_h, hp_c, hp_b = blend_oof(X_train_v26hp, y_train, cv, SEED)
    print(f"  v26h_pure base   HGB={hp_h:.5f}  Cat={hp_c:.5f}  blend={hp_b:.5f}")

    winning = []
    deltas = {}
    for j, label in enumerate(labels):
        X_tr = np.hstack([X_train_v26hp, cand_train[:, j:j+1]])
        h, c, b = blend_oof(X_tr, y_train, cv, SEED)
        delta = b - hp_b
        deltas[label] = (h, c, b, delta)
        flag = "  *" if delta > 0 else ""
        print(f"  +{label:13s}   HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  "
              f"({delta:+.5f}){flag}")
        if delta > 0:
            winning.append(j)

    if not winning:
        print("\n[v26j] no feature strictly improved the blend. Stopping.")
        return

    # Build the winners-only combo
    print(f"\n[v26j winners] columns: {[labels[j] for j in winning]}")
    cand_train_win = cand_train[:, winning]
    cand_test_win = cand_test[:, winning]

    X_train_v26j = np.hstack([X_train_v26hp, cand_train_win])
    X_test_v26j = np.hstack([X_test_v26hp, cand_test_win])

    h, c, b = blend_oof(X_train_v26j, y_train, cv, SEED)
    print(f"  v26j HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  ({b - hp_b:+.5f})")

    # Final 30-seed ensemble
    print("\n[final] 30-seed HGB+CatBoost on v26j winners")
    pred_h = predict_hgb(X_train_v26j, y_train, X_test_v26j, n_seeds=30)
    pred_c = predict_cat(X_train_v26j, y_train, X_test_v26j, n_seeds=30)
    pred_v26j = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)
    save_sub("submission_v26j.csv", pred_v26j)

    if Path("submission_v26h_pure.csv").exists():
        v26hp = pd.read_csv("submission_v26h_pure.csv")["Predicted"].to_numpy()
        rhp = rnk(v26hp)
        corr, _ = spearmanr(rhp, pred_v26j)
        print(f"\n  v26j vs submission_v26h_pure.csv spearman = {corr:.5f}")
        save_sub("submission_v26j_blend_50.csv", 0.5 * pred_v26j + 0.5 * rhp)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
