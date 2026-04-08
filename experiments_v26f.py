"""
v26f — v26d + TEXT-WEIGHTED community features.

v26d's community graph treats every candidate edge equally (weight 1.0).
v26f instead weights each candidate edge by the TF-IDF cosine similarity
of the two endpoints' keyword features, which makes Louvain text-aware:
actors with similar Wikipedia keywords are pulled into the same community
more strongly than unrelated actors.

Intuition: two actors who share lots of keywords AND co-occur in candidate
pairs are much more likely to actually have an edge in the original graph
than two actors who happen to co-occur in a pair but have no text
similarity (e.g., a random negative).

v26e tried a different *algorithm* (Leiden) on the same graph and got only
+0.00030 OOF. v26f tries a different *graph* construction — potentially a
much more orthogonal direction.
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

import networkx as nx


def build_text_weighted_candidate_graph(
    train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf,
    alpha=1.0, beta=1.0
):
    """Candidate graph where each edge weight = alpha + beta * tfidf_cos(u, v).

    - alpha: base weight so every candidate edge still contributes
    - beta:  scales how much text similarity amplifies edge weight
    - tfidf_cos is clamped to [0, 1] (it's already non-negative)
    """
    # Precompute TF-IDF cosine for each unique candidate pair on the fly
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    def tfidf_cos(u, v):
        a = node_tfidf[u]
        b = node_tfidf[v]
        # sparse inner product
        sim = float((a.multiply(b)).sum())
        return max(min(sim, 1.0), 0.0)

    for u, v in train_pairs:
        u, v = int(u), int(v)
        if u != v:
            w = alpha + beta * tfidf_cos(u, v)
            G.add_edge(u, v, weight=w)
    for u, v in test_pairs:
        u, v = int(u), int(v)
        if u != v and not G.has_edge(u, v):
            w = alpha + beta * tfidf_cos(u, v)
            G.add_edge(u, v, weight=w)
    if extra_edges is not None:
        for u, v in extra_edges:
            u, v = int(u), int(v)
            if u != v and not G.has_edge(u, v):
                w = alpha + beta * tfidf_cos(u, v)
                G.add_edge(u, v, weight=w)

    return G


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

    print("\n[v19] self-training")
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
    print(f"  v26d features: {X_train_v26d.shape[1]}")

    # === v26f NEW: text-weighted candidate graph ===
    print("\n[v26f] text-weighted candidate graph (alpha=1.0, beta=3.0)")
    G_text = build_text_weighted_candidate_graph(
        train_pairs, test_pairs, extra_edges, n_nodes, node_tfidf,
        alpha=1.0, beta=3.0,
    )
    print(f"  |V|={G_text.number_of_nodes()}  |E|={G_text.number_of_edges()}")

    # How different is the text-weighted partition from the unweighted one?
    part_text = run_louvain(G_text, seed=SEED)
    n_comms_text = len(set(part_text.values()))
    print(f"  {n_comms_text} text-weighted Louvain communities")

    from sklearn.metrics import adjusted_rand_score
    nodes_list = sorted(part_unwt.keys())
    u_lab = [part_unwt[n] for n in nodes_list]
    t_lab = [part_text[n] for n in nodes_list]
    ari = adjusted_rand_score(u_lab, t_lab)
    print(f"  ARI(unweighted, text-weighted) = {ari:.4f}")

    comm_text_train = compute_community_features(train_pairs, part_text, n_nodes)
    comm_text_test = compute_community_features(test_pairs, part_text, n_nodes)
    print(f"  text same_comm: pos={comm_text_train[y_train==1, 0].mean():.3f}  "
          f"neg={comm_text_train[y_train==0, 0].mean():.3f}")

    cn_text_train = compute_comm_cn(train_pairs, G_text, part_text)
    cn_text_test = compute_comm_cn(test_pairs, G_text, part_text)
    print(f"  text comm_cn: pos={cn_text_train[y_train==1].mean():.3f}  "
          f"neg={cn_text_train[y_train==0].mean():.3f}")

    # Text-weighted spectral embedding (a second spectral view)
    emb_text, emb_text_normed = compute_spectral_embedding(G_text, n_nodes, k=16, seed=SEED)
    spec_text_train = compute_spectral_features(train_pairs, emb_text, emb_text_normed)
    spec_text_test = compute_spectral_features(test_pairs, emb_text, emb_text_normed)

    X_train_v26f = np.hstack([
        X_train_v26d,
        comm_text_train,
        cn_text_train,
        spec_text_train,
    ])
    X_test_v26f = np.hstack([
        X_test_v26d,
        comm_text_test,
        cn_text_test,
        spec_text_test,
    ])
    print(f"  v26f features: {X_train_v26f.shape[1]} (+{X_train_v26f.shape[1] - X_train_v26d.shape[1]})")

    # === CV ===
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v26d = np.zeros(len(y_train), dtype=np.float64)
    oof_v26f = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26f, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v26d[tr], y_train[tr])
        oof_v26d[va] = m1.predict_proba(X_train_v26d[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26f[tr], y_train[tr])
        oof_v26f[va] = m2.predict_proba(X_train_v26f[va])[:, 1]
    auc_v26d = roc_auc_score(y_train, oof_v26d)
    auc_v26f = roc_auc_score(y_train, oof_v26f)
    print(f"  v26d OOF = {auc_v26d:.5f}")
    print(f"  v26f OOF = {auc_v26f:.5f}  ({auc_v26f - auc_v26d:+.5f})")

    # Ablations
    print("\n[ablation] per-family OOF")
    ablations = [
        ("+text same_comm only", np.hstack([X_train_v26d, comm_text_train]),
                                  np.hstack([X_test_v26d, comm_text_test])),
        ("+text comm_cn only",   np.hstack([X_train_v26d, cn_text_train]),
                                  np.hstack([X_test_v26d, cn_text_test])),
        ("+text spectral only",  np.hstack([X_train_v26d, spec_text_train]),
                                  np.hstack([X_test_v26d, spec_text_test])),
    ]
    for name, X_tr, X_te in ablations:
        oof_a = np.zeros(len(y_train), dtype=np.float64)
        for fold, (tr, va) in enumerate(cv.split(X_tr, y_train), 1):
            m = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
            m.fit(X_tr[tr], y_train[tr])
            oof_a[va] = m.predict_proba(X_tr[va])[:, 1]
        auc = roc_auc_score(y_train, oof_a)
        print(f"  {name:30s} OOF = {auc:.5f}  ({auc - auc_v26d:+.5f})")

    # === Final 30-seed ensemble ===
    print("\n[final] 30-seed HGB+CatBoost on v26f")
    pred_h = predict_hgb(X_train_v26f, y_train, X_test_v26f, n_seeds=30)
    pred_c = predict_cat(X_train_v26f, y_train, X_test_v26f, n_seeds=30)
    pred_v26f = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
    save_sub("submission_v26f.csv", pred_v26f)

    v26d_known = pd.read_csv("submission_v26d.csv")["Predicted"].to_numpy()
    rd = normalize_rank(v26d_known)
    corr, _ = spearmanr(rd, pred_v26f)
    print(f"\n  v26f vs submission_v26d.csv spearman = {corr:.5f}")

    save_sub("submission_v26f_blend_50v26d.csv", 0.5 * pred_v26f + 0.5 * rd)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
