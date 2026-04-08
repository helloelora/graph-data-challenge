"""
v26b — community + spectral features on a label-free candidate graph.
Kaggle public: 0.88038 (1st place, +0.00830 over v25).

v25 saturated at 0.87208 and every "more seeds / another HP / different
booster" perturbation (v21-v23) cost ~0.001 Kaggle. The only prior jumps
had come from fundamentally new *information sources* (node-level
transductive counts in v19, pair-level intersections in v24/v25). v26b
continues in that direction: global community structure + spectral
geometry on a graph that includes the test pair list itself as a
structural signal.

The "candidate graph" contains:
  - ALL train pairs (regardless of 0/1 label) as edges        w = 1.0
  - ALL test pairs as candidate edges                         w = 1.0
  - Self-training pseudo-edges (from v25 at threshold 0.95)   w = 1.0

Leakage-free by construction for training pairs:
  - Train labels are NEVER used to decide which edges go in the graph
  - Both positive and negative train pairs contribute equal edges
  - For a held-out train pair (u,v), the direct edge is in the graph
    whether y=0 or y=1, so same_community cannot trivially memorize the
    label
  - For test pairs, the direct test candidate edge is always in the
    graph — this is the same transductive trick used in v24/v25

The signal comes from *surrounding* community structure: if u and v
share many other partners in the candidate graph, Louvain clusters them
together beyond the trivial effect of the direct edge. This generalizes
v24's shared_test_partners — instead of a pairwise set intersection we
get a full partition of the node set.

We also add a Laplacian eigenmap (k=16) of the same graph and turn it
into pair-level L2 distance, dot product and cosine features (inspired
by Kunegis & Lommatzsch, ICML 2009).

Feature additions on top of v25 (56):
  community (3) : same_community, community_size_min, community_size_max
  spectral  (3) : spectral_dist_l2, spectral_dot, spectral_cos
  -> v26b 62 features total
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
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

# python-louvain (PyPI: python-louvain) installs the top-level `community`
# package. There is a DIFFERENT unrelated PyPI package also called `community`,
# so we import defensively: prefer the submodule path which only python-louvain
# exposes, then fall back to the top-level module.
try:
    from community import community_louvain
except (ImportError, AttributeError):
    import community as community_louvain
import networkx as nx

if not hasattr(community_louvain, "best_partition"):
    raise ImportError(
        "community module has no best_partition() — the wrong `community` "
        "package is installed. Run: pip uninstall -y community && "
        "pip install --force-reinstall python-louvain"
    )


def build_candidate_graph(train_pairs, test_pairs, extra_edges, n_nodes):
    """Label-FREE undirected graph.

    Contains every pair that appears in train.txt OR test.txt, plus the
    v25 self-training pseudo-edges. The y_train labels are NEVER used,
    so this is leakage-free by construction for training pairs.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # ALL train pairs regardless of label — this is the key leakage fix.
    for u, v in train_pairs:
        u, v = int(u), int(v)
        if u != v:
            G.add_edge(u, v, weight=1.0)

    # ALL test pairs as candidate edges
    for u, v in test_pairs:
        u, v = int(u), int(v)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, weight=1.0)

    # Self-training pseudo-edges (derived from model predictions, not labels)
    if extra_edges is not None:
        for u, v in extra_edges:
            u, v = int(u), int(v)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v, weight=1.0)

    return G


def run_louvain(G, seed=SEED):
    return community_louvain.best_partition(G, weight="weight", random_state=seed)


def compute_community_features(pairs, partition, n_nodes):
    sizes = {}
    for node, cid in partition.items():
        sizes[cid] = sizes.get(cid, 0) + 1

    n = pairs.shape[0]
    same_comm = np.zeros(n, dtype=np.float32)
    size_min = np.zeros(n, dtype=np.float32)
    size_max = np.zeros(n, dtype=np.float32)

    for i in range(n):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        cu = partition.get(u, -1)
        cv = partition.get(v, -2)
        same_comm[i] = 1.0 if cu == cv else 0.0
        su = sizes.get(cu, 1)
        sv = sizes.get(cv, 1)
        size_min[i] = min(su, sv)
        size_max[i] = max(su, sv)

    return np.column_stack([same_comm, size_min, size_max]).astype(np.float32)


def compute_spectral_embedding(G, n_nodes, k=16, seed=SEED):
    nodes = list(range(n_nodes))
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr", dtype=np.float64)
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(d)
    nz = d > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L_norm = sparse.eye(n_nodes, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

    try:
        vals, vecs = eigsh(L_norm, k=k + 1, sigma=0, which="LM")
    except Exception:
        vals, vecs = eigsh(L_norm, k=k + 1, which="SM")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    emb = vecs[:, 1 : k + 1]

    norms = np.linalg.norm(emb, axis=1, keepdims=True) + EPS
    emb_normed = emb / norms
    return emb, emb_normed


def compute_spectral_features(pairs, emb, emb_normed):
    u = pairs[:, 0].astype(np.int64)
    v = pairs[:, 1].astype(np.int64)
    diff = emb[u] - emb[v]
    dist_l2 = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
    dot = np.einsum("ij,ij->i", emb[u], emb[v]).astype(np.float32)
    cos = np.einsum("ij,ij->i", emb_normed[u], emb_normed[v]).astype(np.float32)
    return np.column_stack([dist_l2, dot, cos]).astype(np.float32)


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

    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in train_pairs:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in test_pairs:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    # === Pair transductive (v24 + v25) ===
    print("[v25] pair-level transductive features")
    train_partners, test_partners = build_partner_sets(train_pairs, test_pairs, n_nodes)
    pair_v24_train = compute_pair_transductive_v24(train_pairs, train_partners, test_partners)
    pair_v24_test = compute_pair_transductive_v24(test_pairs, train_partners, test_partners)
    pair_v25_train = compute_pair_transductive_v25(
        train_pairs, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_test = compute_pair_transductive_v25(
        test_pairs, train_partners, test_partners, test_count, train_count, total_count)

    # === v19 pipeline with self-training ===
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

    # === v26b NEW: community + spectral on LABEL-FREE candidate graph ===
    print("\n[v26b] label-free candidate graph")
    G_cand = build_candidate_graph(train_pairs, test_pairs, extra_edges, n_nodes)
    print(f"  |V|={G_cand.number_of_nodes()}  |E|={G_cand.number_of_edges()}  "
          f"components={nx.number_connected_components(G_cand)}")

    print("[v26b] Louvain")
    partition = run_louvain(G_cand, seed=SEED)
    n_comms = len(set(partition.values()))
    print(f"  found {n_comms} communities")
    comm_train = compute_community_features(train_pairs, partition, n_nodes)
    comm_test = compute_community_features(test_pairs, partition, n_nodes)
    print(f"  train same_comm:  pos={comm_train[y_train==1,0].mean():.3f}  "
          f"neg={comm_train[y_train==0,0].mean():.3f}")
    print(f"  test  same_comm:  {comm_test[:,0].mean():.3f}")

    print("[v26b] spectral embedding k=16")
    emb, emb_normed = compute_spectral_embedding(G_cand, n_nodes, k=16, seed=SEED)
    spec_train = compute_spectral_features(train_pairs, emb, emb_normed)
    spec_test = compute_spectral_features(test_pairs, emb, emb_normed)
    print(f"  train spectral_cos: pos={spec_train[y_train==1,2].mean():.3f}  "
          f"neg={spec_train[y_train==0,2].mean():.3f}")
    print(f"  train spectral_dist_l2: pos={spec_train[y_train==1,0].mean():.3f}  "
          f"neg={spec_train[y_train==0,0].mean():.3f}")

    X_train_v26 = np.hstack([X_train_v25, comm_train, spec_train])
    X_test_v26 = np.hstack([X_test_v25, comm_test, spec_test])
    print(f"  v26b features: {X_train_v26.shape[1]} (+{X_train_v26.shape[1] - X_train_v25.shape[1]})")

    # === CV comparison ===
    print("\n[CV] OOF AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_v25 = np.zeros(len(y_train), dtype=np.float64)
    oof_v26 = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train_v26, y_train), 1):
        m1 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m1.fit(X_train_v25[tr], y_train[tr])
        oof_v25[va] = m1.predict_proba(X_train_v25[va])[:, 1]
        m2 = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
        m2.fit(X_train_v26[tr], y_train[tr])
        oof_v26[va] = m2.predict_proba(X_train_v26[va])[:, 1]
    auc_v25 = roc_auc_score(y_train, oof_v25)
    auc_v26 = roc_auc_score(y_train, oof_v26)
    print(f"  v25  OOF AUC = {auc_v25:.5f}")
    print(f"  v26b OOF AUC = {auc_v26:.5f}  ({auc_v26 - auc_v25:+.5f})")

    if auc_v26 <= auc_v25:
        print("\n  [WARN] v26b did not beat v25 on OOF. Still saving for diagnostic.")

    # Ablations: community-only vs spectral-only vs both
    print("\n[ablation] OOF per added family")
    for name, X_tr, X_te in [
        ("v25 + community only", np.hstack([X_train_v25, comm_train]),
                                  np.hstack([X_test_v25, comm_test])),
        ("v25 + spectral only",  np.hstack([X_train_v25, spec_train]),
                                  np.hstack([X_test_v25, spec_test])),
    ]:
        oof_a = np.zeros(len(y_train), dtype=np.float64)
        for fold, (tr, va) in enumerate(cv.split(X_tr, y_train), 1):
            m = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED + fold)
            m.fit(X_tr[tr], y_train[tr])
            oof_a[va] = m.predict_proba(X_tr[va])[:, 1]
        auc = roc_auc_score(y_train, oof_a)
        print(f"  {name:30s} OOF = {auc:.5f}  ({auc - auc_v25:+.5f})")

    # === Final 30-seed HGB+CatBoost ===
    print("\n[final] 30-seed HGB+CatBoost on v26b")
    pred_h = predict_hgb(X_train_v26, y_train, X_test_v26, n_seeds=30)
    pred_c = predict_cat(X_train_v26, y_train, X_test_v26, n_seeds=30)
    pred_v26 = 0.5 * normalize_rank(pred_h) + 0.5 * normalize_rank(pred_c)
    save_sub("submission_v26b.csv", pred_v26)

    print("[ref] 30-seed HGB+CatBoost on v25")
    pred_h25 = predict_hgb(X_train_v25, y_train, X_test_v25, n_seeds=30)
    pred_c25 = predict_cat(X_train_v25, y_train, X_test_v25, n_seeds=30)
    pred_v25_ref = 0.5 * normalize_rank(pred_h25) + 0.5 * normalize_rank(pred_c25)
    save_sub("submission_v26b_v25ref.csv", pred_v25_ref)

    # Blends
    save_sub("submission_v26b_blend_50.csv", 0.5 * pred_v26 + 0.5 * pred_v25_ref)
    save_sub("submission_v26b_blend_70v26.csv", 0.7 * pred_v26 + 0.3 * pred_v25_ref)
    save_sub("submission_v26b_blend_30v26.csv", 0.3 * pred_v26 + 0.7 * pred_v25_ref)

    corr, _ = spearmanr(pred_v25_ref, pred_v26)
    print(f"\n  v26b vs v25_ref spearman = {corr:.5f}")

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
