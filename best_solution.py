"""
Link Prediction v16b — New structural & text signals with leakage correction.

Additions over v12 (0.861):
  1. Paths of length 3 (A³) — corrected for edge leakage in training
  2. Neighborhood text similarity — corrected for edge leakage in training
  3. Separated train/test pair counts
  4. Katz-like index
  5. Hub-promoted / hub-depressed indices
  6. Sorensen index
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

SEED = 42
EPS = 1e-12


def load_data(train_path, test_path, node_path):
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=["u", "v", "label"])
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=["u", "v"])
    node_df = pd.read_csv(node_path, header=None)
    node_ids = node_df.iloc[:, 0].astype(np.int32).to_numpy()
    node_features = node_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    train_pairs = _map_ids(train_df[["u", "v"]].to_numpy(), id_to_idx)
    test_pairs = _map_ids(test_df[["u", "v"]].to_numpy(), id_to_idx)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    return train_pairs, y_train, test_pairs, node_features


def _map_ids(pairs, id_to_idx):
    mapped = np.empty_like(pairs, dtype=np.int32)
    for i in range(pairs.shape[0]):
        mapped[i, 0] = id_to_idx[int(pairs[i, 0])]
        mapped[i, 1] = id_to_idx[int(pairs[i, 1])]
    return mapped


def build_graph(train_pairs, y_train, num_nodes, extra_edges=None):
    adjacency = [set() for _ in range(num_nodes)]
    for u, v in train_pairs[y_train == 1]:
        if u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)
    if extra_edges is not None:
        for u, v in extra_edges:
            if u != v:
                adjacency[u].add(v)
                adjacency[v].add(u)
    degree = np.array([len(n) for n in adjacency], dtype=np.int32)
    comp = _cc(adjacency)
    return adjacency, degree, comp


def _cc(adj):
    n = len(adj)
    comp = np.full(n, -1, dtype=np.int32)
    cid = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        stack = [s]
        comp[s] = cid
        while stack:
            nd = stack.pop()
            for nx in adj[nd]:
                if comp[nx] == -1:
                    comp[nx] = cid
                    stack.append(nx)
        cid += 1
    return comp


def build_sparse_adj(adjacency, num_nodes):
    rows, cols = [], []
    for u in range(num_nodes):
        for v in adjacency[u]:
            rows.append(u)
            cols.append(v)
    if not rows:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(num_nodes, num_nodes),
    )


def build_features(
    pairs, adjacency, degree, comp, node_features, node_tfidf,
    total_count, train_count, test_count,
    adj_matrix, neighbor_tfidf,
    y=None, remove_pos=False,
):
    u = pairs[:, 0]
    v = pairs[:, 1]
    n = pairs.shape[0]

    deg_u = degree[u].astype(np.float32)
    deg_v = degree[v].astype(np.float32)
    is_pos = None
    if remove_pos and y is not None:
        is_pos = (y == 1)
        pos_f = is_pos.astype(np.float32)
        deg_u_eff = np.maximum(deg_u - pos_f, 0.0)
        deg_v_eff = np.maximum(deg_v - pos_f, 0.0)
    else:
        deg_u_eff = deg_u
        deg_v_eff = deg_v

    # --- Classic graph heuristics ---
    cn = np.zeros(n, dtype=np.float32)
    aa = np.zeros(n, dtype=np.float32)
    ra = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nu, nv = adjacency[int(u[i])], adjacency[int(v[i])]
        small, large = (nu, nv) if len(nu) <= len(nv) else (nv, nu)
        c, a_s, r_s = 0.0, 0.0, 0.0
        for w in small:
            if w in large:
                c += 1.0
                dw = degree[w]
                if dw > 1:
                    a_s += 1.0 / math.log(dw)
                if dw > 0:
                    r_s += 1.0 / dw
        cn[i], aa[i], ra[i] = c, a_s, r_s

    union_deg = deg_u_eff + deg_v_eff - cn
    jaccard = cn / np.maximum(union_deg, 1.0)
    sorensen = 2.0 * cn / np.maximum(deg_u_eff + deg_v_eff, 1.0)
    pa = deg_u_eff * deg_v_eff
    same_comp = (comp[u] == comp[v]).astype(np.float32)
    both_iso = ((deg_u == 0) & (deg_v == 0)).astype(np.float32)

    # --- Text features ---
    fu, fv = node_features[u], node_features[v]
    raw_dot = np.einsum("ij,ij->i", fu, fv).astype(np.float32)
    norm_u = np.linalg.norm(fu, axis=1).astype(np.float32)
    norm_v = np.linalg.norm(fv, axis=1).astype(np.float32)
    raw_cosine = raw_dot / (norm_u * norm_v + EPS)

    nnz_u = (fu > 0).sum(axis=1).astype(np.float32)
    nnz_v = (fv > 0).sum(axis=1).astype(np.float32)
    keyword_union = nnz_u + nnz_v - raw_dot
    keyword_jaccard = raw_dot / np.maximum(keyword_union, 1.0)

    tfidf_cosine = np.asarray(
        node_tfidf[u].multiply(node_tfidf[v]).sum(axis=1)
    ).ravel().astype(np.float32)
    diff_tfidf = node_tfidf[u] - node_tfidf[v]
    tfidf_l2 = np.sqrt(
        np.asarray(diff_tfidf.multiply(diff_tfidf).sum(axis=1)).ravel()
    ).astype(np.float32)

    overlap_u = raw_dot / np.maximum(nnz_u, 1.0)
    overlap_v = raw_dot / np.maximum(nnz_v, 1.0)

    # --- Transductive pair counts ---
    tc_u = total_count[u]
    tc_v = total_count[v]
    tec_u = test_count[u]
    tec_v = test_count[v]

    # --- Paths of length 3 (A³) with leakage correction ---
    # A³[u,v] counts all walks of length 3. For positive training pairs,
    # the direct edge u-v contributes deg(u) + deg(v) - 1 spurious walks.
    A2 = adj_matrix @ adj_matrix
    A3 = adj_matrix @ A2
    paths3 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        paths3[i] = A3[int(u[i]), int(v[i])]
    if is_pos is not None:
        # Subtract leakage for positive training pairs
        leakage = (deg_u + deg_v - 1.0) * is_pos.astype(np.float32)
        paths3 = np.maximum(paths3 - leakage, 0.0)

    # Katz-like: β²·CN + β³·paths3
    beta = 0.005
    katz = (beta ** 2) * cn + (beta ** 3) * paths3

    # --- Neighborhood text similarity with leakage correction ---
    # neighbor_tfidf[u] = (1/deg(u)) * Σ_{w ∈ N(u)} tfidf(w)
    # For positive pair (u,v): v ∈ N(u), so we correct by removing v's contribution
    ntfidf_u = neighbor_tfidf[u]  # sparse (n, d)
    ntfidf_v = neighbor_tfidf[v]  # sparse (n, d)
    tfidf_u_sp = node_tfidf[u]
    tfidf_v_sp = node_tfidf[v]

    if is_pos is not None:
        # For positive pairs: corrected_ntfidf_u = (d_u * ntfidf_u - tfidf_v) / max(d_u - 1, 1)
        # Build corrected versions only for positive pairs
        d_u_arr = deg_u.reshape(-1, 1)  # (n, 1)
        d_v_arr = deg_v.reshape(-1, 1)

        # Convert to dense for correction (only n pairs × d features)
        ntfidf_u_dense = np.asarray(ntfidf_u.todense())
        ntfidf_v_dense = np.asarray(ntfidf_v.todense())
        tfidf_u_dense = np.asarray(tfidf_u_sp.todense())
        tfidf_v_dense = np.asarray(tfidf_v_sp.todense())

        pos_mask = is_pos.astype(np.float32).reshape(-1, 1)

        # Corrected neighbor tfidf for u (removing v's contribution for positive pairs)
        corr_ntfidf_u = ntfidf_u_dense.copy()
        corr_denom_u = np.maximum(d_u_arr - 1.0, 1.0)
        corr_ntfidf_u = np.where(
            pos_mask > 0,
            (d_u_arr * ntfidf_u_dense - tfidf_v_dense) / corr_denom_u,
            ntfidf_u_dense,
        )
        # Corrected neighbor tfidf for v (removing u's contribution for positive pairs)
        corr_ntfidf_v = ntfidf_v_dense.copy()
        corr_denom_v = np.maximum(d_v_arr - 1.0, 1.0)
        corr_ntfidf_v = np.where(
            pos_mask > 0,
            (d_v_arr * ntfidf_v_dense - tfidf_u_dense) / corr_denom_v,
            ntfidf_v_dense,
        )

        # cos(corrected_ntfidf_u, tfidf_v)
        neigh_text_uv = np.einsum("ij,ij->i", corr_ntfidf_u, tfidf_v_dense).astype(np.float32)
        neigh_text_vu = np.einsum("ij,ij->i", corr_ntfidf_v, tfidf_u_dense).astype(np.float32)
        neigh_text_nn = np.einsum("ij,ij->i", corr_ntfidf_u, corr_ntfidf_v).astype(np.float32)
    else:
        # No correction needed for test pairs
        neigh_text_uv = np.asarray(
            ntfidf_u.multiply(tfidf_v_sp).sum(axis=1)
        ).ravel().astype(np.float32)
        neigh_text_vu = np.asarray(
            ntfidf_v.multiply(tfidf_u_sp).sum(axis=1)
        ).ravel().astype(np.float32)
        neigh_text_nn = np.asarray(
            ntfidf_u.multiply(ntfidf_v).sum(axis=1)
        ).ravel().astype(np.float32)

    # --- Hub indices ---
    min_deg = np.minimum(deg_u_eff, deg_v_eff)
    max_deg = np.maximum(deg_u_eff, deg_v_eff)
    hub_promoted = cn / np.maximum(min_deg, 1.0)
    hub_depressed = cn / np.maximum(max_deg, 1.0)

    return np.column_stack([
        # v4 base (20 features)
        deg_u_eff, deg_v_eff,
        np.abs(deg_u_eff - deg_v_eff), deg_u_eff + deg_v_eff,
        np.log1p(deg_u_eff), np.log1p(deg_v_eff),
        cn, jaccard, aa, ra,
        pa, same_comp, both_iso,
        raw_dot, raw_cosine, keyword_jaccard,
        tfidf_cosine, tfidf_l2,
        overlap_u, overlap_v,
        # v12 transductive (4 features)
        tc_u, tc_v, tc_u * tc_v, np.abs(tc_u - tc_v),
        # v16 NEW features
        tec_u, tec_v,                       # test-only pair counts
        paths3,                             # paths of length 3 (corrected)
        katz,                               # Katz-like index
        neigh_text_uv, neigh_text_vu,       # neighbor text similarity (corrected)
        neigh_text_nn,                      # neighborhood-to-neighborhood
        hub_promoted, hub_depressed,        # hub indices
        sorensen,                           # Sorensen index
        min_deg, max_deg,                   # sorted degrees
    ]).astype(np.float32)


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

    # Transductive pair counts
    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in train_pairs:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in test_pairs:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    # --- Self-training round (threshold 0.95) ---
    adj0, deg0, comp0 = build_graph(train_pairs, y_train, n_nodes)
    A0 = build_sparse_adj(adj0, n_nodes)
    d_inv0 = np.zeros(n_nodes, dtype=np.float32)
    mask0 = deg0 > 0
    d_inv0[mask0] = 1.0 / deg0[mask0]
    ntfidf0 = sparse.diags(d_inv0) @ A0 @ node_tfidf

    X_tr0 = build_features(
        train_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count,
        A0, ntfidf0,
        y=y_train, remove_pos=True,
    )
    X_te0 = build_features(
        test_pairs, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count,
        A0, ntfidf0,
    )
    print(f"[features] {X_tr0.shape[1]} features")

    pred_init = np.zeros(len(test_pairs), dtype=np.float64)
    for s in range(5):
        m = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=400,
            min_samples_leaf=40, l2_regularization=0.1,
            random_state=SEED + s * 77,
        )
        m.fit(X_tr0, y_train)
        pred_init += m.predict_proba(X_te0)[:, 1]
    pred_init /= 5

    extra_edges = test_pairs[pred_init >= 0.95]
    print(f"[self-train] +{len(extra_edges)} pseudo-edges")

    # --- Rebuild graph with pseudo-edges ---
    adjacency, degree, comp = build_graph(train_pairs, y_train, n_nodes, extra_edges)
    adj_matrix = build_sparse_adj(adjacency, n_nodes)
    d_inv = np.zeros(n_nodes, dtype=np.float32)
    mask = degree > 0
    d_inv[mask] = 1.0 / degree[mask]
    neighbor_tfidf = sparse.diags(d_inv) @ adj_matrix @ node_tfidf

    X_train = build_features(
        train_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
        y=y_train, remove_pos=True,
    )
    X_test = build_features(
        test_pairs, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count,
        adj_matrix, neighbor_tfidf,
    )
    print(f"[final features] {X_train.shape[1]}")

    # --- CV ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
        m = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=400,
            min_samples_leaf=40, l2_regularization=0.1,
            random_state=SEED + fold,
        )
        m.fit(X_train[tr], y_train[tr])
        oof[va] = m.predict_proba(X_train[va])[:, 1]
        print(f"  fold {fold}/5 AUC={roc_auc_score(y_train[va], oof[va]):.5f}")
    print(f"  OOF AUC={roc_auc_score(y_train, oof):.5f}")

    # --- Final predictions: HGB 15-seed ---
    n_seeds = 15
    pred_hgb = np.zeros(len(test_pairs), dtype=np.float64)
    for s in range(n_seeds):
        m = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=400,
            min_samples_leaf=40, l2_regularization=0.1,
            random_state=SEED + s * 31,
        )
        m.fit(X_train, y_train)
        pred_hgb += m.predict_proba(X_test)[:, 1]
    pred_hgb /= n_seeds

    # Also do CatBoost blend if available
    if HAS_CATBOOST:
        pred_cat = np.zeros(len(test_pairs), dtype=np.float64)
        for s in range(n_seeds):
            m = CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=3,
                l2_leaf_reg=10, random_seed=SEED + s * 31, verbose=0,
            )
            m.fit(X_train, y_train)
            pred_cat += m.predict_proba(X_test)[:, 1]
        pred_cat /= n_seeds
        pred_blend = np.clip(0.5 * pred_hgb + 0.5 * pred_cat, 0, 1).astype(np.float32)
        out2 = Path("submission_v16b_blend.csv")
        pd.DataFrame({"ID": np.arange(len(pred_blend)), "Predicted": pred_blend}).to_csv(out2, index=False)
        print(f"[output] {out2}  min={pred_blend.min():.5f} max={pred_blend.max():.5f}")

    pred_hgb = np.clip(pred_hgb, 0, 1).astype(np.float32)
    out1 = Path("submission_v16b_hgb.csv")
    pd.DataFrame({"ID": np.arange(len(pred_hgb)), "Predicted": pred_hgb}).to_csv(out1, index=False)
    print(f"[output] {out1}  min={pred_hgb.min():.5f} max={pred_hgb.max():.5f}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
