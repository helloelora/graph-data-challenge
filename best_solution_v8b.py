"""
v8b — Same iterative self-training but with the EXACT v4 feature set
(the one that scored 0.851 on Kaggle). No SVD, no symmetry changes.
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold

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
    if extra_edges is not None and len(extra_edges) > 0:
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


def build_features(pairs, adjacency, degree, comp, node_features, node_tfidf,
                    y=None, remove_positive_edge=False):
    u = pairs[:, 0]
    v = pairs[:, 1]
    n = pairs.shape[0]

    deg_u = degree[u].astype(np.float32)
    deg_v = degree[v].astype(np.float32)
    if remove_positive_edge and y is not None:
        pos = (y == 1).astype(np.float32)
        deg_u_eff = np.maximum(deg_u - pos, 0.0)
        deg_v_eff = np.maximum(deg_v - pos, 0.0)
    else:
        deg_u_eff = deg_u
        deg_v_eff = deg_v

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
    pref_attach = deg_u_eff * deg_v_eff
    same_comp = (comp[u] == comp[v]).astype(np.float32)
    both_isolated = ((deg_u == 0) & (deg_v == 0)).astype(np.float32)

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

    diff = node_tfidf[u] - node_tfidf[v]
    tfidf_l2 = np.sqrt(np.asarray(diff.multiply(diff).sum(axis=1)).ravel()).astype(np.float32)

    overlap_u = raw_dot / np.maximum(nnz_u, 1.0)
    overlap_v = raw_dot / np.maximum(nnz_v, 1.0)

    return np.column_stack([
        deg_u_eff, deg_v_eff,
        np.abs(deg_u_eff - deg_v_eff),
        deg_u_eff + deg_v_eff,
        np.log1p(deg_u_eff), np.log1p(deg_v_eff),
        cn, jaccard, aa, ra,
        pref_attach, same_comp, both_isolated,
        raw_dot, raw_cosine, keyword_jaccard,
        tfidf_cosine, tfidf_l2,
        overlap_u, overlap_v,
    ]).astype(np.float32)


def make_hgb(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=3, max_iter=400,
        min_samples_leaf=40, l2_regularization=0.1,
        random_state=seed,
    )


def predict_ensemble(X_train, y_train, X_test, n_seeds=5):
    pred = np.zeros(X_test.shape[0], dtype=np.float64)
    for s in range(n_seeds):
        m = make_hgb(SEED + s * 77)
        m.fit(X_train, y_train)
        pred += m.predict_proba(X_test)[:, 1]
    return pred / n_seeds


def main():
    t0 = time.time()
    np.random.seed(SEED)

    train_pairs, y_train, test_pairs, node_features = load_data(
        Path("train.txt"), Path("test.txt"), Path("node_information.csv")
    )
    n_nodes = node_features.shape[0]
    nf_sparse = sparse.csr_matrix(node_features)
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    node_tfidf = tfidf.fit_transform(nf_sparse)

    # Iterative self-training
    thresholds = [0.90, 0.85, 0.80]
    extra_edges = None

    for it, thr in enumerate(thresholds):
        print(f"\n=== Iteration {it} (threshold={thr}) ===")
        adj, deg, comp = build_graph(train_pairs, y_train, n_nodes, extra_edges)
        n_edges = sum(len(n) for n in adj) // 2
        print(f"[graph] edges={n_edges}  mean_deg={deg.mean():.2f}")

        X_train = build_features(
            train_pairs, adj, deg, comp, node_features, node_tfidf,
            y=y_train, remove_positive_edge=True,
        )
        X_test = build_features(
            test_pairs, adj, deg, comp, node_features, node_tfidf,
        )

        test_pred = predict_ensemble(X_train, y_train, X_test)
        confident = test_pred >= thr
        print(f"[augment] {confident.sum()} pairs above {thr}")

        if confident.sum() > 0:
            extra_edges = test_pairs[confident]
        else:
            break

    # Final prediction with augmented graph
    print(f"\n=== Final ===")
    adj, deg, comp = build_graph(train_pairs, y_train, n_nodes, extra_edges)
    print(f"[graph] edges={sum(len(n) for n in adj) // 2}  mean_deg={deg.mean():.2f}")

    X_train = build_features(
        train_pairs, adj, deg, comp, node_features, node_tfidf,
        y=y_train, remove_positive_edge=True,
    )
    X_test = build_features(
        test_pairs, adj, deg, comp, node_features, node_tfidf,
    )

    pred = np.zeros(len(test_pairs), dtype=np.float64)
    n_seeds = 7
    for s in range(n_seeds):
        m = make_hgb(SEED + s * 77)
        m.fit(X_train, y_train)
        pred += m.predict_proba(X_test)[:, 1]
    pred = np.clip(pred / n_seeds, 0.0, 1.0).astype(np.float32)

    out = Path("submission_best_v8b.csv")
    pd.DataFrame({"ID": np.arange(len(pred)), "Predicted": pred}).to_csv(out, index=False)
    print(f"[output] {out}  min={pred.min():.5f}  max={pred.max():.5f}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
