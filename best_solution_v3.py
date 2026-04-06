"""
Link Prediction v3 — careful feature additions beyond baseline.

Baseline (16 features, 0.849 Kaggle): degree stats, CN, Jaccard, AA, RA,
    pref attach, same comp, both isolated, raw dot/cosine, keyword Jaccard.

Additions (only safe scalar features):
  - TF-IDF cosine similarity
  - Shared keyword ratio (shared / min keywords)
  - Min/max degree (sorted, removes order dependence)
  - Sorensen index
  - Total keywords per node (nnz_u, nnz_v)
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
EPS = 1e-12


def load_data(
    train_path: Path, test_path: Path, node_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _map_ids(pairs: np.ndarray, id_to_idx: Dict[int, int]) -> np.ndarray:
    mapped = np.empty_like(pairs, dtype=np.int32)
    for i in range(pairs.shape[0]):
        mapped[i, 0] = id_to_idx[int(pairs[i, 0])]
        mapped[i, 1] = id_to_idx[int(pairs[i, 1])]
    return mapped


def build_graph(
    train_pairs: np.ndarray, y_train: np.ndarray, num_nodes: int
) -> Tuple[List[set], np.ndarray, np.ndarray]:
    adjacency: List[set] = [set() for _ in range(num_nodes)]
    for u, v in train_pairs[y_train == 1]:
        if u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)
    degree = np.array([len(n) for n in adjacency], dtype=np.int32)
    comp = _connected_components(adjacency)
    return adjacency, degree, comp


def _connected_components(adjacency: List[set]) -> np.ndarray:
    n = len(adjacency)
    comp = np.full(n, -1, dtype=np.int32)
    cid = 0
    for start in range(n):
        if comp[start] != -1:
            continue
        stack = [start]
        comp[start] = cid
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if comp[nxt] == -1:
                    comp[nxt] = cid
                    stack.append(nxt)
        cid += 1
    return comp


def build_features(
    pairs: np.ndarray,
    adjacency: List[set],
    degree: np.ndarray,
    comp: np.ndarray,
    node_features: np.ndarray,
    node_tfidf: sparse.csr_matrix,
    y: Optional[np.ndarray] = None,
    remove_positive_edge: bool = False,
) -> np.ndarray:
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

    # Common neighbors, Adamic-Adar, Resource Allocation
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
    pref_attach = deg_u_eff * deg_v_eff
    same_comp = (comp[u] == comp[v]).astype(np.float32)
    both_isolated = ((deg_u == 0) & (deg_v == 0)).astype(np.float32)

    # Text features
    fu, fv = node_features[u], node_features[v]
    raw_dot = np.einsum("ij,ij->i", fu, fv).astype(np.float32)
    norm_u = np.linalg.norm(fu, axis=1).astype(np.float32)
    norm_v = np.linalg.norm(fv, axis=1).astype(np.float32)
    raw_cosine = raw_dot / (norm_u * norm_v + EPS)

    nnz_u = (fu > 0).sum(axis=1).astype(np.float32)
    nnz_v = (fv > 0).sum(axis=1).astype(np.float32)
    keyword_union = nnz_u + nnz_v - raw_dot
    keyword_jaccard = raw_dot / np.maximum(keyword_union, 1.0)
    shared_ratio = raw_dot / np.maximum(np.minimum(nnz_u, nnz_v), 1.0)

    # TF-IDF cosine
    tfidf_cosine = (
        np.asarray(node_tfidf[u].multiply(node_tfidf[v]).sum(axis=1))
        .ravel().astype(np.float32)
    )

    return np.column_stack([
        # Original 16 baseline features
        deg_u_eff, deg_v_eff,
        np.abs(deg_u_eff - deg_v_eff),
        deg_u_eff + deg_v_eff,
        np.log1p(deg_u_eff), np.log1p(deg_v_eff),
        cn, jaccard, aa, ra,
        pref_attach, same_comp, both_isolated,
        raw_dot, raw_cosine, keyword_jaccard,
        # Careful additions
        tfidf_cosine,
        shared_ratio,
        sorensen,
        nnz_u, nnz_v,
        np.minimum(deg_u_eff, deg_v_eff),
        np.maximum(deg_u_eff, deg_v_eff),
    ]).astype(np.float32)


def make_hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_depth=4,
        max_iter=250,
        min_samples_leaf=30,
        l2_regularization=0.02,
        random_state=seed,
    )


def main() -> None:
    t0 = time.time()
    np.random.seed(SEED)

    train_pairs, y_train, test_pairs, node_features = load_data(
        Path("train.txt"), Path("test.txt"), Path("node_information.csv")
    )
    n_nodes = node_features.shape[0]
    print(f"[data] train={len(y_train)}  test={len(test_pairs)}  nodes={n_nodes}")

    adjacency, degree, comp = build_graph(train_pairs, y_train, n_nodes)

    nf_sparse = sparse.csr_matrix(node_features)
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    node_tfidf = tfidf.fit_transform(nf_sparse)

    X_train = build_features(
        train_pairs, adjacency, degree, comp, node_features, node_tfidf,
        y=y_train, remove_positive_edge=True,
    )
    X_test = build_features(
        test_pairs, adjacency, degree, comp, node_features, node_tfidf,
    )
    print(f"[features] {X_train.shape[1]} features")

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_train), dtype=np.float64)
    for fold, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
        model = make_hgb(SEED + fold)
        model.fit(X_train[tr], y_train[tr])
        oof[va] = model.predict_proba(X_train[va])[:, 1]
        print(f"  fold {fold}/5  auc={roc_auc_score(y_train[va], oof[va]):.5f}")
    print(f"  OOF AUC={roc_auc_score(y_train, oof):.5f}")

    # Final: 5-seed average
    pred = np.zeros(len(test_pairs), dtype=np.float64)
    for s in range(5):
        model = make_hgb(SEED + s * 100)
        model.fit(X_train, y_train)
        pred += model.predict_proba(X_test)[:, 1]
    pred = np.clip(pred / 5.0, 0.0, 1.0).astype(np.float32)

    out = Path("submission_best_v3.csv")
    pd.DataFrame({"ID": np.arange(len(pred)), "Predicted": pred}).to_csv(out, index=False)
    print(f"[output] {out}  rows={len(pred)}  min={pred.min():.5f}  max={pred.max():.5f}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
