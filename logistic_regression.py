import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple trained baseline for link prediction (LR + handcrafted features)."
    )
    parser.add_argument("--train", default="train.txt", help="Path to train file.")
    parser.add_argument("--test", default="test.txt", help="Path to test file.")
    parser.add_argument(
        "--node-info", default="node_information.csv", help="Path to node feature file."
    )
    parser.add_argument(
        "--output",
        default="submission_lr_baseline.csv",
        help="Output submission CSV path.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_data(
    train_path: Path, test_path: Path, node_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=["u", "v", "label"])
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=["u", "v"])
    node_df = pd.read_csv(node_path, header=None)

    node_ids = node_df.iloc[:, 0].astype(np.int32).to_numpy()
    node_features = node_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    train_pairs = map_ids(train_df[["u", "v"]].to_numpy(), id_to_idx)
    test_pairs = map_ids(test_df[["u", "v"]].to_numpy(), id_to_idx)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    return train_pairs, y_train, test_pairs, node_features


def map_ids(pairs_with_ids: np.ndarray, id_to_idx: Dict[int, int]) -> np.ndarray:
    pairs = np.zeros_like(pairs_with_ids, dtype=np.int32)
    for i in range(pairs_with_ids.shape[0]):
        u = int(pairs_with_ids[i, 0])
        v = int(pairs_with_ids[i, 1])
        pairs[i, 0] = id_to_idx[u]
        pairs[i, 1] = id_to_idx[v]
    return pairs


def build_graph(
    train_pairs: np.ndarray, y_train: np.ndarray, num_nodes: int
) -> Tuple[List[set], np.ndarray, np.ndarray]:
    adjacency = [set() for _ in range(num_nodes)]
    for u, v in train_pairs[y_train == 1]:
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    degree = np.array([len(neigh) for neigh in adjacency], dtype=np.int32)
    components = connected_components(adjacency)
    return adjacency, degree, components


def connected_components(adjacency: List[set]) -> np.ndarray:
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
    components: np.ndarray,
    node_features: np.ndarray,
    y: np.ndarray = None,
    remove_positive_edge: bool = False,
) -> np.ndarray:
    u = pairs[:, 0]
    v = pairs[:, 1]
    n_pairs = pairs.shape[0]

    deg_u = degree[u].astype(np.float32)
    deg_v = degree[v].astype(np.float32)
    if remove_positive_edge and y is not None:
        pos_mask = (y == 1).astype(np.float32)
        deg_u_eff = np.maximum(deg_u - pos_mask, 0.0)
        deg_v_eff = np.maximum(deg_v - pos_mask, 0.0)
    else:
        deg_u_eff = deg_u
        deg_v_eff = deg_v

    common_neighbors = np.zeros(n_pairs, dtype=np.float32)
    adamic_adar = np.zeros(n_pairs, dtype=np.float32)
    resource_allocation = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        ui = int(u[i])
        vi = int(v[i])
        neigh_u = adjacency[ui]
        neigh_v = adjacency[vi]
        if len(neigh_u) <= len(neigh_v):
            small, large = neigh_u, neigh_v
        else:
            small, large = neigh_v, neigh_u

        cn = 0.0
        aa = 0.0
        ra = 0.0
        for w in small:
            if w not in large:
                continue
            cn += 1.0
            dw = degree[w]
            if dw > 1:
                aa += 1.0 / math.log(dw)
            if dw > 0:
                ra += 1.0 / dw
        common_neighbors[i] = cn
        adamic_adar[i] = aa
        resource_allocation[i] = ra

    union_graph = deg_u_eff + deg_v_eff - common_neighbors
    jaccard_graph = common_neighbors / np.maximum(union_graph, 1.0)
    preferential_attachment = deg_u_eff * deg_v_eff
    same_component = (components[u] == components[v]).astype(np.float32)
    both_isolated = ((deg_u == 0) & (deg_v == 0)).astype(np.float32)

    feat_u = node_features[u]
    feat_v = node_features[v]
    dot = np.einsum("ij,ij->i", feat_u, feat_v).astype(np.float32)
    norm_u = np.linalg.norm(feat_u, axis=1).astype(np.float32)
    norm_v = np.linalg.norm(feat_v, axis=1).astype(np.float32)
    cosine = dot / (norm_u * norm_v + EPS)

    nnz_u = (feat_u > 0).sum(axis=1).astype(np.float32)
    nnz_v = (feat_v > 0).sum(axis=1).astype(np.float32)
    keyword_union = nnz_u + nnz_v - dot
    keyword_jaccard = dot / np.maximum(keyword_union, 1.0)

    features = np.column_stack(
        [
            deg_u_eff,
            deg_v_eff,
            np.abs(deg_u_eff - deg_v_eff),
            deg_u_eff + deg_v_eff,
            np.log1p(deg_u_eff),
            np.log1p(deg_v_eff),
            common_neighbors,
            jaccard_graph,
            adamic_adar,
            resource_allocation,
            preferential_attachment,
            same_component,
            both_isolated,
            dot,
            cosine,
            keyword_jaccard,
        ]
    ).astype(np.float32)

    return features


def get_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.7,
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        ),
    )


def cross_validate(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=np.float32)
    fold_aucs = []
    fold_aps = []

    for i, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        model = get_model(seed + i)
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict_proba(X[va_idx])[:, 1]
        oof[va_idx] = pred

        auc = roc_auc_score(y[va_idx], pred)
        ap = average_precision_score(y[va_idx], pred)
        fold_aucs.append(auc)
        fold_aps.append(ap)
        print(f"[cv] fold={i}/{folds} auc={auc:.5f} ap={ap:.5f}")

    auc_oof = roc_auc_score(y, oof)
    ap_oof = average_precision_score(y, oof)
    print(
        f"[cv] auc_oof={auc_oof:.5f} ap_oof={ap_oof:.5f} "
        f"auc_mean={np.mean(fold_aucs):.5f} +- {np.std(fold_aucs):.5f}"
    )
    return oof


def save_submission(path: Path, pred: np.ndarray) -> None:
    pd.DataFrame({"ID": np.arange(pred.shape[0]), "Predicted": pred}).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    train_pairs, y_train, test_pairs, node_features = load_data(
        Path(args.train), Path(args.test), Path(args.node_info)
    )
    print(
        f"[data] train={train_pairs.shape[0]} test={test_pairs.shape[0]} "
        f"nodes={node_features.shape[0]} feats={node_features.shape[1]}"
    )

    adjacency, degree, components = build_graph(train_pairs, y_train, node_features.shape[0])
    print(f"[graph] positive_edges={int(y_train.sum())} mean_degree={degree.mean():.2f}")

    X_train = build_features(
        train_pairs,
        adjacency,
        degree,
        components,
        node_features,
        y=y_train,
        remove_positive_edge=True,
    )
    X_test = build_features(test_pairs, adjacency, degree, components, node_features)
    print(f"[features] train_shape={X_train.shape} test_shape={X_test.shape}")

    if args.cv_folds >= 2:
        cross_validate(X_train, y_train, args.cv_folds, args.seed)

    model = get_model(args.seed)
    model.fit(X_train, y_train)
    test_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)

    save_submission(Path(args.output), test_pred)
    print(
        f"[output] saved={args.output} rows={test_pred.shape[0]} "
        f"min={test_pred.min():.5f} max={test_pred.max():.5f}"
    )


if __name__ == "__main__":
    main()
