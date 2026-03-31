import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

EPS = 1e-12


@dataclass
class FeatureContext:
    adjacency: List[set]
    degree: np.ndarray
    component_id: np.ndarray
    node_features: np.ndarray
    node_nonzero: np.ndarray
    node_norm: np.ndarray
    node_tfidf: sparse.csr_matrix
    text_embedding: np.ndarray
    text_embedding_norm: np.ndarray
    graph_embedding: np.ndarray
    graph_embedding_norm: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ensemble for link prediction and create a Kaggle submission."
    )
    parser.add_argument("--train", default="train.txt", help="Path to train file.")
    parser.add_argument("--test", default="test.txt", help="Path to test file.")
    parser.add_argument(
        "--node-info", default="node_information.csv", help="Path to node feature file."
    )
    parser.add_argument(
        "--output", default="submission_best.csv", help="Path to output submission CSV."
    )
    parser.add_argument(
        "--text-dim", type=int, default=64, help="Number of text SVD components."
    )
    parser.add_argument(
        "--graph-dim", type=int, default=64, help="Number of graph SVD components."
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of CV folds used for blend weights."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--skip-xgb",
        action="store_true",
        help="Disable XGBoost even if it is installed.",
    )
    parser.add_argument(
        "--binary-output",
        action="store_true",
        help="Also export a thresholded 0/1 submission with suffix _binary.csv.",
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.5,
        help="Threshold used for binary output.",
    )
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

    train_pairs = map_pair_ids_to_indices(train_df[["u", "v"]].to_numpy(), id_to_idx)
    test_pairs = map_pair_ids_to_indices(test_df[["u", "v"]].to_numpy(), id_to_idx)
    y_train = train_df["label"].to_numpy(dtype=np.int32)

    return train_pairs, y_train, test_pairs, node_features


def map_pair_ids_to_indices(
    pairs_with_ids: np.ndarray, id_to_idx: Dict[int, int]
) -> np.ndarray:
    mapped = np.zeros_like(pairs_with_ids, dtype=np.int32)
    for i in range(pairs_with_ids.shape[0]):
        u = int(pairs_with_ids[i, 0])
        v = int(pairs_with_ids[i, 1])
        if u not in id_to_idx or v not in id_to_idx:
            raise ValueError(f"Node id not found in node_information.csv: ({u}, {v})")
        mapped[i, 0] = id_to_idx[u]
        mapped[i, 1] = id_to_idx[v]
    return mapped


def build_graph(
    train_pairs: np.ndarray, y_train: np.ndarray, num_nodes: int
) -> Tuple[List[set], np.ndarray, np.ndarray, sparse.csr_matrix]:
    adjacency = [set() for _ in range(num_nodes)]
    positive_pairs = train_pairs[y_train == 1]

    for u, v in positive_pairs:
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    degree = np.array([len(neigh) for neigh in adjacency], dtype=np.int32)
    component_id = connected_components(adjacency)
    adjacency_matrix = adjacency_to_sparse_matrix(adjacency, num_nodes)
    return adjacency, degree, component_id, adjacency_matrix


def connected_components(adjacency: List[set]) -> np.ndarray:
    num_nodes = len(adjacency)
    component_id = np.full(num_nodes, -1, dtype=np.int32)
    current_component = 0

    for start in range(num_nodes):
        if component_id[start] != -1:
            continue

        stack = [start]
        component_id[start] = current_component

        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if component_id[nxt] == -1:
                    component_id[nxt] = current_component
                    stack.append(nxt)

        current_component += 1

    return component_id


def adjacency_to_sparse_matrix(adjacency: List[set], num_nodes: int) -> sparse.csr_matrix:
    rows = []
    cols = []
    for u, neigh in enumerate(adjacency):
        if not neigh:
            continue
        rows.extend([u] * len(neigh))
        cols.extend(neigh)

    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def fit_node_representations(
    node_features: np.ndarray,
    adjacency_matrix: sparse.csr_matrix,
    text_dim: int,
    graph_dim: int,
    seed: int,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    node_features_sparse = sparse.csr_matrix(node_features)
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    node_tfidf = tfidf.fit_transform(node_features_sparse)

    text_components = min(text_dim, node_tfidf.shape[0] - 1, node_tfidf.shape[1] - 1)
    text_components = max(text_components, 2)
    text_svd = TruncatedSVD(n_components=text_components, random_state=seed)
    text_embedding = text_svd.fit_transform(node_tfidf).astype(np.float32)

    graph_components = min(graph_dim, adjacency_matrix.shape[0] - 1, adjacency_matrix.shape[1] - 1)
    graph_components = max(graph_components, 2)
    graph_svd = TruncatedSVD(n_components=graph_components, random_state=seed)
    graph_embedding = graph_svd.fit_transform(adjacency_matrix).astype(np.float32)

    return node_tfidf, text_embedding, graph_embedding


def build_feature_context(
    adjacency: List[set],
    degree: np.ndarray,
    component_id: np.ndarray,
    node_features: np.ndarray,
    node_tfidf: sparse.csr_matrix,
    text_embedding: np.ndarray,
    graph_embedding: np.ndarray,
) -> FeatureContext:
    node_nonzero = (node_features > 0).sum(axis=1).astype(np.float32)
    node_norm = np.linalg.norm(node_features, axis=1).astype(np.float32)
    text_embedding_norm = np.linalg.norm(text_embedding, axis=1).astype(np.float32)
    graph_embedding_norm = np.linalg.norm(graph_embedding, axis=1).astype(np.float32)

    return FeatureContext(
        adjacency=adjacency,
        degree=degree,
        component_id=component_id,
        node_features=node_features,
        node_nonzero=node_nonzero,
        node_norm=node_norm,
        node_tfidf=node_tfidf,
        text_embedding=text_embedding,
        text_embedding_norm=text_embedding_norm,
        graph_embedding=graph_embedding,
        graph_embedding_norm=graph_embedding_norm,
    )


def build_pair_features(
    pairs: np.ndarray,
    context: FeatureContext,
    y: Optional[np.ndarray] = None,
    remove_observed_edge_for_positive_pairs: bool = False,
) -> np.ndarray:
    u = pairs[:, 0]
    v = pairs[:, 1]
    num_pairs = pairs.shape[0]

    deg_u = context.degree[u].astype(np.float32)
    deg_v = context.degree[v].astype(np.float32)

    if remove_observed_edge_for_positive_pairs and y is not None:
        positive_mask = (y == 1).astype(np.float32)
        deg_u_eff = np.maximum(deg_u - positive_mask, 0.0)
        deg_v_eff = np.maximum(deg_v - positive_mask, 0.0)
    else:
        deg_u_eff = deg_u
        deg_v_eff = deg_v

    common_neighbors = np.zeros(num_pairs, dtype=np.float32)
    adamic_adar = np.zeros(num_pairs, dtype=np.float32)
    resource_allocation = np.zeros(num_pairs, dtype=np.float32)

    for i in range(num_pairs):
        node_u = int(u[i])
        node_v = int(v[i])
        neigh_u = context.adjacency[node_u]
        neigh_v = context.adjacency[node_v]

        if len(neigh_u) <= len(neigh_v):
            smaller, larger = neigh_u, neigh_v
        else:
            smaller, larger = neigh_v, neigh_u

        cn_count = 0.0
        aa_score = 0.0
        ra_score = 0.0

        for w in smaller:
            if w not in larger:
                continue
            cn_count += 1.0
            dw = context.degree[w]
            if dw > 1:
                aa_score += 1.0 / math.log(dw)
            if dw > 0:
                ra_score += 1.0 / dw

        common_neighbors[i] = cn_count
        adamic_adar[i] = aa_score
        resource_allocation[i] = ra_score

    graph_union = deg_u_eff + deg_v_eff - common_neighbors
    jaccard_graph = common_neighbors / np.maximum(graph_union, 1.0)
    sorensen = (2.0 * common_neighbors) / np.maximum(deg_u_eff + deg_v_eff, 1.0)
    hub_promoted = common_neighbors / np.maximum(np.minimum(deg_u_eff, deg_v_eff), 1.0)
    hub_depressed = common_neighbors / np.maximum(np.maximum(deg_u_eff, deg_v_eff), 1.0)
    preferential_attachment = deg_u_eff * deg_v_eff

    same_component = (context.component_id[u] == context.component_id[v]).astype(np.float32)
    both_isolated = ((deg_u == 0) & (deg_v == 0)).astype(np.float32)

    node_u_raw = context.node_features[u]
    node_v_raw = context.node_features[v]
    raw_dot = np.einsum("ij,ij->i", node_u_raw, node_v_raw).astype(np.float32)
    raw_cosine = raw_dot / (context.node_norm[u] * context.node_norm[v] + EPS)
    raw_l1 = np.abs(node_u_raw - node_v_raw).sum(axis=1).astype(np.float32)
    raw_l2 = np.sqrt(np.square(node_u_raw - node_v_raw).sum(axis=1)).astype(np.float32)

    keyword_union = context.node_nonzero[u] + context.node_nonzero[v] - raw_dot
    jaccard_keywords = raw_dot / np.maximum(keyword_union, 1.0)
    shared_keyword_ratio = raw_dot / np.maximum(
        np.minimum(context.node_nonzero[u], context.node_nonzero[v]), 1.0
    )

    tfidf_cosine = (
        np.asarray(context.node_tfidf[u].multiply(context.node_tfidf[v]).sum(axis=1))
        .ravel()
        .astype(np.float32)
    )

    node_u_text = context.text_embedding[u]
    node_v_text = context.text_embedding[v]
    text_dot = np.einsum("ij,ij->i", node_u_text, node_v_text).astype(np.float32)
    text_cosine = text_dot / (context.text_embedding_norm[u] * context.text_embedding_norm[v] + EPS)
    text_l2 = np.sqrt(np.square(node_u_text - node_v_text).sum(axis=1)).astype(np.float32)
    text_hadamard = (node_u_text * node_v_text).astype(np.float32)
    text_absdiff = np.abs(node_u_text - node_v_text).astype(np.float32)

    node_u_graph = context.graph_embedding[u]
    node_v_graph = context.graph_embedding[v]
    graph_dot = np.einsum("ij,ij->i", node_u_graph, node_v_graph).astype(np.float32)
    graph_cosine = graph_dot / (
        context.graph_embedding_norm[u] * context.graph_embedding_norm[v] + EPS
    )
    graph_l2 = np.sqrt(np.square(node_u_graph - node_v_graph).sum(axis=1)).astype(np.float32)
    graph_hadamard = (node_u_graph * node_v_graph).astype(np.float32)
    graph_absdiff = np.abs(node_u_graph - node_v_graph).astype(np.float32)

    scalar_features = np.column_stack(
        [
            deg_u_eff,
            deg_v_eff,
            np.log1p(deg_u_eff),
            np.log1p(deg_v_eff),
            np.abs(deg_u_eff - deg_v_eff),
            deg_u_eff + deg_v_eff,
            common_neighbors,
            jaccard_graph,
            adamic_adar,
            resource_allocation,
            sorensen,
            hub_promoted,
            hub_depressed,
            preferential_attachment,
            same_component,
            both_isolated,
            raw_dot,
            raw_cosine,
            raw_l1,
            raw_l2,
            jaccard_keywords,
            shared_keyword_ratio,
            tfidf_cosine,
            text_dot,
            text_cosine,
            text_l2,
            graph_dot,
            graph_cosine,
            graph_l2,
        ]
    ).astype(np.float32)

    full_features = np.hstack([scalar_features, text_hadamard, text_absdiff, graph_hadamard, graph_absdiff]).astype(
        np.float32
    )

    return full_features


def build_model_factories(seed: int, use_xgb: bool) -> Dict[str, Callable[[], object]]:
    factories: Dict[str, Callable[[], object]] = {}

    if use_xgb and HAS_XGBOOST:
        factories["xgb"] = lambda: XGBClassifier(
            n_estimators=600,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )

    factories["hgb"] = lambda: HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=0.01,
        random_state=seed,
    )

    factories["lr"] = lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=2.0,
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
        ),
    )

    return factories


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    factories: Dict[str, Callable[[], object]],
    cv_folds: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    if cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    summary: Dict[str, Dict[str, float]] = {}
    oof_predictions: Dict[str, np.ndarray] = {}

    for name, factory in factories.items():
        model_oof = np.zeros(X.shape[0], dtype=np.float32)
        fold_aucs = []
        fold_aps = []

        for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            model = factory()
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[valid_idx])[:, 1]
            model_oof[valid_idx] = proba

            fold_auc = roc_auc_score(y[valid_idx], proba)
            fold_ap = average_precision_score(y[valid_idx], proba)
            fold_aucs.append(fold_auc)
            fold_aps.append(fold_ap)

            print(
                f"[cv] model={name} fold={fold_id}/{cv_folds} "
                f"auc={fold_auc:.5f} ap={fold_ap:.5f}"
            )

        auc_oof = roc_auc_score(y, model_oof)
        ap_oof = average_precision_score(y, model_oof)
        summary[name] = {
            "auc_oof": float(auc_oof),
            "ap_oof": float(ap_oof),
            "auc_mean": float(np.mean(fold_aucs)),
            "auc_std": float(np.std(fold_aucs)),
            "ap_mean": float(np.mean(fold_aps)),
            "ap_std": float(np.std(fold_aps)),
        }
        oof_predictions[name] = model_oof

        print(
            f"[cv] model={name} auc_oof={auc_oof:.5f} ap_oof={ap_oof:.5f} "
            f"auc_mean={np.mean(fold_aucs):.5f} +- {np.std(fold_aucs):.5f}"
        )

    return summary, oof_predictions


def optimize_blend_weights(
    oof_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    seed: int,
    num_trials: int = 5000,
) -> Tuple[Dict[str, float], float, float]:
    model_names = list(oof_predictions.keys())
    stacked_oof = np.column_stack([oof_predictions[name] for name in model_names]).astype(np.float64)
    num_models = stacked_oof.shape[1]

    candidates: List[np.ndarray] = []

    for i in range(num_models):
        one_hot = np.zeros(num_models, dtype=np.float64)
        one_hot[i] = 1.0
        candidates.append(one_hot)

    candidates.append(np.full(num_models, 1.0 / num_models, dtype=np.float64))

    rng = np.random.default_rng(seed)
    for _ in range(num_trials):
        candidates.append(rng.dirichlet(np.ones(num_models)))

    best_auc = -1.0
    best_ap = -1.0
    best_weights = candidates[0]

    for weights in candidates:
        blend = stacked_oof @ weights
        auc = roc_auc_score(y_true, blend)
        if auc > best_auc:
            best_auc = auc
            best_ap = average_precision_score(y_true, blend)
            best_weights = weights

    return (
        {name: float(weight) for name, weight in zip(model_names, best_weights)},
        float(best_auc),
        float(best_ap),
    )


def train_full_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    factories: Dict[str, Callable[[], object]],
    weights: Dict[str, float],
) -> np.ndarray:
    test_pred = np.zeros(X_test.shape[0], dtype=np.float64)

    for name, factory in factories.items():
        model = factory()
        model.fit(X_train, y_train)
        model_pred = model.predict_proba(X_test)[:, 1]
        test_pred += weights[name] * model_pred
        print(f"[fit] model={name} done, weight={weights[name]:.4f}")

    test_pred = np.clip(test_pred, 0.0, 1.0)
    return test_pred.astype(np.float32)


def save_submission(path: Path, predictions: np.ndarray) -> None:
    submission = pd.DataFrame({"ID": np.arange(predictions.shape[0]), "Predicted": predictions})
    submission.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    start_time = time.time()

    train_path = Path(args.train)
    test_path = Path(args.test)
    node_path = Path(args.node_info)
    output_path = Path(args.output)

    train_pairs, y_train, test_pairs, node_features = load_data(train_path, test_path, node_path)
    print(
        f"[data] train_pairs={train_pairs.shape[0]} test_pairs={test_pairs.shape[0]} "
        f"nodes={node_features.shape[0]} node_features={node_features.shape[1]} pos_rate={y_train.mean():.4f}"
    )

    adjacency, degree, component_id, adjacency_matrix = build_graph(
        train_pairs, y_train, node_features.shape[0]
    )
    print(
        f"[graph] nodes={adjacency_matrix.shape[0]} edges={adjacency_matrix.nnz // 2} "
        f"mean_degree={degree.mean():.2f}"
    )

    node_tfidf, text_embedding, graph_embedding = fit_node_representations(
        node_features=node_features,
        adjacency_matrix=adjacency_matrix,
        text_dim=args.text_dim,
        graph_dim=args.graph_dim,
        seed=args.seed,
    )
    print(
        f"[repr] text_dim={text_embedding.shape[1]} graph_dim={graph_embedding.shape[1]}"
    )

    context = build_feature_context(
        adjacency=adjacency,
        degree=degree,
        component_id=component_id,
        node_features=node_features,
        node_tfidf=node_tfidf,
        text_embedding=text_embedding,
        graph_embedding=graph_embedding,
    )

    X_train = build_pair_features(
        train_pairs,
        context,
        y=y_train,
        remove_observed_edge_for_positive_pairs=True,
    )
    X_test = build_pair_features(test_pairs, context)
    print(f"[features] train_shape={X_train.shape} test_shape={X_test.shape}")

    use_xgb = (not args.skip_xgb) and HAS_XGBOOST
    if not use_xgb:
        print("[models] XGBoost not used.")
    factories = build_model_factories(seed=args.seed, use_xgb=use_xgb)

    if args.cv_folds >= 2:
        summary, oof_predictions = cross_validate_models(
            X=X_train,
            y=y_train,
            factories=factories,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )
        weights, ensemble_auc, ensemble_ap = optimize_blend_weights(
            oof_predictions=oof_predictions,
            y_true=y_train,
            seed=args.seed,
        )

        print("[blend] model weights:")
        for name, weight in weights.items():
            print(f"  - {name}: {weight:.4f}")
        print(f"[blend] ensemble_oof_auc={ensemble_auc:.5f} ensemble_oof_ap={ensemble_ap:.5f}")
    else:
        equal_weight = 1.0 / len(factories)
        weights = {name: equal_weight for name in factories.keys()}
        print("[blend] CV disabled, using equal model weights.")

    test_pred = train_full_and_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        factories=factories,
        weights=weights,
    )

    save_submission(output_path, test_pred)
    print(
        f"[output] saved={output_path} rows={test_pred.shape[0]} "
        f"min={test_pred.min():.5f} max={test_pred.max():.5f}"
    )

    if args.binary_output:
        binary_path = output_path.with_name(output_path.stem + "_binary.csv")
        binary_pred = (test_pred >= args.binary_threshold).astype(np.int32)
        save_submission(binary_path, binary_pred)
        print(
            f"[output] binary_saved={binary_path} rows={binary_pred.shape[0]} "
            f"threshold={args.binary_threshold:.2f}"
        )

    elapsed = time.time() - start_time
    print(f"[done] total_time_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
