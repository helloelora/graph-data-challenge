import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_link_predictor_lr_baseline import (
    build_features,
    build_graph,
    load_data,
    save_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intermediate trained baseline (HistGradientBoosting + compact handcrafted features)."
    )
    parser.add_argument("--train", default="train.txt", help="Path to train file.")
    parser.add_argument("--test", default="test.txt", help="Path to test file.")
    parser.add_argument(
        "--node-info", default="node_information.csv", help="Path to node feature file."
    )
    parser.add_argument(
        "--output",
        default="submission_hgb_intermediate.csv",
        help="Output submission CSV path.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def get_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_depth=4,
        max_iter=250,
        min_samples_leaf=30,
        l2_regularization=0.02,
        random_state=seed,
    )


def cross_validate(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> None:
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
    pred_test = model.predict_proba(X_test)[:, 1].astype(np.float32)

    save_submission(Path(args.output), pred_test)
    print(
        f"[output] saved={args.output} rows={pred_test.shape[0]} "
        f"min={pred_test.min():.5f} max={pred_test.max():.5f}"
    )


if __name__ == "__main__":
    main()
