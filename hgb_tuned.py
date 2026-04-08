"""
HGB with keyword hadamard features.
Instead of summarizing 932 keywords into 3 features (dot, cosine, jaccard),
give HGB the full hadamard product (shared keyword indicators).
HGB selects the most predictive keywords automatically.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from logistic_regression import (
    build_features,
    build_graph,
    load_data,
    save_submission,
)

EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.txt")
    p.add_argument("--test", default="test.txt")
    p.add_argument("--node-info", default="node_information.csv")
    p.add_argument("--output", default="submission_hgb_tuned.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================
# Graph utils
# ============================================================

def _build_graph(train_pairs, y_train, num_nodes):
    adjacency = [set() for _ in range(num_nodes)]
    for u, v in train_pairs[y_train == 1]:
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
    degree = np.array([len(n) for n in adjacency], dtype=np.int32)
    comp = np.full(num_nodes, -1, dtype=np.int32)
    cid = 0
    for start in range(num_nodes):
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
    return adjacency, degree, comp


def augment_graph(adjacency, pairs, probs, threshold, num_nodes):
    new_adj = [s.copy() for s in adjacency]
    added = 0
    for i in range(len(probs)):
        if probs[i] >= threshold:
            u, v = int(pairs[i, 0]), int(pairs[i, 1])
            if v not in new_adj[u]:
                new_adj[u].add(v)
                new_adj[v].add(u)
                added += 1
    new_deg = np.array([len(n) for n in new_adj], dtype=np.int32)
    comp = np.full(num_nodes, -1, dtype=np.int32)
    cid = 0
    for start in range(num_nodes):
        if comp[start] != -1:
            continue
        stack = [start]
        comp[start] = cid
        while stack:
            node = stack.pop()
            for nxt in new_adj[node]:
                if comp[nxt] == -1:
                    comp[nxt] = cid
                    stack.append(nxt)
        cid += 1
    return new_adj, new_deg, comp, added


# ============================================================
# Feature building
# ============================================================

def select_useful_keywords(node_features, min_df=5, max_df_ratio=0.5):
    """Select keywords that appear in at least min_df nodes and at most max_df_ratio of nodes."""
    n_nodes = node_features.shape[0]
    df = (node_features > 0).sum(axis=0)
    mask = (df >= min_df) & (df <= max_df_ratio * n_nodes)
    return np.where(mask)[0]


def build_features_with_hadamard(pairs, adjacency, degree, components, node_features,
                                  kw_indices, y=None, remove_positive_edge=False):
    """16 base features + selected keyword hadamard product."""
    X_base = build_features(pairs, adjacency, degree, components, node_features,
                            y=y, remove_positive_edge=remove_positive_edge)

    u = pairs[:, 0]
    v = pairs[:, 1]

    # Hadamard product of selected keywords
    fu = node_features[u][:, kw_indices]
    fv = node_features[v][:, kw_indices]
    hadamard = (fu * fv).astype(np.float32)

    return np.hstack([X_base, hadamard])


# ============================================================
# Model + pipeline
# ============================================================

def hgb_predict(X_tr, y_tr, X_te, seed, n_seeds, params):
    pred = np.zeros(len(X_te), dtype=np.float64)
    for s in range(n_seeds):
        m = HistGradientBoostingClassifier(**params, random_state=seed + s * 111)
        m.fit(X_tr, y_tr)
        pred += m.predict_proba(X_te)[:, 1]
    return pred / n_seeds


def run_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                 kw_indices, seed, n_seeds, params, thresholds, label=""):
    adj, deg, comp = _build_graph(train_pairs, y_train, num_nodes)

    X_tr = build_features_with_hadamard(
        train_pairs, adj, deg, comp, node_features, kw_indices,
        y=y_train, remove_positive_edge=True)
    X_te = build_features_with_hadamard(
        test_pairs, adj, deg, comp, node_features, kw_indices)
    print(f"  features: {X_tr.shape[1]} (16 base + {len(kw_indices)} keywords)")

    pred = hgb_predict(X_tr, y_train, X_te, seed, n_seeds, params)

    for thresh in thresholds:
        adj, deg, comp, added = augment_graph(adj, test_pairs, pred, thresh, num_nodes)
        if added == 0:
            break
        print(f"  +{added} edges (t={thresh}), mean_deg={deg.mean():.2f}")
        X_tr = build_features_with_hadamard(
            train_pairs, adj, deg, comp, node_features, kw_indices,
            y=y_train, remove_positive_edge=True)
        X_te = build_features_with_hadamard(
            test_pairs, adj, deg, comp, node_features, kw_indices)
        pred = hgb_predict(X_tr, y_train, X_te, seed, n_seeds, params)

    print(f"  -> [{pred.min():.4f}, {pred.max():.4f}]")
    return pred


# Base only pipeline (no hadamard)
def run_base_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                      seed, n_seeds, params, thresholds):
    adj, deg, comp = _build_graph(train_pairs, y_train, num_nodes)
    X_tr = build_features(train_pairs, adj, deg, comp, node_features,
                          y=y_train, remove_positive_edge=True)
    X_te = build_features(test_pairs, adj, deg, comp, node_features)

    pred = hgb_predict(X_tr, y_train, X_te, seed, n_seeds, params)

    for thresh in thresholds:
        adj, deg, comp, added = augment_graph(adj, test_pairs, pred, thresh, num_nodes)
        if added == 0:
            break
        X_tr = build_features(train_pairs, adj, deg, comp, node_features,
                              y=y_train, remove_positive_edge=True)
        X_te = build_features(test_pairs, adj, deg, comp, node_features)
        pred = hgb_predict(X_tr, y_train, X_te, seed, n_seeds, params)

    print(f"  -> [{pred.min():.4f}, {pred.max():.4f}]")
    return pred


HGB_DEFAULT = dict(
    learning_rate=0.07, max_depth=4, max_iter=250,
    min_samples_leaf=30, l2_regularization=0.02,
)
HGB_SHALLOW = dict(
    learning_rate=0.07, max_depth=3, max_iter=300,
    min_samples_leaf=30, l2_regularization=0.02,
)
# More regularized for high-dim hadamard features
HGB_REGULARIZED = dict(
    learning_rate=0.05, max_depth=3, max_iter=350,
    min_samples_leaf=50, l2_regularization=0.10,
)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    train_pairs, y_train, test_pairs, node_features = load_data(
        Path(args.train), Path(args.test), Path(args.node_info))
    num_nodes = node_features.shape[0]
    print(f"[data] train={len(y_train)} test={len(test_pairs)} nodes={num_nodes}")

    # Select useful keywords
    kw_all = select_useful_keywords(node_features, min_df=5, max_df_ratio=0.5)
    kw_strict = select_useful_keywords(node_features, min_df=20, max_df_ratio=0.3)
    print(f"[keywords] all={len(kw_all)}, strict={len(kw_strict)} (from 932)")

    strategies = []

    # A: Base 16 features + self-training (proven)
    print("\n[A] Base 16 features")
    p = run_base_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                          seed=args.seed, n_seeds=8, params=HGB_DEFAULT,
                          thresholds=[0.9, 0.85, 0.8])
    strategies.append(p)

    # B: Base 16 + aggressive self-training
    print("\n[B] Base aggressive")
    p = run_base_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                          seed=args.seed, n_seeds=8, params=HGB_DEFAULT,
                          thresholds=[0.85, 0.75, 0.65])
    strategies.append(p)

    # C: Hadamard (strict keywords) + regularized HGB
    print("\n[C] Hadamard strict + regularized")
    p = run_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                     kw_strict, seed=args.seed, n_seeds=8, params=HGB_REGULARIZED,
                     thresholds=[0.9, 0.85, 0.8])
    strategies.append(p)

    # D: Hadamard (all keywords) + regularized HGB
    print("\n[D] Hadamard all + regularized")
    p = run_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                     kw_all, seed=args.seed, n_seeds=8, params=HGB_REGULARIZED,
                     thresholds=[0.9, 0.85, 0.8])
    strategies.append(p)

    # E: Hadamard strict + default HGB
    print("\n[E] Hadamard strict + default HGB")
    p = run_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                     kw_strict, seed=args.seed, n_seeds=8, params=HGB_DEFAULT,
                     thresholds=[0.9, 0.85, 0.8])
    strategies.append(p)

    # F: Base + different seed
    print("\n[F] Base seed 2")
    p = run_base_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                          seed=args.seed + 7, n_seeds=8, params=HGB_DEFAULT,
                          thresholds=[0.9, 0.85, 0.8])
    strategies.append(p)

    # G: Hadamard strict + different seed
    print("\n[G] Hadamard strict seed 2")
    p = run_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                     kw_strict, seed=args.seed + 7, n_seeds=8, params=HGB_REGULARIZED,
                     thresholds=[0.9, 0.85, 0.8])
    strategies.append(p)

    # H: Base shallow
    print("\n[H] Base shallow")
    p = run_base_pipeline(train_pairs, y_train, test_pairs, node_features, num_nodes,
                          seed=args.seed, n_seeds=8, params=HGB_SHALLOW,
                          thresholds=[0.9, 0.85, 0.8])
    strategies.append(p)

    # --- Blends ---
    mega = np.mean(strategies, axis=0).astype(np.float32)
    save_submission(Path("submission_hadamard_mega.csv"), mega)
    print(f"\n[mega] 8 strategies -> [{mega.min():.5f}, {mega.max():.5f}]")

    # Hadamard-only blend (C + D + E + G)
    had_blend = np.mean([strategies[2], strategies[3], strategies[4], strategies[6]],
                        axis=0).astype(np.float32)
    save_submission(Path("submission_hadamard_only.csv"), had_blend)
    print(f"[hadamard only] -> [{had_blend.min():.5f}, {had_blend.max():.5f}]")

    # Base-only blend (A + B + F + H)
    base_blend = np.mean([strategies[0], strategies[1], strategies[5], strategies[7]],
                         axis=0).astype(np.float32)
    save_submission(Path("submission_base_only.csv"), base_blend)
    print(f"[base only] -> [{base_blend.min():.5f}, {base_blend.max():.5f}]")

    # Save default output as mega blend
    save_submission(Path(args.output), mega)


if __name__ == "__main__":
    main()
