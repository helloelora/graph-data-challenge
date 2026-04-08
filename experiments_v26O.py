"""
v26O — adversarial validation feature pruning on v26L.

Completely different optimization axis from v26L -> v26M -> v26N:
rather than *adding* features, *remove* features that are distribution-
shifted between the training pairs and the test pairs and don't
generalize. A noisy feature that trees use for training-set splits
but which has different behavior at Kaggle test time can actively
hurt generalization.

Method:
  1. Build the v26L feature matrix for BOTH train and test pairs.
  2. Stack them and label train rows 0, test rows 1.
  3. Train a gradient booster to distinguish train vs test rows on
     the feature matrix. The resulting classifier's per-feature
     importance is a measure of train/test distribution shift.
  4. For each feature with importance > some threshold, drop it and
     re-evaluate the HGB+CatBoost blend OOF on v26L's actual label
     task.
  5. Keep only drops that strictly improve the blend (same discipline).
  6. Final model uses the pruned feature set.

Expected effect: small but real. A typical gain from this technique
on a well-tuned pipeline is +0.0005 to +0.002 Kaggle. Not a big jump
but cleanly orthogonal to everything else we've tried (this doesn't
add features, it subtracts them).

Why this might help on v26L specifically: several v26 features use
transductive counts that are computed jointly across train+test, so
in principle they should NOT be distribution-shifted (their inputs
are symmetric). But some v19 features (degrees, paths3, neighbor TF-
IDF) are computed on the real adjacency graph which is derived from
train labels only — those features CAN have train/test distribution
shift if the training pairs are drawn from a different structural
regime than the test pairs.

Risk: adversarial-validation pruning is a variance trap — some
features with high importance for train-vs-test classification are
also important for the real label task and dropping them is
catastrophic. We mitigate by dropping one at a time and keeping
only strict positive blend deltas.
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
from experiments_v26g import build_text_weighted_candidate_graph
from experiments_v26h import compute_consensus_same_community
from experiments_v26L import compute_consensus_at_resolution


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


def build_v26L_feature_matrix(
    pairs_train, y_train, pairs_test, node_features, node_tfidf,
):
    """Recompute the v26L 70-feature matrix for train and test pairs."""
    n_nodes = node_features.shape[0]

    total_count = np.zeros(n_nodes, dtype=np.float32)
    train_count = np.zeros(n_nodes, dtype=np.float32)
    test_count = np.zeros(n_nodes, dtype=np.float32)
    for u, v in pairs_train:
        total_count[u] += 1; total_count[v] += 1
        train_count[u] += 1; train_count[v] += 1
    for u, v in pairs_test:
        total_count[u] += 1; total_count[v] += 1
        test_count[u] += 1; test_count[v] += 1

    train_partners, test_partners = build_partner_sets(pairs_train, pairs_test, n_nodes)
    pair_v24_tr = compute_pair_transductive_v24(pairs_train, train_partners, test_partners)
    pair_v24_te = compute_pair_transductive_v24(pairs_test, train_partners, test_partners)
    pair_v25_tr = compute_pair_transductive_v25(
        pairs_train, train_partners, test_partners, test_count, train_count, total_count)
    pair_v25_te = compute_pair_transductive_v25(
        pairs_test, train_partners, test_partners, test_count, train_count, total_count)

    adj0, deg0, comp0 = build_graph(pairs_train, y_train, n_nodes)
    A0 = build_sparse_adj(adj0, n_nodes)
    d_inv0 = np.zeros(n_nodes, dtype=np.float32)
    d_inv0[deg0 > 0] = 1.0 / deg0[deg0 > 0]
    ntfidf0 = sparse.diags(d_inv0) @ A0 @ node_tfidf
    X_tr0 = build_features_v19(
        pairs_train, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count, A0, ntfidf0, y=y_train, remove_pos=True)
    X_te0 = build_features_v19(
        pairs_test, adj0, deg0, comp0, node_features, node_tfidf,
        total_count, train_count, test_count, A0, ntfidf0)
    pred_init = predict_hgb(X_tr0, y_train, X_te0, n_seeds=5)
    extra_edges = pairs_test[pred_init >= 0.95]

    adjacency, degree, comp = build_graph(pairs_train, y_train, n_nodes, extra_edges)
    adj_matrix = build_sparse_adj(adjacency, n_nodes)
    d_inv = np.zeros(n_nodes, dtype=np.float32)
    d_inv[degree > 0] = 1.0 / degree[degree > 0]
    neighbor_tfidf = sparse.diags(d_inv) @ adj_matrix @ node_tfidf

    X_train_v19 = build_features_v19(
        pairs_train, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count, adj_matrix, neighbor_tfidf,
        y=y_train, remove_pos=True)
    X_test_v19 = build_features_v19(
        pairs_test, adjacency, degree, comp, node_features, node_tfidf,
        total_count, train_count, test_count, adj_matrix, neighbor_tfidf)

    X_train_v25 = np.hstack([X_train_v19, pair_v24_tr, pair_v25_tr])
    X_test_v25 = np.hstack([X_test_v19, pair_v24_te, pair_v25_te])

    G_unwt = build_candidate_graph(pairs_train, pairs_test, extra_edges, n_nodes)
    part_unwt = run_louvain(G_unwt, seed=SEED)
    comm_train = compute_community_features(pairs_train, part_unwt, n_nodes)
    comm_test = compute_community_features(pairs_test, part_unwt, n_nodes)
    emb, emb_normed = compute_spectral_embedding(G_unwt, n_nodes, k=16, seed=SEED)
    spec_train = compute_spectral_features(pairs_train, emb, emb_normed)
    spec_test = compute_spectral_features(pairs_test, emb, emb_normed)
    cn_train = compute_comm_cn(pairs_train, G_unwt, part_unwt)
    cn_test = compute_comm_cn(pairs_test, G_unwt, part_unwt)
    X_train_v26d = np.hstack([X_train_v25, comm_train, spec_train, cn_train])
    X_test_v26d = np.hstack([X_test_v25, comm_test, spec_test, cn_test])

    G_text = build_text_weighted_candidate_graph(
        pairs_train, pairs_test, extra_edges, n_nodes, node_tfidf, alpha=1.0, beta=3.0)
    part_text = run_louvain(G_text, seed=SEED)
    comm_text_train = compute_community_features(pairs_train, part_text, n_nodes)
    comm_text_test = compute_community_features(pairs_test, part_text, n_nodes)
    X_train_v26g = np.hstack([X_train_v26d, comm_text_train])
    X_test_v26g = np.hstack([X_test_v26d, comm_text_test])

    cons_unwt_train = compute_consensus_same_community(
        pairs_train, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    cons_unwt_test = compute_consensus_same_community(
        pairs_test, G_unwt, n_seeds=20, base_seed=SEED).reshape(-1, 1)
    X_train_v26hp = np.hstack([X_train_v26g, cons_unwt_train])
    X_test_v26hp = np.hstack([X_test_v26g, cons_unwt_test])

    # v26L multi-res
    for r in [0.7, 1.3, 2.0]:
        tr_arr, _ = compute_consensus_at_resolution(
            pairs_train, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        te_arr, _ = compute_consensus_at_resolution(
            pairs_test, G_unwt, n_seeds=20, base_seed=SEED, resolution=r)
        X_train_v26hp = np.hstack([X_train_v26hp, tr_arr.reshape(-1, 1)])
        X_test_v26hp = np.hstack([X_test_v26hp, te_arr.reshape(-1, 1)])

    return X_train_v26hp, X_test_v26hp


def main():
    t0 = time.time()
    np.random.seed(SEED)

    train_pairs, y_train, test_pairs, node_features = load_data(
        Path("train.txt"), Path("test.txt"), Path("node_information.csv")
    )
    nf_sparse = sparse.csr_matrix(node_features)
    node_tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(nf_sparse)

    print("[v26L] rebuilding 70-feature matrix")
    X_train_v26L, X_test_v26L = build_v26L_feature_matrix(
        train_pairs, y_train, test_pairs, node_features, node_tfidf,
    )
    n_features = X_train_v26L.shape[1]
    print(f"  train: {X_train_v26L.shape}  test: {X_test_v26L.shape}")

    # === Step 1: adversarial validation — predict train vs test ===
    print("\n[adversarial] train-vs-test classifier on v26L features")
    X_stack = np.vstack([X_train_v26L, X_test_v26L])
    y_adv = np.concatenate([
        np.zeros(len(X_train_v26L), dtype=np.int32),
        np.ones(len(X_test_v26L), dtype=np.int32),
    ])
    print(f"  stacked: {X_stack.shape}  class balance: "
          f"train={(y_adv==0).sum()}  test={(y_adv==1).sum()}")

    cv_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_adv = np.zeros(len(y_adv), dtype=np.float64)
    feat_importance = np.zeros(n_features, dtype=np.float64)
    for fold, (tr, va) in enumerate(cv_adv.split(X_stack, y_adv), 1):
        m = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=5, max_iter=200,
            min_samples_leaf=30, l2_regularization=0.1,
            random_state=SEED + fold,
        )
        m.fit(X_stack[tr], y_adv[tr])
        oof_adv[va] = m.predict_proba(X_stack[va])[:, 1]
        # HGB doesn't expose feature_importances_; use permutation importance
        # but that's expensive. Use the raw leaf counts by fitting an XGBoost-
        # style CatBoost instead for importances.
    adv_auc = roc_auc_score(y_adv, oof_adv)
    print(f"  adversarial AUC = {adv_auc:.5f}  "
          f"({'strong shift' if adv_auc > 0.6 else 'weak/no shift' if adv_auc < 0.55 else 'moderate shift'})")

    # Permutation importance via CatBoost for feature ranking
    print("\n[importance] CatBoost feature importance for adversarial task")
    m_adv_cat = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=4,
        l2_leaf_reg=10, verbose=0, random_seed=SEED,
    )
    m_adv_cat.fit(X_stack, y_adv)
    feat_importance = np.asarray(m_adv_cat.feature_importances_, dtype=np.float64)
    ranked = np.argsort(-feat_importance)
    print("  top-10 adversarial importance (most distribution-shifted):")
    for rank_i, j in enumerate(ranked[:10]):
        print(f"    #{rank_i+1:2d}  feat[{j}] importance={feat_importance[j]:.3f}")

    # === Step 2: drop each of the top-shifted features one at a time ===
    print("\n[prune] ablating feature drops on the v26L blend OOF")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_h, base_c, base_b = blend_oof(X_train_v26L, y_train, cv, SEED)
    print(f"  v26L full   HGB={base_h:.5f}  Cat={base_c:.5f}  blend={base_b:.5f}")

    drop_results = {}
    for rank_i, j in enumerate(ranked[:15]):
        X_dropped = np.delete(X_train_v26L, j, axis=1)
        h, c, b = blend_oof(X_dropped, y_train, cv, SEED)
        delta = b - base_b
        drop_results[j] = (h, c, b, delta)
        flag = "  *" if delta > 0 else ""
        print(f"  drop feat[{j:2d}] (adv imp {feat_importance[j]:.2f})  "
              f"HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  ({delta:+.5f}){flag}")

    drops = [j for j, (_, _, _, d) in drop_results.items() if d > 0]
    if not drops:
        print("\n[v26O] no single drop improved the blend. Pipeline is already clean.")
        return

    print(f"\n[v26O] dropping {len(drops)} features: {drops}")
    keep_mask = np.ones(n_features, dtype=bool)
    keep_mask[drops] = False
    X_train_v26O = X_train_v26L[:, keep_mask]
    X_test_v26O = X_test_v26L[:, keep_mask]
    print(f"  v26O: {X_train_v26O.shape[1]} features (dropped {len(drops)})")

    h, c, b = blend_oof(X_train_v26O, y_train, cv, SEED)
    print(f"  v26O HGB={h:.5f}  Cat={c:.5f}  blend={b:.5f}  ({b - base_b:+.5f})")

    # Final 30-seed
    print("\n[final] 30-seed HGB+CatBoost on pruned feature set")
    pred_h = predict_hgb(X_train_v26O, y_train, X_test_v26O, n_seeds=30)
    pred_c = predict_cat(X_train_v26O, y_train, X_test_v26O, n_seeds=30)
    pred_v26O = 0.5 * rnk(pred_h) + 0.5 * rnk(pred_c)
    save_sub("submission_v26O.csv", pred_v26O)

    if Path("submission_v26L.csv").exists():
        v26L_known = pd.read_csv("submission_v26L.csv")["Predicted"].to_numpy()
        rL = rnk(v26L_known)
        corr, _ = spearmanr(rL, pred_v26O)
        print(f"\n  v26O vs submission_v26L.csv spearman = {corr:.5f}")
        save_sub("submission_v26O_blend_50.csv", 0.5 * pred_v26O + 0.5 * rL)

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
