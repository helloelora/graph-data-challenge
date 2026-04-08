# Link Prediction — Results Summary

🥇 **Final Kaggle public AUC: 0.87091 (1st place)** with `submission_v24.csv`

## Final leaderboard (top 5)

| # | Team | Score |
|---|---|---|
| **1** | **GraphTheWorld (us)** | **0.87091** |
| 2 | SOTAlmost | 0.86925 |
| 3 | LavaXMaster | 0.86820 |
| 4 | ninjjacat | 0.86769 |
| 5 | Adam Y | 0.85590 |

## Full progression of submissions

| Strategy | Code | Submission CSV | OOF AUC | Kaggle Score |
|----------|------|----------------|---------|--------------|
| Logistic Regression | `logistic_regression.py` | `submission_lr_baseline.csv` | 0.5662 | — |
| HistGradientBoosting (16 features) | `hist_gradient_boosting.py` | `submission_hgb_intermediate.csv` | 0.5648 | 0.84927 |
| HGB 5-seed ensemble (16 features) | `hgb_tuned.py` | `submission_16feat_depth4_ensemble.csv` | — | 0.84936 |
| HGB self-training t=0.9 (16 features) | `hgb_tuned.py` | `submission_t90.csv` | — | 0.85094 |
| HGB 4-way blend (16 features) | `hgb_tuned.py` | `submission_blend_noadvsvd.csv` | — | 0.85358 |
| HGB + N2V mega blend (16 features) | `hgb_tuned.py` | `submission_n2v_mega_blend.csv` | — | 0.85373 |
| HGB + 2-hop completion blend | `hgb_tuned.py` | `submission_2hop_mega_blend.csv` | — | 0.85298 |
| XGB+HGB+LR Ensemble (256d embeddings) | `xgb_hgb_lr_ensemble.py` | `submission_best.csv` | 0.5371 | 0.76833 |
| **HGB v19 (41 features, HGB+CatBoost rank blend)** | `best_solution.py` | `submission_v19_hgb_cat_rank.csv` | 0.90245 | **0.86766** |
| HGB v21 (v19 + dual self-training) | `experiments_v21.py` | `submission_v21_dual_st.csv` | — | 0.86660 |
| HGB v21 (v19 + 50 seeds) | `experiments_v21.py` | `submission_v21_50seeds.csv` | — | 0.86655 |
| HGB v21 (v19 + LightGBM) | `experiments_v21.py` | `submission_v21_3way_2hgbcat.csv` | — | 0.86629 |
| HGB v22 (v19 + hyperparam variants avg) | `experiments_v22.py` | `submission_v22_hp_avg.csv` | — | 0.86672 |
| HGB v22 (v19 + pseudo-labeling) | `experiments_v22.py` | `submission_v22_safe.csv` | — | 0.86674 |
| HGB v23 (v19 + symmetric TTA blend) | `experiments_v23.py` | `submission_v23_tta_v19_blend.csv` | — | 0.86741 |
| HGB v24 (v19 + 7 pair-transductive) blend 70/30 | `experiments_v24.py` | `submission_v24_blend_70v19.csv` | 0.90319 | 0.86948 |
| **HGB v24 pure (48 features, HGB+CatBoost rank blend)** 🥇 | `experiments_v24.py` | `submission_v24.csv` | 0.90319 | **0.87091** |

> OOF AUCs are inflated by the leaky transductive features (computed once on train+test). Kaggle score is the public leaderboard AUC.

---

## The winning idea: pair-level transductive features (v24)

After v19 plateaued at 0.86766, six controlled experiments (more seeds, +LightGBM, dual self-training, pseudo-labeling, feature bagging, hyperparam variants, train/test symmetric augmentation) all scored 0.001-0.0014 *below* v19. v19 was at a tight local optimum.

**The breakthrough** came from adding a fundamentally new family of features rather than perturbing the existing pipeline. v19's biggest single gain (+0.011 AUC) had come from *node-level* transductive counts — how many times each node appears across train and test pairs. v24 pushes this idea one order higher: instead of looking at how often each node appears, we look at which **other nodes are shared partners across the test set**.

For each pair $(u, v)$:
- $\text{test\_partners}(u)$ = set of nodes that appear in some test pair with $u$
- `shared_test_partners` = $|\text{test\_partners}(u) \cap \text{test\_partners}(v)|$
- Same for train partners and combined train+test partners
- Jaccard variants of these intersections
- min/max of test-partner-set sizes for each side

**Why it works:** if two actors share many test partners, they likely belong to the same cluster in the original graph and the edge between them was probably one of those that got randomly deleted. The signal is leakage-free (no labels used), sparse but strongly directional: positive training pairs have on average **12× more shared test partners** than negative pairs.

**Results:**
- v19 (41 features): 0.86766
- v24 (48 features) blended 70/30 with v19: 0.86948 (+0.00182, took us to #1)
- **v24 (48 features) pure: 0.87091 (+0.00325 over v19)** 🥇

---

## Strategy details

### Final solution — `experiments_v24.py`

**Model:** HistGradientBoosting + CatBoost (30 random seeds each), rank-averaged 50/50.

**Features (48 total):**
1. **Graph topology (14):** degree (u, v, |Δ|, sum, log_u, log_v, min, max), common neighbors, Jaccard, Adamic-Adar, resource allocation, Sorensen, preferential attachment, same component, both isolated, hub promoted, hub depressed, paths of length 3 (with leakage correction), Katz approximation
2. **Text similarity (9):** raw dot/cosine, keyword Jaccard, TF-IDF cosine, TF-IDF L2, asymmetric overlap (u and v), neighborhood TF-IDF (uv, vu, nn — 1-hop GCN-style aggregation with leakage correction)
3. **Node-level transductive (6):** total/test counts for u and v + product and absolute diff of total counts
4. **Pair-level transductive (7) — v24's contribution:** shared test partners, shared train partners, shared total partners, Jaccard of test partners, Jaccard of all partners, min/max of test partner set size
5. **Feature interactions (5):** common neighbors × TF-IDF cosine, CN × cosine, PA × cosine, same_comp × TF-IDF, paths3 × TF-IDF
6. **Self-training:** one conservative round at threshold 0.95 adds ~305 high-confidence test pairs as pseudo-edges, then graph-dependent features are recomputed

### v19 baseline — `best_solution.py`

Same as the final solution minus the 7 pair-level transductive features. Reaches 0.86766 on Kaggle. Built up over many iterations (v4 → v12 → v16 → v19), each adding a small set of carefully designed features and verifying the improvement transfers to Kaggle (not just OOF, since OOF is leaky from transductive counts).

### Earlier experiments — `hgb_tuned.py`

Iterative exploration of 16-feature HGB with self-training and blending strategies. Best score on this branch: 0.85373 (N2V mega blend). Plateaued well below v19 because it never adds the transductive node counts.

### Failed strategies (`xgb_hgb_lr_ensemble.py`, embeddings)

256-dimensional SVD adjacency + TF-IDF SVD embeddings combined with hand-crafted features in an XGB+HGB+LR ensemble. Scored 0.768 — embeddings overfit on the small dataset and dragged down the strong scalar features.

---

## Key lessons

1. **Feature discipline beats model complexity** on small datasets. Every embedding-based approach (GCN, Node2Vec, SVD) underperformed simple scalar features.
2. **Local search has hard limits.** Once v19 plateaued, six rounds of perturbations all hurt by the same 0.001 — we needed orthogonal information, not better tuning.
3. **Transductive signal compounds.** Node-level test counts gave +0.011. Pair-level test partner intersections gave +0.003 more on top.
4. **Self-training works in moderation.** One round at threshold 0.95 helps. Multiple rounds or lower thresholds added noise and hurt every time.
5. **Symmetric features matter — but the model already learns them.** Augmenting (u, v) ↔ (v, u) at training and test time hurt slightly, suggesting the order signal in v19 features is not spurious.
