# Link Prediction — Results Summary

🥇 **Best Kaggle public AUC: 0.88080 (1st place)** with `submission_v26g.csv`

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
| HGB v24 pure (48 features, HGB+CatBoost rank blend) | `experiments_v24.py` | `submission_v24.csv` | 0.90319 | 0.87091 |
| HGB v25 pure (56 features, HGB+CatBoost rank blend) | `experiments_v25.py` | `submission_v25.csv` | 0.90357 | 0.87208 |
| HGB v26b (v25 + community + spectral on candidate graph, 62 features) | `experiments_v26b.py` | `submission_v26b.csv` | 0.90686 | 0.88038 |
| HGB v26c (v26b + 10 extra community features) — regressed | `experiments_v26c.py` | `submission_v26c.csv` | 0.90749 | 0.87930 |
| HGB v26d (v26b + comm_cn only, 63 features) | `experiments_v26d.py` | `submission_v26d.csv` | 0.90720 | 0.88050 |
| **HGB v26g (v26d + text-weighted Louvain community, 66 features)** 🥇 | `experiments_v26g.py` | `submission_v26g.csv` | 0.90794 | **0.88080** |

> OOF AUCs are inflated by the transductive features (computed once on train+test). Kaggle score is the public leaderboard AUC.

---

## The winning idea: pair-level transductive features (v24 then v25)

After v19 plateaued at 0.86766, six controlled experiments (more seeds, +LightGBM, dual self-training, pseudo-labeling, feature bagging, hyperparam variants, train/test symmetric augmentation) all scored 0.001-0.0014 *below* v19. v19 was at a tight local optimum.

**The breakthrough** came from adding a fundamentally new family of features rather than perturbing the existing pipeline. v19's biggest single gain (+0.011 AUC) had come from *node-level* transductive counts — how many times each node appears across train and test pairs. v24 pushes this idea one order higher: instead of looking at how often each node appears, we look at which **other nodes are shared partners across the test set**. v25 then pushes the same direction even further with second-order features (transductive triangles, Adamic-Adar in test partner space).

*v24 features (7) — for each pair $(u, v)$:*
- $\text{test\_partners}(u)$ = set of nodes that appear in some test pair with $u$
- `shared_test_partners` = $|\text{test\_partners}(u) \cap \text{test\_partners}(v)|$
- Same for train partners and combined train+test partners
- Jaccard variants of these intersections
- min/max of test-partner-set sizes for each side

*v25 features (8) — same idea, higher order:*
- `test_triangles` = $|\{w : (u,w) \in \text{test} \land (w,v) \in \text{test}\}|$ — paths of length 2 through the test set
- `train_triangles`, `mixed_triangles` = same for train pairs and mixed train/test
- `shared_test_aa` = Adamic-Adar style: $\sum_w 1/\log(\text{test\_count}(w))$ over shared test partners
- `shared_test_ra` = resource-allocation variant
- `shared_total_pa` = $|\text{test\_partners}(u)| \cdot |\text{test\_partners}(v)|$ — preferential attachment in test space
- `exclusive_test_u`, `exclusive_test_v` = asymmetric set-difference sizes

**Why it works:** if two actors share many test partners (or many transductive triangles), they likely belong to the same cluster in the original graph and the edge between them was probably one of those that got randomly deleted. The signal is leakage-free (no labels used), sparse but strongly directional: positive training pairs have on average **12× more shared test partners** (v24) and **23× more test triangles** (v25) than negative pairs.

**Results progression:**
- v19 (41 features): 0.86766
- v24 (48 features) pure: 0.87091 (+0.00325 over v19)
- v25 (56 features) pure: 0.87208 (+0.00117 over v24)

The marginal Kaggle gain (v24→v25) is smaller than the first jump (v19→v24), as expected — v25 mines deeper in the same vein that v24 opened up. Each later submission in the v25 family confirmed that pure v25 is the best, outperforming any blend with earlier versions.

---

## The v26 breakthrough: community structure on a label-free candidate graph

v25 saturated just like v19 had before it. The path forward was the same meta-lesson: stop perturbing the pipeline, find a new information source.

### v26b (+0.00830, 0.87208 → 0.88038)

We build a **candidate graph** that contains every pair in `train.txt` (regardless of label), every pair in `test.txt`, and the v25 self-training pseudo-edges — all with weight 1.0. Running Louvain on this graph gives a partition of the 3,597 actors into ≈16 communities.

From that partition we add **three community features** (`same_community`, `community_size_min`, `community_size_max`) and a **Laplacian eigenmap** (k=16) of the same graph turned into three pair-level features (`spectral_dist_l2`, `spectral_dot`, `spectral_cos`). Six new scalar columns on top of v25's 56, for a total of 62.

**Why it is leakage-free for training pairs:** the graph is built without using training labels. Both positive and negative train pairs contribute equal edges. For a held-out positive pair `(u, v)` the direct edge is in the graph regardless of its label, so `same_community` cannot memorize the label; the remaining signal has to come from the surrounding community structure — which is exactly the kind of transductive signal v24/v25 already exploited, just lifted from pair-wise set intersections to a global node partition.

The jump was +0.00830 Kaggle from 6 new columns — the biggest improvement since v12's node-level transductive counts.

### v26c (regression: −0.00108, 0.88038 → 0.87930)

We tried to push the community idea further with ten more features in a single pass (percentile community sizes, internal density, multi-resolution Louvain same-community flags at resolutions 0.5 and 2.0, fraction of degree inside community, `comm_cn`). Of those, only `comm_cn` had a truly strong pos/neg gap (+0.272 vs. marginal gaps of +0.01 to +0.03 for the others). Bundling them all together *hurt* the gradient booster — the marginal columns diluted the strong signal. OOF went up by +0.00063 but Kaggle dropped by −0.00108.

### v26d (+0.00012, 0.88038 → 0.88050)

v26b *exactly* plus **only** `comm_cn` — nothing else. Minimal-risk single-feature addition. Small but positive Kaggle gain confirmed that the feature itself was real, not noise.

### v26e (Leiden, not submitted)

We ran Leiden (Traag et al. 2019) on the same candidate graph. It produced a structurally different partition (ARI vs. Louvain ≈ 0.21) but the per-pair signal was essentially identical — same positives, slightly different community boundaries. OOF gain only +0.00030. Not worth a submission slot.

### v26f (regression, not submitted)

v26d + **text-weighted Louvain** community + text-weighted `comm_cn` + text-weighted spectral. OOF barely moved (-0.00001) because the text spectral features actively hurt (-0.00065 alone) and the text `comm_cn` was redundant with the unweighted version.

### v26g (+0.00030, 0.88050 → 0.88080) — current 1st place

Apply the same discipline as v26d: keep only the proven winner from the v26f ablation. The winning family was the **three text-weighted Louvain community features**:

- Build the candidate graph with edge weight `1.0 + 3.0 × tfidf_cosine(u, v)`
- Run Louvain on it to get a **second** partition, independent of the v26d one
- Add `text_same_community`, `text_community_size_min`, `text_community_size_max` on top of the v26d feature set

ARI between the unweighted and text-weighted Louvain partitions is **0.063** — they are genuinely different. The text weighting biases clustering toward actors with similar Wikipedia keywords, giving Louvain a structurally different view of the same candidate edge set. Three new columns, +0.00030 Kaggle, new 1st place.

**v26 meta-lesson:** every gain in the v26 family came from **single-family additions with strict ablation discipline**. Every time we bundled multiple "promising" features together (v26c, v26f), we lost. With 10K training samples and a gradient booster, the right recipe is: one proven feature at a time, ablate alone first, add only if OOF strictly improves AND Spearman between the new and old predictions is <0.999.

---

## Reproducibility

The pipeline is deterministic. Running `experiments_v25.py` twice produced byte-identical `submission_v25.csv` files (max abs diff = 0). The v26 family preserves the same determinism guarantees: Louvain is called through `python-louvain` with an explicit `random_state=42` for both the unweighted and the text-weighted passes, and the Laplacian eigensolver (`scipy.sparse.linalg.eigsh` with `sigma=0` shift-invert) is deterministic.

What guarantees this:
- All model `random_state` are explicitly set (`SEED = 42` for HGB, CatBoost, plus `np.random.seed`)
- Seed offsets in the 30-seed ensembles are deterministic (`SEED + s * 31`)
- Louvain partitions use `random_state=42` for both the unweighted and the text-weighted candidate graphs
- Sets used in feature computation contain only integers; Python only randomizes string hashes, so set iteration order is stable
- No GPU, no multi-threaded RNG, no external network calls

To reproduce the current winning submission:

```bash
pip install numpy pandas scikit-learn scipy catboost networkx python-louvain
python experiments_v26g.py
```

Outputs `submission_v26g.csv` (the 0.88080 winning entry). Runs in ~1 minute on a laptop. `experiments_v26g.py` imports helpers from `best_solution.py`, `experiments_v25.py`, `experiments_v26b.py`, and `experiments_v26d.py`, so all five files must sit in the same directory.

## Strategy details

### Final solution — `experiments_v26g.py`

**Model:** HistGradientBoosting + CatBoost (30 random seeds each), rank-averaged 50/50.

**Features (66 total):**
1. **Graph topology (14):** degree (u, v, |Δ|, sum, log_u, log_v, min, max), common neighbors, Jaccard, Adamic-Adar, resource allocation, Sorensen, preferential attachment, same component, both isolated, hub promoted, hub depressed, paths of length 3 (with leakage correction), Katz approximation
2. **Text similarity (9):** raw dot/cosine, keyword Jaccard, TF-IDF cosine, TF-IDF L2, asymmetric overlap (u and v), neighborhood TF-IDF (uv, vu, nn — 1-hop GCN-style aggregation with leakage correction)
3. **Node-level transductive (6):** total/test counts for u and v + product and absolute diff of total counts
4. **Pair-level transductive — v24 (7):** shared test partners, shared train partners, shared total partners, Jaccard of test partners, Jaccard of all partners, min/max of test partner set size
5. **Pair-level transductive — v25 (8):** test/train/mixed triangles, Adamic-Adar and resource-allocation in test partner space, preferential attachment in test space, exclusive test partners (u and v)
6. **Feature interactions (5):** common neighbors × TF-IDF cosine, CN × cosine, PA × cosine, same_comp × TF-IDF, paths3 × TF-IDF
7. **Community (v26b, 3):** same_community, community_size_min, community_size_max on the unweighted candidate graph (label-free: all train pairs regardless of label + all test pairs + v25 pseudo-edges)
8. **Spectral (v26b, 3):** L2 distance, dot product, cosine between `u` and `v` in a k=16 Laplacian eigenmap of the unweighted candidate graph
9. **Community common neighbors (v26d, 1):** `comm_cn` — count of common neighbors of `u` and `v` that belong to `u`'s or `v`'s Louvain community
10. **Text-weighted community (v26g, 3):** same_community, community_size_min, community_size_max on a second Louvain pass over the candidate graph, this time with edge weights `1.0 + 3.0 × tfidf_cosine(u, v)` — ARI with the unweighted partition is ≈ 0.063, confirming the two partitions capture structurally different signals
11. **Self-training:** one conservative round at threshold 0.95 adds ~332 high-confidence test pairs as pseudo-edges, then graph-dependent features are recomputed

### v19 baseline — `best_solution.py`

Same as the final solution minus the 7 pair-level transductive features. Reaches 0.86766 on Kaggle. Built up over many iterations (v4 → v12 → v16 → v19), each adding a small set of carefully designed features and verifying the improvement transfers to Kaggle (not just OOF, since OOF is leaky from transductive counts).

### Earlier experiments — `hgb_tuned.py`

Iterative exploration of 16-feature HGB with self-training and blending strategies. Best score on this branch: 0.85373 (N2V mega blend). Plateaued well below v19 because it never adds the transductive node counts.

### Failed strategies (`xgb_hgb_lr_ensemble.py`, embeddings)

256-dimensional SVD adjacency + TF-IDF SVD embeddings combined with hand-crafted features in an XGB+HGB+LR ensemble. Scored 0.768 — embeddings overfit on the small dataset and dragged down the strong scalar features.

---

## Key lessons

1. **Feature discipline beats model complexity** on small datasets. Every embedding-based approach (GCN, Node2Vec, SVD, TabPFN) underperformed or flattened out.
2. **Local search has hard limits.** Every plateau (v19, v25, v26d) broke only when we added a fundamentally new information source, not when we tuned or ensembled the existing one.
3. **Transductive signal compounds across levels of abstraction.** Node-level test counts gave +0.011 (v19). Pair-level test partner intersections gave +0.003 more on top (v24/v25). Global community structure on the label-free candidate graph gave another +0.0083 on top of *that* (v26b). Each level answered a different question about "how close are these two actors in graph space" that the previous level couldn't express.
4. **Self-training works in moderation.** One round at threshold 0.95 helps. Multiple rounds or lower thresholds added noise and hurt every time.
5. **Symmetric features matter — but the model already learns them.** Augmenting (u, v) ↔ (v, u) at training and test time hurt slightly, suggesting the order signal in v19 features is not spurious.
6. **Single-family additions only.** v26c tried to add 10 new community features at once and regressed by 0.00108. v26f tried three text-weighted families at once and flatlined. Both times the fix was to strip back to the single proven column or family — v26d added only `comm_cn`, v26g added only the text-weighted Louvain community triplet.
7. **At the ceiling, models converge and blending stops helping.** TabPFN on v26d features scored OOF 0.9048 with Spearman 0.983 against the HGB+CatBoost predictions. The OOF-tuned blend gained only +0.00004 regardless of weight. Once the feature set dominates the learning problem, ensemble diversity is exhausted — the only way forward is more features.
8. **Change the graph, not the algorithm.** When unweighted Louvain plateaued, Leiden (ARI 0.21 vs. Louvain) gave the same per-pair signal. The breakthrough came from weighting the edges by TF-IDF cosine and running Louvain again (ARI 0.063 vs. the unweighted partition) — an actually different view of the data, not a different decoder of the same view.
