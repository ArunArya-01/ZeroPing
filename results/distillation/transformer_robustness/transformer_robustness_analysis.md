# Phase 2 — Understanding Transformer Robustness

**Date:** 2026-07-31
**Status:** Analysis only — no training; frozen Large MLP + FT-Transformer + Teacher

## Central question

Why does architecture ranking reverse?

| Protocol | Ranking |
|----------|---------|
| Flight Holdout | Teacher > Large MLP > XLarge > FT |
| Type-macro | Teacher > **FT** > Large MLP > XLarge |

---

## Overall metrics (this re-run on Final)

| Model | Final RMSE | Type-macro | Body-macro |
|-------|-----------:|-----------:|-----------:|
| Large MLP | 215.85 | 270.61 | 239.63 |
| FT-Transformer | 224.12 | 261.15 | 249.58 |

Fraction of intervals with lower |error| for FT: **0.514**

Mean |err| FT − Large: overall **-2.52**, rare **-16.55**, common **-2.47**, heavy **-11.49**, narrow **+0.57**

---

## Study A — Representation geometry

| Metric | Large MLP | FT-Transformer |
|--------|----------:|---------------:|
| Silhouette (type) | 0.0375 | -0.0141 |
| Davies–Bouldin (↓ better) | 4.8309 | 5.1902 |
| Mean intra-type dist | 35.1882 | 14.0303 |
| Mean inter-type dist | 44.6179 | 18.3604 |
| Inter/intra ratio | 1.2680 | 1.3086 |
| Local type consistency (5-NN) | 0.9407 | 0.9368 |
| Mean dist (rare → common centroids) | 21.7952 | 7.0180 |
| PCA var explained (PC1, PC2) | [0.6377311746142104, 0.1279358537151526] | [0.7030922452317864, 0.2422722218219421] |

Embeddings: penultimate layer (MLP backbone; FT CLS after LayerNorm). Metrics on standardized embeddings; silhouette uses up to 5k subsample.

### Figures (geometry)

![pca L type](figures/fig_p2_pca_large_type.png)

![pca FT type](figures/fig_p2_pca_ft_type.png)

![umap L](figures/fig_p2_umap_large_type.png)

![umap FT](figures/fig_p2_umap_ft_type.png)

![tsne L](figures/fig_p2_tsne_large_type.png)

![tsne FT](figures/fig_p2_tsne_ft_type.png)

![geom](figures/fig_p2_geometry_metrics.png)

---

## Study B — Error localization

### Types where FT has lower type-RMSE

[np.str_('B772'), np.str_('B38M'), np.str_('A333'), np.str_('B789'), np.str_('B788'), np.str_('A321')]

### Types where Large has lower type-RMSE

[np.str_('A21N'), np.str_('B738'), np.str_('B744'), np.str_('A332'), np.str_('A20N'), np.str_('A320'), np.str_('A359'), np.str_('A319'), np.str_('B77W')]

Full tables: `results/distillation/transformer_robustness/error_by_*.csv`

![err type](figures/fig_p2_error_by_type.png)

![err body](figures/fig_p2_error_by_body.png)

![err phase](figures/fig_p2_error_by_phase.png)

![err rare](figures/fig_p2_error_by_rare.png)

![err dur](figures/fig_p2_error_by_duration_bin.png)

---

## Study C — Feature utilization

Method: mean |grad × input| on numeric features (n≈2500 samples), L1-normalized.

| Stability / agreement | Value |
|----------------------|------:|
| Large half-split Spearman | 0.989 |
| FT half-split Spearman | 0.991 |
| Cross-model Spearman | 0.554 |
| Large physics-like share | 0.642 |
| FT physics-like share | 0.380 |

Top features by |Large−FT| attribution:

| Feature | Large | FT | |diff| |
|---------|------:|---:|------:|
| duration_s | 0.1462 | 0.4055 | 0.2593 |
| r3_tow_kg | 0.0090 | 0.0627 | 0.0537 |
| r3_mass_consumed_kg | 0.0765 | 0.0308 | 0.0457 |
| r3_mass_rate_kgps | 0.0494 | 0.0054 | 0.0441 |
| physics_fuel_kg | 0.0479 | 0.0051 | 0.0428 |
| r3_fuel_fraction | 0.0756 | 0.0338 | 0.0419 |
| end_fraction_of_flight | 0.0073 | 0.0478 | 0.0405 |
| ref_mass_kg | 0.0292 | 0.0033 | 0.0259 |
| r3_tow_mtow_ratio | 0.0314 | 0.0059 | 0.0255 |
| median_altitude | 0.0124 | 0.0373 | 0.0249 |
| r3_mass_std_kg | 0.0769 | 0.0571 | 0.0198 |
| r3_cruise_mass_fuel_ratio | 0.0196 | 0.0009 | 0.0187 |

![attr](figures/fig_p2_feature_attribution.png)

---

## Evidence-supported hypotheses

### H1_type_clustered_ft — **not_supported**

**Claim:** FT embeddings are more type-separated than Large.

**Evidence:** silhouette FT=-0.014 vs Large=0.038

### H2_smoother_geometry — **supported**

**Claim:** FT has higher inter/intra type distance ratio (more separable types).

**Evidence:** inter/intra FT=1.309 Large=1.268

### H3_ft_helps_rare_or_heavy — **supported**

**Claim:** FT reduces error more on rare and/or heavy types than on common/narrow.

**Evidence:** Δ|err| FT-Large rare=-16.55 common=-2.47 heavy=-11.49 narrow=+0.57

### H4_type_macro_from_hard_types — **supported**

**Claim:** FT type-macro gain comes from improving a subset of hard aircraft types.

**Evidence:** types FT better type-RMSE: [np.str_('B772'), np.str_('B38M'), np.str_('A333'), np.str_('B789'), np.str_('B788'), np.str_('A321')]

### H5_feature_use_differs — **supported**

**Claim:** MLP and FT emphasize different numeric features (attribution rank correlation < 0.7).

**Evidence:** Spearman attr Large vs FT=0.554; physics share L=0.642 FT=0.380

---

## Synthesis: why the ranking reverses

Evidence-based synthesis (not speculation beyond measurements):

1. **Overall vs entity-equal metrics.** Large wins on frequency-weighted Final RMSE; FT wins when each aircraft type is weighted equally (type-macro). That alone can reverse rankings if FT is relatively better on lower-frequency / higher-error types even when worse on dominant narrow-bodies (A20N/A320 mass).

2. **Where FT gains.** See type list and rare/heavy Δ|err| above — interpret relative to Large, not absolute teacher.

3. **Representation structure.** Compare silhouette / inter-intra / local consistency between Large and FT (Study A table). Higher type separation or consistency supports a geometry story; lower does not.

4. **Feature use.** Cross-model attribution correlation and physics-share differences (Study C) indicate whether the models rely on different numeric cues.

5. **Not explained by training new weights.** Both models are frozen KD students on the same α/β and data; differences are architectural inductive bias + representation geometry under the same supervision.

---

## Open questions & limitations

- Post-hoc type-macro ≠ re-trained leave-one-type-out.
- Grad×input is a local linearization; not causal feature importance.
- Attention maps not fully dissected (FT token attention avg left for follow-up).
- Embedding metrics are sensitive to standardization and class imbalance handling.
- XLarge not re-analyzed (ranking already known from Phase 0).
- No new models trained; mechanisms are descriptive.

---

## Artifacts

`results/distillation/transformer_robustness/`

*Generated 2026-07-31T01:34:50.795633+00:00*
