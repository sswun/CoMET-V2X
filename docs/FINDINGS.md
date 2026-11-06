# Key findings

All numbers: V2X-Seq SPD, strict arrival, official ground truth unless stated, AP at BEV IoU 0.5/0.7, mean over the four waits (0/100/200/400 ms). Differences in pp with 95% whole-sequence paired bootstrap CIs (2000 replicates). "Three heads" = mean over the three frozen readout heads. Every number below was recomputed by an independent implementation from saved detections; the file holding it is given in brackets.

## 1. The dataset's per-frame offset is not a reliable reference

[`results/dataset_audit/`]

* `system_error_offset` is exactly zero on 27.99% (val21), 21.76% (holdout14) and 25.58% (calibration8) of messages. Zero frames are almost exactly the frames without co-labelled objects, i.e. a placeholder.
* Roadside-only ground truth equals the public projection plus the official offset (residual about 4e-7 m). On zero frames it is therefore the uncorrected position. In calibration8, 35 zero frames still have a co-labelled object whose roadside and vehicle labels differ by 1.22 m (median).
* The non-zero offsets come from a bounded discrete search: 26 values per axis including 0, in five groups with step 0.04 m, bounds −1.1 and +0.66 m.
* At least one axis sits on a bound in 44.6% (val21), 42.5% (holdout14) and 77.7% (calibration8) of non-zero frames. On the x = −1.1 bound (2413 frames with ≥3 co-labelled objects) the frame's median co-labelled displacement is −1.374 m and 91.4% of frames lie beyond the bound. Displacements above 1.5 m occur in 19.6% of frames, official offsets in 1.07%.
* The official offset is a label-consistent reference, not a physical oracle; the generating algorithm is inferred from its value structure, not documented by the dataset authors.

## 2. CoMET (main method)

Main table, val21 [`results/tables_v3/table1.csv`]:

| Method | AP50 | AP70 |
|---|---:|---:|
| **CoMET (author detector, three heads)** | **78.87** | **66.63** |
| V2X-ViT, native, three detector seeds | 77.81 | 62.96 |
| F-Cooper, native | 76.94 | 62.18 |
| TraF, native | 75.27 | 61.11 |
| Stronger late fusion (column-wise max of two official late baselines) | 73.79 | 62.84 |

Five detector back-ends, val21 AP70, CoMET minus reference, three-head mean; brackets give the envelope of the three per-head CIs [`results/population_split/*/contrasts.json.gz`]:

| Detector | − no correction (PUBLIC) | − single-message mean | − previous anchor-gated KF (N0) |
|---|---:|---:|---:|
| author | +3.45 [1.82, 5.08] | +0.57 [−0.01, 1.10] | +0.82 [0.41, 1.27] |
| strict60 | +2.74 [1.64, 3.93] | +0.30 [0.00, 0.58] | +0.08 [−0.02, 0.22] |
| v2xvit_seed2024 | +2.38 [1.38, 3.37] | +0.49 [0.29, 0.72] | +0.19 [−0.04, 0.38] |
| v2xvit_seed2025 | +2.22 [1.23, 3.34] | +0.42 [0.23, 0.66] | +0.00 [−0.24, 0.19] |
| where2comm | +3.05 [1.99, 4.12] | +0.46 [0.27, 0.68] | −0.06 [−0.25, 0.11] |

* On co-labelled frames the gain over the single-message mean is larger (e.g. author +0.98, strict60 +0.53, all CIs above zero).
* On zero (placeholder) frames CoMET is penalised by the official table (e.g. author −3.42 vs PUBLIC), because the official ground truth there is uncorrected. This negative result is reported, not hidden.
* Robust external estimators with temporal smoothing (FreeAlign / CBM / RoCo, parameters chosen on calibration8): the best (CBM, window-3 median) reaches 64.00 AP70 vs 66.63 for CoMET (−2.62) [`results/tables_v3/R4_independent_read/`].
* CoAlign with CoMET poses instead of its K1 poses: +0.09 (CI crosses 0); vs its pose-graph poses: +1.05 to +1.11 (CI above 0) [`results/tables_v3/CoAlign_R3_independent/`].
* Re-training the readout head with placeholder frames removed does not help (calibration8 five-detector mean 65.66 vs 65.70); the original head is kept [`results/ablations/R6e_readout_head_decontamination/`].

## 3. Auxiliary ground truth (sensitivity only)

Label-median reference, CoMET minus single-message mean, AP70 [`results/aux_gt/R6f_label_median/`]:

| Detector | val21 | holdout14 |
|---|---:|---:|
| author | +0.905 [0.503, 1.351] | +1.424 [0.050, 2.419] |
| strict60 | +0.353 [0.087, 0.597] | +1.105 [0.163, 1.778] |

Interpolated-offset reference (`all_gaps`), author, CoMET − PUBLIC: val21 +4.00 [2.41, 5.69], holdout14 +8.73 [2.51, 12.52]; CoMET − single-message mean: +0.94 [0.58, 1.33] and +1.58 [0.18, 2.58] [`results/aux_gt/R6d_all_gaps/`]. These references are built from labels and are not physical truth.

## 4. Second tier: retraining native fusion with CoMET poses and cleaned labels

TraF fine-tuned from the released e60 weights for 2000 updates (reader and backbone frozen), three seeds; training labels of zero frames re-registered with the interpolated offset ("tier A", chosen on calibration8 over a de-saturated variant). Inference uses CoMET poses only [`results/tier2/`].

| TraF variant | val21 AP70 | val21 AP50 | holdout14 AP70 | holdout14 AP50 |
|---|---:|---:|---:|---:|
| same-budget PUBLIC fine-tune | 65.68 | 80.26 | 68.30 | 87.68 |
| online K1 poses, original labels | 66.18 | 81.19 | 72.15 | 90.09 |
| online K1 poses, cleaned labels | 66.44 | 81.36 | 72.73 | 89.94 |
| **CoMET poses, cleaned labels (tier 2)** | **66.49** | **81.44** | 71.45 | 89.59 |
| CoMET without retraining (tier 1) | 66.63 | 78.87 | 72.57 | 86.99 |

* Tier 2 − same-budget PUBLIC, val21: AP70 +0.81 [0.18, 1.62], AP50 +1.17 [0.58, 1.87].
* Tier 2 − tier 1, val21: AP50 +2.56 [1.12, 3.98], AP70 −0.14 [−1.29, 0.88] (on par). The second tier mainly adds recall.
* Most of the gain comes from retraining under a corrected pose; cleaned training labels add +0.25 AP70.
* Other architectures, tier 2 − same-budget PUBLIC on val21: CoAlign AP70 +1.63 [1.20, 2.10], AP50 +1.75 [1.03, 2.62]; Where2comm AP70 +0.24 [−0.16, 0.54], AP50 +0.32 [−0.29, 0.84].
* On both auxiliary references tier 2 − same-budget PUBLIC stays positive on val21 (AP70 +1.19 and +1.27, AP50 +1.79 and +2.29, intervals above zero) [`results/aux_gt/R6h_tier2/`].

## Reference implementation check

`comet/estimator.py` reproduces the reference per-message offsets, variances and anchor counts exactly (maximum absolute difference 0) for all five detectors on all four splits, 205,202 messages in total [`results/verification/verify_against_recipe.json`].
