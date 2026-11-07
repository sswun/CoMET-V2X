# Native-fusion retraining results

These eight CSV files collect 4,672 rows from the accepted evaluations. Tier 2 uses CoMET poses, tier-A cleaned labels and 2,000 training updates. The tables preserve the AP values and paired confidence intervals from the underlying runs. [Permission terms for using results](../../../LICENSE)

## Validation results

Vehicle BEV AP on `val21`, averaged over the four arrival delays. The comparison uses the same training budget for the online-pose baseline (`PUBLIC`). Differences are in percentage points with 95% paired sequence-bootstrap intervals.

| Architecture | Tier 2 AP70 / AP50 | Online pose AP70 / AP50 | Δ AP70 [95% CI] | Δ AP50 [95% CI] |
|:---|---:|---:|---:|---:|
| TraF-Align | 66.491 / 81.437 | 65.677 / 80.264 | +0.814 [0.181, 1.624] | +1.172 [0.579, 1.869] |
| Where2comm | 63.289 / 77.284 | 63.045 / 76.964 | +0.244 [−0.159, 0.542] | +0.320 [−0.286, 0.839] |
| CoAlign fusion stage | 58.095 / 73.880 | 56.467 / 72.132 | +1.629 [1.200, 2.104] | +1.749 [1.026, 2.624] |

Against CoMET late fusion (tier 1), TraF-Align tier 2 changes AP70 by −0.137 on `val21` and −1.122 on `mechanism_holdout14`. The CoAlign experiment changes the pose input to the fusion stage. Its accompanying `strict60` late-fusion reference uses a different detector, so the archive reports their absolute scores separately. All model and label choices use `calibration8`.

## Files and provenance

| Files | Contents |
|:---|:---|
| `traf_seed_statistics.csv`, `traf_mean_statistics.csv` | TraF per-seed and mean results for online poses, anchor-gated poses, both CoMET label tiers and tier 1. Includes the two report splits, four populations and each wait. |
| `traf_existing_paired_CI.csv` | 640 existing tier-A/B contrasts and 32 cleaned-label K1 contrasts. |
| `cross_seed_statistics.csv`, `cross_mean_statistics.csv` | TraF-Align, Where2comm and CoAlign at the same training budget. `ONLINE` denotes the earlier K1 pose. |
| `cross_existing_paired_CI.csv` | 160 contrasts for the two transfer architectures and 120 corresponding TraF contrasts. |
| `first_tier_seed_statistics.csv`, `first_tier_mean_statistics.csv` | Tier-1 scores. The Where2comm entries use all-frame AP from R1. |

`source_rows` records each source's workspace-relative path, SHA-256, byte count and **zero-based JSON row index**. `values_json` contains the source values or operands. `SD` describes variation across three seeds. Paired confidence intervals come from the sequence bootstrap. Frozen readout-head seeds and native detector training seeds have separate fields.

The [snapshot manifest](manifest.json) records the status when these tables were assembled. The [auxiliary-reference results](../../aux_gt/) are archived separately. The training-budget sweep (R7d) was added afterwards: `R7d_formal_curves.csv.gz` (1200 rows, 2000/8000/20000 updates on val21 and mechanism_holdout14) and `R7d_A_calibration.csv.gz` (360 rows, the calibration8 selection), with `R7d_availability.json` and the decision in [`../R7d_TraF_budget/`](../R7d_TraF_budget/). On val21, A-tier AP70 is 66.49 / 64.68 / 60.55 and same-budget PUBLIC 65.68 / 64.12 / 60.10; calibration8 selects 2000. PUBLIC was not evaluated on calibration8.
