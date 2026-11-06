# Evaluation protocol

## Data and splits

V2X-Seq SPD (vehicle-infrastructure cooperative sequences). Sequences are divided into four disjoint groups:

| Split | Messages | Role |
|---|---:|---|
| `fit24` | 3762 | fitting readout heads and fine-tuning in the second tier |
| `calibration8` | 1247 | the only split used for any selection (q, score threshold τ, head kind, label tier, budget) |
| `mechanism_holdout14` | 2436 | held-out report split; inside the detectors' training domain |
| `val21` | 3316 | official validation sequences; the only split unseen by every detector |

`fit24`, `calibration8` and `mechanism_holdout14` are inside the official training split, so detectors have seen them; `val21` is the primary report split. Numbers are reported per split, never pooled.

## Strict arrival

At decision time `t` the vehicle may use only roadside messages captured at `τ ≤ t − d`, `d ∈ {0, 100, 200, 400} ms`. Poses are public calibrations plus online vehicle poses. Ground truth, future frames and the dataset's per-pair `system_error_offset` are never used at inference. Each wait setting is replayed independently.

## Metrics

* AP at BEV IoU 0.5 and 0.7 (`AP50`, `AP70`), vehicle class, common ROI, official ground truth.
* Unless noted, a reported number is the mean over the four waits, and, where applicable, over three readout heads (frozen heads with seeds 2023/2024/2025) or three training seeds.
* Differences are in percentage points (pp).
* 95% confidence intervals: whole-sequence paired bootstrap, 2000 replicates; draws are paired before averaging over seeds and waits.
* "Independent" means matching, AP and bootstrap were recomputed by a separate implementation from the saved detections; training and detections are shared.

## Ground-truth populations (post-hoc reading only)

The official offset is exactly zero on frames without co-labelled objects (a placeholder), and the roadside-only ground truth equals public projection plus this offset. Results are therefore also reported on populations defined by the labels:

* `all`: every frame (primary);
* `zero`: frames whose official offset is exactly zero;
* `colabelled`: frames with at least one co-labelled object;
* `nonzero`: complement of `zero`.

The populations are not complements of each other (`calibration8` has frames that are both zero and co-labelled). They are defined by annotations, used for reading only, never as inference input and never for selection.

## Auxiliary ground truth (sensitivity only)

* `all_gaps` (R6d): roadside-only boxes on zero frames re-registered with the per-sequence interpolation of non-zero official offsets.
* `label_median` (R6f): roadside-only boxes re-registered with the frame's median co-labelled displacement when the frame has at least three co-labelled objects; otherwise the official or interpolated offset.

Both are label-derived references, not physical truth, and never replace the official table.

## Selection rules

* Every hyper-parameter, head kind, label tier and training budget is chosen on `calibration8` official AP70 only, before `val21` or `mechanism_holdout14` is read.
* Exact ties go to the simpler or pre-declared default.
* Rules that withhold correction on frames without anchors ("hold-off") are rejected, because their gain on the official table comes from placeholder frames.
