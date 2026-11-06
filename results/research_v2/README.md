# Research extensions after v0.1.0

Contact the authors and obtain written permission before using these experimental results. [Permission terms](../../LICENSE)

These experiments extend the released method. The manuscript reports them (lock acquisition, two-tier combination and the injected pose-error study); they do not change any number of the main tables in `results/tables_v3/`. Each one was pre-registered with its parameters and adoption gate fixed before the validation splits were read. A frozen script produced it, and an independent implementation recomputed it. Only accepted results are included.

| Folder | Experiment | Status |
|---|---|---|
| `R8d_acquisition/` | CoMET-A: lost-lock detection and coarse re-acquisition for offsets beyond the 2.5 m association gate | Both pre-registered gates passed |
| `R8c_stress/` | Controlled pose-error injection on five late-fusion detectors and on native TraF fusion | Descriptive (no gate) |
| `R8e_two_tier/` | Box-level combination of tier 1 (CoMET late fusion) and tier 2 (TraF retrained with CoMET poses) | All pre-registered checks passed |

The common fields (`role`, `wait_ms`, `threshold`, CI conventions) follow [`results/README.md`](../README.md). AP is vehicle BEV AP against the official ground truth, averaged over three frozen readout heads (late fusion) or three detector training seeds (native fusion), and four arrival delays. Intervals are 95% whole-sequence paired bootstrap intervals with 2000 replicates. `sources.json` files record the producing artifacts (workspace path, SHA-256, bytes); those artifacts are not part of this repository.

## R8d: CoMET-A

[Implementation](../../comet/acquisition.py). A stream is declared lost after five consecutive messages in which both agents have at least three boxes but no anchor lies within the gate. Ego–remote displacements of the last five messages are then voted in a 0.5 m histogram. The coarse offset is accepted only with at least 6 supporting pairs from 3 messages, a norm of at least 2 m, and a peak 1.5 times the next separate peak. Association then runs on the remote boxes shifted by this fixed coarse offset, and the Kalman filter tracks the remainder. All parameters were fixed in advance and not tuned.

**Injected error, val21 AP70** (each message's remote boxes shifted by a known translation in a per-sequence random direction; GT unchanged):

| Detector | Injected | Online pose | CoMET | CoMET-A | CoMET-A − CoMET [95% CI] | Retained vs. clean CoMET |
|---|---:|---:|---:|---:|---:|---:|
| author | 3 m | 61.86 | 62.70 | **65.63** | +2.93 [1.31, 4.83] | 98.5% |
| author | 5 m | 61.77 | 61.84 | **66.29** | +4.45 [2.85, 6.16] | 99.5% |
| strict60 | 3 m | 55.19 | 55.89 | **57.62** | +1.73 [0.71, 2.96] | 97.7% |
| strict60 | 5 m | 55.07 | 55.15 | **58.62** | +3.48 [2.30, 4.73] | 99.4% |

**Real data (no injection):** across five detectors, val21 AP70 changes by −0.0006 on average relative to CoMET, and mechanism_holdout14 is identical. Acquisition triggers 8 times on author val21 and once on strict60 val21, and never on the other 13 detector/split cells. Streams where it never triggers are bit-identical to CoMET.

Files: `results.csv` (96 rows: real and injected cases, CoMET-A, CoMET, single-message mean and online pose); `adoption_gate.json` (gate checks and panels).

## R8c: pose-error stress test

The same injection harness ([`comet/stress.py`](../../comet/stress.py)) is used, with seed 2026111401. Profiles: T1 constant 0.5/1/2/3/5 m; T2 linear drift to 2 m over the sequence; T3 a 2 m step at the sequence midpoint. Author and strict60 cover all profiles on three splits; the other three detectors cover T1 1/2/3 m on val21.

**Late fusion, val21 AP70, T1 constant error:**

| Detector | Method | 0.5 m | 1 m | 2 m | 3 m | 5 m |
|---|---|---:|---:|---:|---:|---:|
| author | online pose | 62.83 | 62.34 | 61.87 | 61.86 | 61.77 |
| author | CoMET | 66.49 | 66.28 | 64.89 | 62.70 | 61.84 |
| author | CoMET-A | 66.48 | 66.28 | 65.64 | 65.63 | 66.29 |
| strict60 | online pose | 56.04 | 55.63 | 55.18 | 55.19 | 55.07 |
| strict60 | CoMET | 58.78 | 58.52 | 57.52 | 55.89 | 55.15 |
| strict60 | CoMET-A | 58.78 | 58.53 | 57.83 | 57.62 | 58.62 |

At 3 m the CoMET-A − CoMET gain is +1.44 [0.62, 2.47] for V2X-ViT seed 2024, +1.44 [0.68, 2.41] for seed 2025 and +1.97 [0.92, 3.30] for Where2comm. Under drift (T2) and a 2 m step (T3), CoMET already tracks the error, and CoMET-A adds +0.01 to +0.13 AP70 on val21.

**Native TraF fusion, val21 AP70** (detector seed 20261101). The pose given to the network is either the injected online pose or the pose corrected by CoMET-A from the injected late-fusion boxes:

| Model | Injected | Uncorrected | CoMET-A pose | Δ [95% CI] |
|---|---:|---:|---:|---:|
| same-budget PUBLIC fine-tune | 1 m | 63.58 | 65.74 | +2.15 [1.49, 2.92] |
| same-budget PUBLIC fine-tune | 2 m | 62.72 | 65.46 | +2.74 [1.79, 3.78] |
| same-budget PUBLIC fine-tune | 3 m | 62.67 | 65.11 | +2.44 [1.58, 3.32] |
| tier 2 (CoMET poses, cleaned labels) | 1 m | 61.91 | 66.32 | +4.41 [2.78, 6.34] |
| tier 2 (CoMET poses, cleaned labels) | 2 m | 61.99 | 65.90 | +3.90 [2.49, 5.32] |
| tier 2 (CoMET poses, cleaned labels) | 3 m | 62.11 | 66.41 | +4.29 [2.67, 6.12] |

AP50 intervals are also above zero for all six cases. Under a 2 m step, author val21 streams return within 0.5 m of their clean state in 42 of 84 cases with CoMET-A and in 25 of 84 with CoMET. The remaining streams are right-censored at the end of the sequence.

Files: `late/<detector>/results.csv` (AP, CI, contrasts to CoMET, recovery relative to the online pose, retention relative to clean CoMET); `late/<detector>/T3_recovery.json.gz`; `native/results.csv` and `native/per_wait.json`; `capture/` (per-stream share without any in-gate anchor, all streams and first five messages; T3 messages to first return within 0.5 m). Recovery errors are measured against the clean CoMET state, not physical truth.

## R8e: two-tier combination

[Implementation](../../comet/fusion.py). Tier-1 and tier-2 boxes are paired one-to-one by BEV IoU ≥ 0.3. A pair keeps the tier-1 box and the larger score. An unpaired box keeps its score scaled by its tier's calibration8 coefficient (tier 1: 0.389, tier 2: 1.0; `coefficients.json`), followed by BEV NMS at 0.15. The four candidate settings (IoU 0.3/0.5, with or without isotonic score calibration) were compared on calibration8 only (`calibration_choice.json`).

| Split | Metric | Tier 1 | Tier 2 | **Two-tier** | − tier 1 [95% CI] | − tier 2 [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| val21 | AP70 | 66.63 | 66.49 | **67.85** | +1.23 [0.42, 2.06] | +1.36 [0.71, 2.18] |
| val21 | AP50 | 78.87 | 81.44 | **82.62** | +3.75 [2.20, 5.34] | +1.18 [0.67, 1.70] |
| mechanism_holdout14 | AP70 | 72.57 | 71.45 | **73.28** | +0.71 [0.49, 1.04] | +1.84 [0.25, 2.53] |
| mechanism_holdout14 | AP50 | 86.99 | 89.59 | **90.01** | +3.02 [2.24, 4.00] | +0.42 [−0.06, 1.63] |

Pre-registered gate: val21 AP70 at least +0.50 above the better tier with CI above zero, hold-out AP70 above both tiers, val21 AP50 no lower than tier 2 − 0.20. All checks passed. Tier 1 is the author detector's CoMET late fusion (heads 2023/2024/2025), and tier 2 is TraF R7b A with 2000 updates (seeds 20261101/02/03), paired by position. Files: `results.csv`, `adoption_gate.json`.

## Verification

`comet/acquisition.py`, `comet/stress.py` and `comet/fusion.py` reproduce the accepted outputs exactly. The checks cover all 15 real-data CoMET-A state streams, all 8 accepted injected cases (injected shifts and states, maximum absolute difference 0), and the two-tier fused scores, origins and candidate indices of every accepted val21 and hold-out seed. [Report](../verification/verify_research_v2.json) · [Tool](../../tools/verify_research_v2.py)
