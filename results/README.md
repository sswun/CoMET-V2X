# Results

Contact the authors and obtain written permission before using these experimental results, including in benchmarks, figures and publications. [Permission terms](../LICENSE) · [Contact the authors](https://github.com/sswun/CoMET-V2X/issues/new)

Only accepted results are included: each was produced by a frozen script and recomputed by an independent implementation (matching, AP and bootstrap) from saved detections. Failed runs, engineering qualifications, previews and superseded table versions are excluded. Files ending in `.gz` are gzip-compressed copies of the same CSV/JSON.

| Folder | Contents |
|---|---|
| `tables_v3/` | Final paper tables (long format: `row`, `metric`, `value`, `review_status`, `scope`). `table1.csv` is the main table. |
| `provenance/tables_v3/` | The same tables with the `source_operands` column (source file, SHA-256 and operands of every number). |
| `dataset_audit/` | R6a raw-label audit (aggregate and per-sequence counts) and R6g offset grid/saturation tables. |
| `population_split/<detector>/` | R6b paired contrasts (`contrasts.json.gz`) and A1 retention on all / zero / co-labelled / non-zero frames. Entries are per readout-head seed. |
| `aux_gt/` | R6d (`all_gaps`), R6f (`label_median`) and R6h (second tier on both references). Sensitivity only. |
| `ablations/` | R6e readout-head decontamination decision. |
| `mechanism/` | A5 autocorrelation with cluster bootstrap; per-message KF trajectories of the author detector. |
| `tier2/` | R7, R7b (TraF), R7c (Where2comm, CoAlign): decisions, contrasts, per-seed statistics, paper tables; R7d training-budget sweep. |
| `research_v2/` | Extensions after v0.1.0: R8d CoMET-A, R8c pose-error stress test, R8e two-tier combination ([details](research_v2/README.md)). Paper tables unchanged. |
| `verification/` | `comet/` vs accepted recipe states, CoMET-A states and two-tier detections (all equal). |

## Common fields

* `role`: split (`fit24`, `calibration8`, `val21`, `mechanism_holdout14`).
* `wait_ms`: 0, 100, 200, 400 or `four_wait_mean`.
* `threshold`: BEV IoU threshold ×100 (50 or 70).
* `population`: `all`, `zero`, `colabelled`, `nonzero` (see `docs/PROTOCOL.md`).
* `delta_pp`, `CI95`: difference in percentage points and its 95% whole-sequence paired bootstrap interval (2000 replicates).
* Paths inside files (`artifacts/research_autopilot/...`) refer to the original research workspace and are kept as provenance; those artifacts are not part of this repository.
