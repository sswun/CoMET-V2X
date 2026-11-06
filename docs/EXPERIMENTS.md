# Experiment index

Each row: the question, the entry script(s) in `research_code/preexperiment/research_autopilot/` (unless another folder is given), and where the results are. "Independent" scripts recompute matching, AP and bootstrap separately from the producer.

## Main method

| ID | Question | Entry scripts | Results |
|---|---|---|---|
| R1 | Final CoMET recipe (all-ego anchors, mutual-NN Hungarian ≤2.5 m, KF, no gating) on five detectors; ablations U0–U6, H1–H3, N2; q and τ chosen on calibration8 | `run_peer19_recipe_v3_20261104.py`, `recover_peer19_recipe_interface_v2_20261104.py`; independent `recover_peer19_committed_review_v4_20261105.py`, `recover_peer19_strict_review_v5_20261105.py`; CI `consume_peer19_R1_independent_CI_v1_20261104.py` | `results/tables_v3/R1_independent_CI/`, `H1_cross_detector*.csv`, `learning_untrained_H1_ablation.csv.gz`, `H1_sensitivity.csv.gz`, `H1_cal4_selection.csv`, `H1_strict_cal4.csv`, `H1_readout_fitcal_comparison.csv.gz` |
| Main table | CoMET vs native and late baselines, val21 | `build_peer19_final_tables_v3_20261104.py`, `recover_peer21_v3_row_identity_v1_20261105.py`; baselines `run_peer16_E5_seed_baselines_v1_20261103.py`, `review_peer15_external_raw_AP_v1_20261103.py` | `results/tables_v3/table1.csv` |
| Direct | Paired CoMET − every reference | `compare_peer19_H1_direct_v1_20261104.py` | `results/tables_v3/H1_direct_CI.csv` |
| Temporal | KF vs single message, hold-last, EMA, window | `run_peer19_H1_assets_v1_20261104.py` | `results/tables_v3/H1_temporal_aggregation.csv`, `H1_temporal_CI.csv` |
| R3 | CoAlign with CoMET poses instead of K1 / pose graph | `run_peer19_coalign_H1_v1_20261104.py` | `results/tables_v3/CoAlign_R3_independent/` |
| R4 | Robust external estimators with temporal smoothing | `run_peer19_robust_external_v1_20261104.py`, `consume_peer19_R4_independent_read_v1_20261104.py` | `results/tables_v3/R4_independent_read/`, `H1_robust_external*.csv` |
| Mechanism A1–A5 | retention, external per-frame error, GT source, autocorrelation, anchor counts, source AP | `run_peer19_H1_assets_v1_20261104.py`, `run_peer20_mechanism_v1_20261105.py`, `run_peer20_ACF_cluster_v1_20261105.py` | `results/tables_v3/H1_A*.csv`, `H1_anchor_statistics.csv`, `H1_source_AP.csv`, `H1_fixed_FP_source_TP.csv`, `results/mechanism/` |
| Boundaries | OPV2V pose-time table and V2V4Real boundary | `run_peer19_H1_boundaries_v1_20261104.py` | `results/tables_v3/H1_opv_boundary.csv`, `H1_v2_boundary.csv` |
| KF trajectories | per-message filter state for figures (our outputs only) | `consume_peer19_H1_paper_supplement_v1_20261104.py` | `results/mechanism/KF_trajectories/` |

## Dataset reference audit

| ID | Question | Entry scripts | Results |
|---|---|---|---|
| R6a | Is the zero offset a placeholder, and how is roadside-only GT built? | `audit_peer20_raw_labels_v1_20261105.py`; independent `review_peer20_raw_labels_v2_20261105.py` | `results/dataset_audit/R6a_raw_label_audit/` |
| R6g | Offset value grid, bound saturation, displacement beyond the bound | `run_peer21_offset_grid_v1_20261105.py` | `results/dataset_audit/offset_grid/` |

## Population split and auxiliary GT

| ID | Question | Entry scripts | Results |
|---|---|---|---|
| R6b | All main contrasts on all / zero / co-labelled / non-zero frames, five detectors | `run_peer20_population_v1_20261105.py`, `consume_peer20_core_statistics_v1_20261105.py` | `results/population_split/` |
| R6d | Interpolated-offset auxiliary GT (`gap1s`, `gap3s`, `all_gaps`) | `run_peer20_auxiliary_GT_v1_20261105.py` | `results/aux_gt/R6d_all_gaps/` |
| R6e | Re-train the readout head without placeholder frames | `run_peer20_clean_head_v1_20261105.py`, `consume_peer20_clean_choice_v1_20261105.py` | `results/ablations/R6e_readout_head_decontamination/` |
| R6f | Label-median auxiliary GT | `run_peer21_label_median_GT_v1_20261105.py` | `results/aux_gt/R6f_label_median/` |

## Second tier

| ID | Question | Entry scripts | Results |
|---|---|---|---|
| R7 | TraF with K1 poses and cleaned labels vs original K4 and PUBLIC | `run_peer20_K4_decontamination_v1_20261105.py`, `run_peer20_K4_environment_v2_20261105.py`, `consume_peer20_R7_v1_20261105.py`; trainer `run_peer10_k4_native_finetune_v1_20261101.py`, evaluator `run_peer10_native_evaluate_v1_20261101.py` | `results/tier2/R7_TraF_clean_K1/` |
| R7b | TraF with CoMET poses and cleaned labels (tiers A/B), pre-registered gate | `run_peer21_clean_H1_native_v1_20261105.py`, `consume_peer21_R7b_v1_20261105.py`, `review_peer21_R7b_gate_v1_20261105.py` | `results/tier2/R7b_TraF/`, `results/tier2/tables/` |
| R7c | Same recipe on Where2comm and CoAlign | `run_peer21_R7c_clean_cross_v1_20261105.py`, `consume_peer21_R7c_cross_v1_20261105.py` | `results/tier2/R7c_cross_architecture/`, `results/tier2/tables/` |
| R6h | Tier 2 on both auxiliary references | `run_peer22_R6h_v1_20261106.py` | `results/aux_gt/R6h_tier2/` |
| R7d | Training-budget dose-response (2000 / 8000 / 20000 updates) | not included | — |

## Shared infrastructure

* Strict-arrival loader: `research_code/preexperiment/research_baselines/traf_native4_recorded_arrival_20261010.py`.
* Cached detector materials and readout: `run_peer16_cached_methods_v1_20261103.py`, `run_peer15_author_recipe_v1_20261103.py`.
* Association used by CoMET: `match()` in `run_peer16_E1_timing_audit_v1_20261103.py` (re-implemented in `comet/estimator.py`).
* Polygon matcher and AP used for independent verification: `research_code/preexperiment/research_baselines/review_traf_v2xseq_adapted_fixed60_raw_20261006.py`.
* Several modules carry historical names (for example `qualify_traf_native4_cpu_v2_20261010.py` builds the TraF model; `peer1x_cycleNN_*.py` hold path constants). They are kept only because the released scripts import them.
