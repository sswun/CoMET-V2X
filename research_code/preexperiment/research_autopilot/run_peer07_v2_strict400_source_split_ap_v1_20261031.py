#!/usr/bin/env python3
"""V2 strict400 provenance AP, reusing immutable saved predictions/GT."""
from __future__ import annotations
import argparse
from collections import defaultdict
import os
from pathlib import Path
import resource
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "preexperiment/research_autopilot"), str(ROOT / "preexperiment/research_baselines")]
import run_v2_label_provenance_strict400_native_attribution_v1_20261029 as old
from peer07_source_ap_producer_utils_v1_20261031 import atomic_json, evidence, load, metric, save_npz
from e1_receiver_value_core_v1_20261020 import greedy, iou32

GROUPS = ("all", "local_visible", "remote_exclusive")
THRESHOLDS = (.5, .7)


def outcomes(matrix, target, threshold):
    match = greedy(matrix[:, target], np.arange(len(matrix)), threshold)
    tp = (match >= 0).astype(np.int8)
    ignored = np.zeros(len(matrix), np.int8)
    if (~target).any():
        ignored[tp == 0] = (matrix[tp == 0][:, ~target].max(axis=1, initial=0.) >= threshold).astype(np.int8)
    return tp, ((tp == 0) & (ignored == 0)).astype(np.int8), ignored


def bootstrap(rows, methods):
    table = {(r["role"], r["sequence"], r["source_group"], r["method"]): r for r in rows}
    rng = np.random.default_rng(20261031)
    result = []
    for role in old.ROLES:
        sequences = sorted({r["sequence"] for r in rows if r["role"] == role})
        for group in GROUPS:
            eligible = [s for s in sequences if table[role, s, group, "native"]["gt"] > 0]
            for method in methods[1:]:
                values = np.array([table[role, s, group, method]["ap70"] - table[role, s, group, "native"]["ap70"] for s in eligible])
                samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
                result.append({"role": role, "source_group": group, "left": method, "right": "native",
                    "eligible_sequences": len(values), "sequence_mean_delta_ap70_pp": float(values.mean()),
                    "bootstrap_seed": 20261031, "bootstrap_replicates": 10000,
                    "ci95_low_pp": float(np.quantile(samples, .025)), "ci95_high_pp": float(np.quantile(samples, .975)),
                    "probability_positive": float((samples > 0).mean())})
    return result


def compute(contract_path, output):
    started, cpu = time.monotonic(), time.process_time()
    contract = load(contract_path)
    for spec in contract["inputs"].values():
        if evidence(ROOT / spec["path"]) != spec: raise RuntimeError(("input changed", spec))
    output.mkdir(parents=True, exist_ok=False)
    lineage, _ = old.lineage_map(ROOT / contract["lineage"]["path"])
    panels, sequence_panels, coverage, arrays = [], [], {}, {}
    for role in old.ROLES:
        frames = old.read_predictions(ROOT / contract["roles"][role]["predictions"], tuple(contract["methods"]))
        references = old.source_panels(ROOT / contract["roles"][role]["source_summary"])
        labels, frame_sequences, offsets = [], [], [0]
        targets_by_frame = []
        for (sequence, timestamp), rows in sorted(frames.items()):
            frame = lineage[("train" if role == "mechanism_holdout14" else "val", sequence, timestamp)]
            gt = np.asarray(rows["native"]["gt_boxes"], np.float32).reshape(-1, 7)
            if not np.array_equal(gt, frame["boxes"]): raise RuntimeError("lineage GT identity")
            categories = frame["categories"]
            if len(categories) != len(gt) or set(categories) - {"ego_only", "common_explicit_link", "remote_only"}:
                raise RuntimeError("unknown or collision GT requires explicit new decision")
            local = np.array([c != "remote_only" for c in categories], bool)
            targets_by_frame.append((gt, (np.ones(len(gt), bool), local, ~local)))
            labels.extend(local.tolist()); frame_sequences.append(sequence); offsets.append(offsets[-1] + len(gt))
        if len(frames) != contract["expected_frames"][role]: raise RuntimeError("frame coverage")
        coverage[role] = {"frames": len(frames), "actual_sequences": len(set(frame_sequences)),
            "gt": len(labels), "local_visible": sum(labels), "remote_exclusive": len(labels)-sum(labels),
            "collision_gt": 0, "GT_drift": 0}
        arrays[role + "__gt_local_visible"] = np.asarray(labels, np.int8)
        arrays[role + "__gt_offsets"] = np.asarray(offsets, np.int64)
        for method in contract["methods"]:
            scores, output_sequences, candidate_offsets = [], [], [0]
            parts = defaultdict(list)
            counts = defaultdict(int)
            for ordinal, ((sequence, timestamp), rows) in enumerate(sorted(frames.items())):
                row = rows[method]; gt, masks = targets_by_frame[ordinal]
                if row["role"] != role or row["sequence"] != sequence or str(row["timestamp"]) != timestamp or not np.array_equal(np.asarray(row["gt_boxes"], np.float32).reshape(-1, 7), gt):
                    raise RuntimeError("prediction identity")
                boxes, score = np.asarray(row["boxes"], np.float32).reshape(-1, 7), np.asarray(row["scores"], np.float64)
                if len(boxes) != len(score) or not np.isfinite(boxes).all() or not np.isfinite(score).all() or np.any(np.diff(score) > 0):
                    raise RuntimeError("nonfinite/misordered prediction")
                matrix = iou32(boxes, gt)
                saved_match = greedy(matrix, np.arange(len(boxes)), .7).astype(np.int32)
                if not np.array_equal(saved_match, row["matched_gt70"]): raise RuntimeError("full GT saved TP drift")
                scores.extend(score.tolist()); output_sequences.extend([sequence] * len(score)); candidate_offsets.append(len(scores))
                for group, target in zip(GROUPS, masks):
                    counts[group] += int(target.sum()); counts[group, sequence] += int(target.sum())
                    for threshold in THRESHOLDS:
                        for kind, values in zip(("tp", "fp", "ignore"), outcomes(matrix, target, threshold)):
                            parts[group, threshold, kind].append(values)
            score = np.asarray(scores, np.float64); sequences = np.asarray(output_sequences)
            prefix = role + "__" + method
            arrays[prefix + "__offsets"] = np.asarray(candidate_offsets, np.int64)
            for key in list(parts):
                parts[key] = np.concatenate(parts[key])
                group, threshold, kind = key
                arrays[prefix + "__" + group + "__" + str(int(threshold*100)) + "__" + kind] = parts[key]
            for group in GROUPS:
                def panel(sequence=None):
                    use = np.ones(len(score), bool) if sequence is None else sequences == sequence
                    n_gt = counts[group] if sequence is None else counts[group, sequence]
                    result = {"role": role, "protocol": "strict_grid400", "wait_ms": 400, "method": method, "source_group": group, "gt": n_gt}
                    if sequence is not None: result["sequence"] = sequence
                    for threshold in THRESHOLDS:
                        tag = str(int(threshold*100))
                        tp, fp, ignored = (parts[group, threshold, kind][use] for kind in ("tp", "fp", "ignore"))
                        values = metric(tp, fp, score[use], n_gt)
                        result.update({"ap"+tag: values["ap"], "tp"+tag: values["tp"], "fp"+tag: values["fp"], "ignore"+tag: int(ignored.sum())})
                        if not np.all(tp+fp+ignored == 1): raise RuntimeError("outcome partition")
                    return result
                aggregate = panel(); panels.append(aggregate)
                sequence_panels.extend(panel(sequence) for sequence in sorted(set(frame_sequences)))
                if group == "all":
                    ref = references[method]
                    for threshold in (50, 70):
                        if abs(aggregate["ap"+str(threshold)] - 100*ref["ap"+str(threshold)]) > 1e-9 or any(aggregate[k+str(threshold)] != ref[k+str(threshold)] for k in ("tp", "fp")) or aggregate["gt"] != ref["gt"]:
                            raise RuntimeError((role, method, "accepted full GT AP/count replay"))
            atomic_json(output / "progress.json", {"role": role, "last_completed_method": method, "status": "running"})
    save_npz(output / "outcome_deltas.npz", **arrays)
    result = {"status": "produced_pending_independent_review", "contract": evidence(contract_path),
        "formal_panels": panels, "per_sequence_panels": sequence_panels, "paired_sequence_bootstrap": bootstrap(sequence_panels, contract["methods"]),
        "coverage": coverage, "outcomes": evidence(output / "outcome_deltas.npz"),
        "checks": {"all_full_GT_AP50_AP70_TP_FP_replayed": True, "exact_GT_lineage": True, "all_saved_matches_replayed": True, "all_outcomes_partition": True},
        "actual_cost": {"wall_seconds": time.monotonic()-started, "cpu_process_seconds": time.process_time()-cpu,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0, "gpu_seconds": 0}}
    atomic_json(output / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); compute(args.contract, args.output)
