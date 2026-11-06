#!/usr/bin/env python3
"""Paired 21-sequence cluster bootstrap for current V2X-ViT geometry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np


WAITS = (0, 100, 200, 400)
SEEDS = (2023, 2024, 2025)
BOOTSTRAP_SEED = 20261025
REPLICATES = 10000
EXPECTED_GT = 31123


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def voc_ap(tp: np.ndarray, score: np.ndarray, gt: int) -> float:
    order = np.argsort(-score.astype(np.float64, copy=False))
    ordered_tp = tp[order].astype(np.int64, copy=False)
    ctp = np.cumsum(ordered_tp)
    cfp = np.cumsum(1 - ordered_tp)
    recall = ctp.astype(np.float64) / max(gt, 1)
    precision = ctp.astype(np.float64) / np.maximum(ctp + cfp, 1)
    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.maximum.accumulate(np.r_[0.0, precision, 0.0][::-1])[::-1]
    changed = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum(np.diff(mrec)[changed] * mpre[changed + 1]))


def weighted_ap_batch(score: np.ndarray, tp: np.ndarray, cluster: np.ndarray,
                      gt_by_cluster: np.ndarray, counts: np.ndarray,
                      batch_size: int = 64) -> np.ndarray:
    """Weighted empirical-cluster AP, preserving accepted within-score order."""
    order = np.argsort(-score.astype(np.float64, copy=False))
    sorted_tp = tp[order].astype(np.int64, copy=False)
    sorted_cluster = cluster[order].astype(np.int64, copy=False)
    result = np.empty(len(counts), np.float64)
    for first in range(0, len(counts), batch_size):
        current = counts[first:first + batch_size].astype(np.int64, copy=False)
        weight = current[:, sorted_cluster]
        ctp = np.cumsum(weight * sorted_tp[None, :], axis=1)
        cfp = np.cumsum(weight * (1 - sorted_tp)[None, :], axis=1)
        gt = current @ gt_by_cluster.astype(np.int64, copy=False)
        recall = ctp / np.maximum(gt[:, None], 1)
        precision = ctp / np.maximum(ctp + cfp, 1)
        padded_precision = np.concatenate((
            np.zeros((len(current), 1)), precision,
            np.zeros((len(current), 1))), axis=1)
        envelope = np.maximum.accumulate(padded_precision[:, ::-1], axis=1)[:, ::-1]
        previous = np.concatenate((np.zeros((len(current), 1)), recall), axis=1)
        following = np.concatenate((recall, np.ones((len(current), 1))), axis=1)
        delta = following - previous
        # Only detection recall increments contribute; the terminal pad mirrors voc_ap.
        result[first:first + len(current)] = np.sum(
            delta[:, :-1] * envelope[:, 1:-1], axis=1)
    return result


def candidate_arrays(ledger_path: Path, diagnostics_path: Path,
                     predictions_path: Path, wait: int, detector_seed: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    suffix = f"__val21__{wait}__spatial_independent3_u__point__score"
    identity_prefix = ("vit_seed20261023__" if detector_seed == 2023
                       else f"vit_seed{detector_seed}__")
    with np.load(ledger_path, allow_pickle=False) as ledger:
        keys = [key for key in ledger.files
                if key.startswith(identity_prefix) and key.endswith(suffix)]
        assert len(keys) == 1, (ledger_path, wait, keys)
        score_key = keys[0]
        prefix = score_key[:-len("__score")]
        score = ledger[score_key].astype(np.float32)
        tp = ledger[prefix + "__tp70"].astype(np.int8)
        assert int(ledger[prefix + "__gt"][0]) == EXPECTED_GT
    all_rows = [json.loads(line) for line in diagnostics_path.read_text().splitlines() if line]
    method_rows = [row for row in all_rows
                   if row["method"] == "spatial_independent3_u"]
    assert len(method_rows) == len(WAITS) * 3316
    condition_indices = [int(row["condition_index"]) for row in method_rows]
    assert sorted(condition_indices) == list(range(len(method_rows)))
    assert len({(int(row["condition_index"]), int(row["global_index"]),
                 str(row["sequence"]), int(row["wait_ms"]), str(row["method"]))
                for row in method_rows}) == len(method_rows)
    with np.load(predictions_path, allow_pickle=False) as predictions:
        offsets = predictions["spatial_independent3_u__offset"].astype(np.int64)
    assert len(offsets) == len(method_rows) + 1
    assert int(offsets[0]) == 0 and np.all(np.diff(offsets) >= 0)
    assert int(offsets[-1]) == int(np.diff(offsets).sum())
    rows = [row for row in method_rows if int(row["wait_ms"]) == wait]
    rows.sort(key=lambda row: int(row["global_index"]))
    assert len(rows) == 3316
    assert [int(row["global_index"]) for row in rows] == list(range(3316))
    assert len({int(row["condition_index"]) for row in rows}) == 3316
    sequences = sorted({str(row["sequence"]) for row in rows})
    assert len(sequences) == 21
    index = {name: value for value, name in enumerate(sequences)}
    clusters = []
    offset = 0
    diagnostics_tp = 0
    rank_tp = 0
    tp_mismatch_frames = 0
    for row in rows:
        condition_index = int(row["condition_index"])
        count = int(offsets[condition_index + 1] - offsets[condition_index])
        segment = tp[offset:offset + count]
        assert len(segment) == count
        segment_tp = int(segment.sum())
        diagnostics_tp += int(row["tp70"])
        rank_tp += segment_tp
        tp_mismatch_frames += segment_tp != int(row["tp70"])
        clusters.append(np.full(count, index[str(row["sequence"])], np.int16))
        offset += count
    assert offset == len(score) == len(tp)
    return score, tp, np.concatenate(clusters), {
        "offset_detections": offset,
        "rank_tp": rank_tp,
        "diagnostics_tp": diagnostics_tp,
        "tp_mismatch_frames": tp_mismatch_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--identity-resolution", required=True, type=Path)
    for seed in SEEDS:
        parser.add_argument(f"--native-{seed}", required=True, type=Path)
        parser.add_argument(f"--candidate-ledger-{seed}", required=True, type=Path)
        parser.add_argument(f"--candidate-diagnostics-{seed}", required=True, type=Path)
        parser.add_argument(f"--candidate-predictions-{seed}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert "embodied-research.slice" in Path("/proc/self/cgroup").read_text()
    root = args.workspace.resolve()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    assert contract["owner"] == "research_autopilot_v2xvit_current_geometry_sequence_bootstrap_v1"
    identity_path = args.identity_resolution.resolve()
    identity = json.loads(identity_path.read_text())
    assert identity["status"] == "resolved_to_accepted_common_roi_ledgers"
    assert identity["task_id"] == contract["task_id"]
    assert identity["owner"] == contract["owner"]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, 21, size=(REPLICATES, 21), endpoint=False)
    counts = np.zeros((REPLICATES, 21), np.int16)
    np.add.at(counts, (np.arange(REPLICATES)[:, None], draws), 1)
    np.testing.assert_array_equal(counts.sum(axis=1), 21)
    bootstrap_arrays: dict[str, np.ndarray] = {"cluster_counts": counts}
    panel_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    source_sha = {
        str(contract_path.relative_to(root)): sha256(contract_path),
        str(identity_path.relative_to(root)): sha256(identity_path),
    }
    common_sequences: list[str] | None = None
    common_gt: np.ndarray | None = None

    for seed in SEEDS:
        native_dir = getattr(args, f"native_{seed}").resolve()
        native_summary_path = native_dir / "summary.json"
        native_summary = json.loads(native_summary_path.read_text())
        assert native_summary["detector_seed"] == seed
        assert native_summary["status"] == "complete_review_pending"
        identity_seed = identity["native_inputs"][str(seed)]
        assert sha256(native_summary_path) == identity_seed["compact_summary_sha256"]
        assert native_summary["checkpoint_sha256"] == identity_seed["checkpoint_sha256"]
        native_archive_path = native_dir / "native_sequence_arrays_v2.npz"
        assert sha256(native_archive_path) == native_summary["output_sha256"]
        ledger_path = getattr(args, f"candidate_ledger_{seed}").resolve()
        diagnostics_path = getattr(args, f"candidate_diagnostics_{seed}").resolve()
        predictions_path = getattr(args, f"candidate_predictions_{seed}").resolve()
        source_sha[str(native_summary_path.relative_to(root))] = sha256(native_summary_path)
        source_sha[str(native_archive_path.relative_to(root))] = sha256(native_archive_path)
        source_sha[str(ledger_path.relative_to(root))] = sha256(ledger_path)
        source_sha[str(diagnostics_path.relative_to(root))] = sha256(diagnostics_path)
        source_sha[str(predictions_path.relative_to(root))] = sha256(predictions_path)
        with np.load(native_archive_path, allow_pickle=False) as native:
            sequences = native["sequences"].astype(str).tolist()
            if common_sequences is None:
                common_sequences = sequences
            assert sequences == common_sequences
            for wait in WAITS:
                prefix = f"wait{wait:03d}"
                native_score = native[prefix + "__score"].astype(np.float32)
                native_tp = native[prefix + "__tp70"].astype(np.int8)
                native_cluster = native[prefix + "__cluster"].astype(np.int16)
                gt_by_cluster = native[prefix + "__gt_by_cluster"].astype(np.int64)
                if common_gt is None:
                    common_gt = gt_by_cluster.copy()
                np.testing.assert_array_equal(gt_by_cluster, common_gt)
                (candidate_score, candidate_tp, candidate_cluster,
                 candidate_segmentation) = candidate_arrays(
                    ledger_path, diagnostics_path, predictions_path, wait, seed)
                native_observed = voc_ap(native_tp, native_score, EXPECTED_GT)
                candidate_observed = voc_ap(candidate_tp, candidate_score, EXPECTED_GT)
                native_boot = weighted_ap_batch(
                    native_score, native_tp, native_cluster, gt_by_cluster, counts)
                candidate_boot = weighted_ap_batch(
                    candidate_score, candidate_tp, candidate_cluster, gt_by_cluster, counts)
                delta = 100.0 * (candidate_boot - native_boot)
                key = f"seed{seed}_wait{wait:03d}_delta_pp"
                bootstrap_arrays[key] = delta
                low, high = np.quantile(delta, (0.025, 0.975))
                per_sequence_delta = []
                for cid, sequence in enumerate(sequences):
                    candidate_select = candidate_cluster == cid
                    native_select = native_cluster == cid
                    candidate_ap = voc_ap(candidate_tp[candidate_select],
                                          candidate_score[candidate_select],
                                          int(gt_by_cluster[cid]))
                    native_ap = voc_ap(native_tp[native_select], native_score[native_select],
                                       int(gt_by_cluster[cid]))
                    value = 100.0 * (candidate_ap - native_ap)
                    per_sequence_delta.append(value)
                    sequence_rows.append({
                        "detector_seed": seed, "wait_ms": wait,
                        "sequence": sequence, "gt": int(gt_by_cluster[cid]),
                        "candidate_ap70_percent": 100.0 * candidate_ap,
                        "native_ap70_percent": 100.0 * native_ap,
                        "candidate_minus_native_pp": value,
                        "candidate_detections": int(candidate_select.sum()),
                        "native_detections": int(native_select.sum()),
                    })
                observed_delta = 100.0 * (candidate_observed - native_observed)
                panel_rows.append({
                    "detector_seed": seed,
                    "wait_ms": wait,
                    "candidate_ap70_percent": 100.0 * candidate_observed,
                    "native_ap70_percent": 100.0 * native_observed,
                    "observed_delta_pp": observed_delta,
                    "bootstrap_mean_delta_pp": float(delta.mean()),
                    "bootstrap_median_delta_pp": float(np.median(delta)),
                    "ci95_low_pp": float(low),
                    "ci95_high_pp": float(high),
                    "probability_delta_gt_zero": float(np.mean(delta > 0)),
                    "bootstrap_positive_zero_negative": [
                        int((delta > 0).sum()), int((delta == 0).sum()),
                        int((delta < 0).sum())],
                    "sequence_positive_zero_negative": [
                        int(sum(value > 0 for value in per_sequence_delta)),
                        int(sum(value == 0 for value in per_sequence_delta)),
                        int(sum(value < 0 for value in per_sequence_delta))],
                    "sequence_delta_min_pp": float(min(per_sequence_delta)),
                    "sequence_delta_max_pp": float(max(per_sequence_delta)),
                    "candidate_detections": len(candidate_score),
                    "native_detections": len(native_score),
                    "gt": EXPECTED_GT,
                    "candidate_rank_tp": candidate_segmentation["rank_tp"],
                    "candidate_diagnostics_tp": candidate_segmentation["diagnostics_tp"],
                    "candidate_diagnostics_rank_tp_mismatch_frames":
                        candidate_segmentation["tp_mismatch_frames"],
                })

    assert common_sequences is not None and common_gt is not None
    archive = output / "bootstrap_deltas_v1.npz"
    np.savez_compressed(archive, **bootstrap_arrays)
    write_csv(output / "panels_v1.csv", panel_rows)
    write_csv(output / "per_sequence_v1.csv", sequence_rows)
    positive_panels = sum(row["observed_delta_pp"] > 0 for row in panel_rows)
    ci_positive = sum(row["ci95_low_pp"] > 0 for row in panel_rows)
    result = {
        "schema": "v2xvit_current_geometry_sequence_cluster_bootstrap_v1",
        "status": "complete_review_pending",
        "task_id": contract["task_id"],
        "owner": contract["owner"],
        "method_identity": contract["method_identity"],
        "bootstrap": {
            "unit": "validation sequence",
            "clusters": 21,
            "replicates": REPLICATES,
            "random_seed": BOOTSTRAP_SEED,
            "paired": True,
            "algorithm": (
                "Multinomial resampling of 21 sequences with replacement; identical cluster "
                "multiplicities for candidate/native; weighted global-score VOC AP preserves "
                "each accepted ledger's original within-score order."
            ),
        },
        "candidate_cluster_identity": {
            "length_source": "spatial_independent3_u__offset in saved predictions.npz",
            "frame_identity_source": "condition_index/global_index/sequence/wait/method in frame_diagnostics.jsonl",
            "tp_and_score_source": "accepted ranking_recompute_ledger_v1.npz",
            "diagnostics_tp_role": (
                "Recorded only as an old-evaluator versus ranking-recompute matching-path "
                "difference; it is not used to segment or alter the accepted rank ledger."
            ),
        },
        "panels": panel_rows,
        "sequence_rows": len(sequence_rows),
        "summary": {
            "observed_positive_panels": positive_panels,
            "observed_negative_panels": len(panel_rows) - positive_panels,
            "ci95_strictly_positive_panels": ci_positive,
            "ci95_includes_or_below_zero_panels": len(panel_rows) - ci_positive,
            "prediction_supported": positive_panels >= len(panel_rows) // 2,
        },
        "source_sha256": source_sha,
        "output_sha256": {
            "bootstrap_deltas_v1.npz": sha256(archive),
            "panels_v1.csv": sha256(output / "panels_v1.csv"),
            "per_sequence_v1.csv": sha256(output / "per_sequence_v1.csv"),
        },
        "actual_cost": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "bootstrap_replicates_per_panel": REPLICATES,
            "panels": len(panel_rows),
            "gpu_seconds": 0,
            "new_detector_or_backbone_forwards": 0,
            "optimizer_updates": 0,
        },
        "boundary": (
            "Descriptive paired cluster bootstrap on exposed val21. It quantifies sequence "
            "concentration for the current saved point-spatial full-refit system versus the "
            "same-detector native feature-fusion system. It is not a hidden-test, new split, "
            "causal module, communication, or online-latency result; CI sign is not an effect gate."
        ),
    }
    atomic_json(output / "summary.json", result)
    print(json.dumps({"status": result["status"], **result["summary"],
                      "wall_seconds": result["actual_cost"]["wall_seconds"]}))


if __name__ == "__main__":
    main()
