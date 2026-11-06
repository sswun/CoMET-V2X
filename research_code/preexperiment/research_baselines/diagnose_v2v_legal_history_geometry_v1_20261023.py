#!/usr/bin/env python3
"""Build and diagnose legal V2V4Real stale/constant-velocity geometry.

This is the zero-forward first stage of the released 2026-10-23 V2 history
transfer.  It reuses the reviewed single-sensor source outputs, the reviewed
native fit400 cache, and canonical native eval predictions.  No detector,
backbone, or trainable head is called.

The script deliberately separates three questions:

* input legality: the previous source output belongs to the same sequence and
  strictly precedes the current source output;
* geometry support: stale versus causal constant-velocity hypotheses on
  native-uncovered GT occurrences;
* ordering: full native-prefix AP and post-hoc fixed-FP diagnostics for simple
  raw-score controls.

GT is used only for offline labels/evaluation and never for association,
propagation, candidate selection, or submitted scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from e1_receiver_value_core_v1_20261020 import error_counts, greedy, iou32, metrics
from v2xseq_real_motion_packets import transform_boxes7d


TASK_ID = "v2v_legal_history_geometry_transfer_20261023"
OWNER = "embodied-v2v-history-geometry-diagnostic-v1-20261023.service"
BASE = Path("artifacts/research_autopilot/status_audit_20261020_ultra/v2v4real_head_confirmation_v1")
SENSOR = Path("artifacts/research_autopilot/status_audit_20261020_ultra/v2v4real_single_sensor_cache_v1")
INVENTORY = Path("artifacts/research_autopilot/status_audit_20261022_ultra_evening/v2v_legal_history_native_complement_inventory_v5")
NATIVE_FIT = Path("artifacts/research_autopilot/status_audit_20261022_ultra_evening/v2v_native_fit400_cache_v3")
NATIVE_FIT_REVIEW = Path("artifacts/research_autopilot/status_audit_20261022_ultra_evening/v2v_native_fit400_cache_review_v3/summary.json")
NATIVE_EVAL = Path("artifacts/runs/traf_align_cost_canonical_gt_rescore_v1_20261003/result/predictions_canonical_gt.jsonl")
UNION = Path("artifacts/research_autopilot/status_audit_20261022_ultra_evening/v2v_native_source_union_controls_v4")
DIRECT_QU = Path("artifacts/research_autopilot/status_audit_20261022_ultra_evening/v2v_native_source_direct_qu_v2")

MANIFESTS = (
    SENSOR / "qualification_v3/scientific_manifest.jsonl",
    SENSOR / "production_gpu0_v1/scientific_manifest.jsonl",
    SENSOR / "production_gpu1_v1/scientific_manifest.jsonl",
    SENSOR / "recovery_gpu0_v2/scientific_manifest.jsonl",
    SENSOR / "recovery_gpu1_v2/scientific_manifest.jsonl",
)

METHODS = (
    "native",
    "native_prefix_stale_raw",
    "native_prefix_cv_replace_raw",
    "native_prefix_stale_plus_cv_raw",
)
FP_BUDGETS = (2048, 4096, 6144, 8169, 10240, 16384)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def append(stream: Any, value: Any) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False) + "\n")


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def local_artifact(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return root / path
    parts = path.parts
    if "artifacts" not in parts:
        raise ValueError(f"absolute non-artifact path is not portable: {path}")
    return root / Path(*parts[parts.index("artifacts"):])


def common_mask(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes).reshape(-1, 7)
    return ((boxes[:, 0] >= -100.0) & (boxes[:, 0] <= 100.0)
            & (boxes[:, 1] >= -40.0) & (boxes[:, 1] <= 40.0))


def stable_unique(scores: np.ndarray) -> np.ndarray:
    """Give exact ties a deterministic descending order without moving non-ties."""
    result = np.asarray(scores, np.float64).copy()
    order = np.argsort(-result, kind="stable")
    for index in range(1, len(order)):
        previous, current = int(order[index - 1]), int(order[index])
        if result[current] >= result[previous]:
            result[current] = np.nextafter(result[previous], -np.inf)
    return result


class PackedSource:
    def __init__(self, directory: Path):
        self.meta = list(rows(directory / "conditions.jsonl"))
        self.data = load_npz(directory / "pools.npz")

    def pool(self, condition_index: int) -> dict[str, np.ndarray]:
        meta = self.meta[condition_index]
        start, stop = int(meta["start"]), int(meta["stop"])
        gt_start, gt_stop = int(meta["gt_start"]), int(meta["gt_stop"])
        result = {name: self.data[name][start:stop] for name in (
            "boxes", "raw_scores", "is_remote", "source_detection_indices",
            "quality_y", "utility_y",
        )}
        result["gt_boxes"] = self.data["gt_boxes"][gt_start:gt_stop]
        result["ious"] = self.data["ious"][int(meta["iou_start"]):int(meta["iou_stop"])].reshape(
            stop - start, gt_stop - gt_start)
        return result


def sensor_manifests(root: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for relative in MANIFESTS:
        for row in rows(root / relative):
            task = int(row["task_index"])
            if task in result:
                assert result[task]["task_identity_sha256"] == row["task_identity_sha256"]
            else:
                result[task] = row
    assert len(result) == 7281
    assert sum(row["sensor_role"] == "remote" for row in result.values()) == 4155
    return result


def native_eval_map(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    result = {}
    for row in rows(path):
        if bool(row["ego_only"]):
            continue
        wait = int(row["delay_ms"])
        assert row["condition"] == f"collaborative_{wait}ms"
        key = (row["scene_name"], int(row["global_index"]), wait)
        assert key not in result
        result[key] = row
    assert len(result) == 2058
    return result


def native_fit_map(root: Path) -> dict[int, dict[str, Any]]:
    completion = json.loads((root / NATIVE_FIT / "completion_state_v1.json").read_text())
    review = json.loads((root / NATIVE_FIT_REVIEW).read_text())
    assert completion["status"] == review["status"] == "complete_independent_review_passed"
    result = {}
    for row in rows(root / NATIVE_FIT / "scientific_manifest.jsonl"):
        condition = int(row["condition_index"])
        assert condition not in result and int(row["wait_ms"]) == 400
        result[condition] = row
    assert len(result) == 3126
    return result


def associate(current: dict[str, np.ndarray], previous: dict[str, np.ndarray] | None,
              current_pose: np.ndarray, previous_pose: np.ndarray | None,
              dt_s: float, horizon_s: float) -> dict[str, np.ndarray]:
    boxes = current["pred_boxes"].astype(np.float32).copy()
    count = len(boxes)
    confirmed = np.zeros(count, bool)
    previous_index = np.full(count, -1, np.int32)
    distance = np.zeros(count, np.float64)
    ratio = np.zeros(count, np.float64)
    velocity = np.zeros((count, 2), np.float64)
    shift = np.zeros((count, 2), np.float64)
    if previous is None or previous_pose is None or horizon_s <= 0.0 or not count:
        return dict(boxes=boxes, confirmed=confirmed, previous_index=previous_index,
                    distance=distance, ratio=ratio, velocity=velocity, shift=shift)
    assert 0.0 < dt_s <= 0.5
    previous_boxes = transform_boxes7d(previous["pred_boxes"], previous_pose, current_pose)
    if not len(previous_boxes):
        return dict(boxes=boxes, confirmed=confirmed, previous_index=previous_index,
                    distance=distance, ratio=ratio, velocity=velocity, shift=shift)
    distances = np.linalg.norm(boxes[:, None, :2] - previous_boxes[None, :, :2], axis=2)
    current_ids, previous_ids = linear_sum_assignment(distances)
    gate = 1.0 + 50.0 * dt_s
    for current_id, previous_id in zip(current_ids, previous_ids):
        value = float(distances[current_id, previous_id])
        if value > gate:
            continue
        v = (boxes[current_id, :2] - previous_boxes[previous_id, :2]) / dt_s
        speed = float(np.linalg.norm(v))
        if speed > 50.0:
            v *= 50.0 / speed
        displacement = v * horizon_s
        boxes[current_id, :2] += displacement.astype(np.float32)
        confirmed[current_id] = True
        previous_index[current_id] = int(previous_id)
        distance[current_id] = value
        ratio[current_id] = value / gate
        velocity[current_id] = v
        shift[current_id] = displacement
    return dict(boxes=boxes, confirmed=confirmed, previous_index=previous_index,
                distance=distance, ratio=ratio, velocity=velocity, shift=shift)


def condition_geometry(root: Path, metadata: dict[str, Any], source_pool: dict[str, np.ndarray],
                       manifests: dict[int, dict[str, Any]],
                       previous_map: dict[int, dict[str, Any]]) -> dict[str, Any]:
    task = int(metadata["source_task_index"])
    manifest = manifests[task]
    assert manifest["sensor_role"] == "remote"
    assert manifest["scene_name"] == metadata["sequence"]
    assert manifest["timestamp"] == metadata["source_timestamp"]
    current_path = local_artifact(root, manifest["raw_path"])
    assert sha256(current_path) == manifest["raw_sha256"]
    current = load_npz(current_path)
    source_pose = np.asarray(metadata["source_pose"], np.float64)
    receiver_pose = np.asarray(metadata["receiver_ego_pose"], np.float64)
    assert np.allclose(source_pose, np.asarray(manifest["source_pose_lidar_to_world"], np.float64),
                       atol=1e-8, rtol=0.0)
    prior_identity = previous_map[task]
    previous = previous_manifest = None
    previous_pose = None
    dt_s = 0.0
    if prior_identity["previous_task_index"] is not None:
        previous_manifest = manifests[int(prior_identity["previous_task_index"])]
        assert previous_manifest["scene_name"] == manifest["scene_name"]
        assert previous_manifest["timestamp"] == prior_identity["previous_timestamp"]
        assert previous_manifest["raw_sha256"] == prior_identity["previous_raw_sha256"]
        assert int(previous_manifest["timestamp"]) < int(manifest["timestamp"])
        assert int(manifest["timestamp"]) - int(previous_manifest["timestamp"]) == 1
        previous_path = local_artifact(root, previous_manifest["raw_path"])
        assert sha256(previous_path) == previous_manifest["raw_sha256"]
        previous = load_npz(previous_path)
        previous_pose = np.asarray(previous_manifest["source_pose_lidar_to_world"], np.float64)
        dt_s = 0.1
    horizon_s = float(metadata["effective_age_ms"]) / 1000.0
    motion = associate(current, previous, source_pose, previous_pose, dt_s, horizon_s)
    transformed_stale = transform_boxes7d(current["pred_boxes"], source_pose, receiver_pose)
    transformed_prop = transform_boxes7d(motion["boxes"], source_pose, receiver_pose)
    remote_ids = np.flatnonzero(source_pool["is_remote"].astype(bool))
    source_ids = source_pool["source_detection_indices"][remote_ids].astype(np.int32)
    assert len(np.unique(source_ids)) == len(source_ids)
    stale = source_pool["boxes"][remote_ids].astype(np.float32)
    reconstructed = transformed_stale[source_ids]
    difference = np.max(np.abs(stale - reconstructed), axis=1, initial=0.0)
    if len(difference):
        assert float(difference.max()) <= 2e-5
    scores = source_pool["raw_scores"][remote_ids].astype(np.float64)
    assert np.max(np.abs(scores - current["pred_scores"][source_ids]), initial=0.0) <= 1e-7
    prop_valid = motion["confirmed"][source_ids] & common_mask(transformed_prop[source_ids])
    prop = transformed_prop[source_ids].astype(np.float32)
    gt = source_pool["gt_boxes"].astype(np.float32)
    stale_iou = iou32(stale, gt)
    prop_iou = iou32(prop, gt)
    if len(prop_iou):
        prop_iou[~prop_valid] = 0.0
    association_features = np.column_stack((
        scores,
        current["pred_boxes"][source_ids].astype(np.float64),
        np.full(len(source_ids), horizon_s),
        np.full(len(source_ids), dt_s),
        motion["velocity"][source_ids],
        motion["distance"][source_ids],
        motion["ratio"][source_ids],
        motion["shift"][source_ids],
        prop_valid.astype(np.float64),
    )).astype(np.float32)
    return {
        "stale_boxes": stale, "prop_boxes": prop, "scores": scores,
        "source_detection_indices": source_ids, "prop_valid": prop_valid,
        "stale_ious": stale_iou, "prop_ious": prop_iou,
        "gt_boxes": gt, "association_features": association_features,
        "current_raw_path": str(current_path.relative_to(root)),
        "current_raw_sha256": manifest["raw_sha256"],
        "previous_raw_path": None if previous_manifest is None else str(local_artifact(root, previous_manifest["raw_path"]).relative_to(root)),
        "previous_raw_sha256": None if previous_manifest is None else previous_manifest["raw_sha256"],
        "previous_task_index": prior_identity["previous_task_index"],
        "previous_timestamp": prior_identity["previous_timestamp"],
        "history_dt_s": dt_s, "horizon_s": horizon_s,
        "reconstruction_max_abs_error": float(difference.max(initial=0.0)),
    }


class GeometryWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.stream = (directory / "conditions.jsonl").open("w", encoding="utf-8")
        self.arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        self.cursor = self.gt_cursor = self.iou_cursor = 0
        self.conditions = 0

    def add(self, metadata: dict[str, Any], value: dict[str, Any]) -> None:
        count, gt_count = len(value["stale_boxes"]), len(value["gt_boxes"])
        record = {
            "sequence": metadata["sequence"], "global_index": int(metadata["global_index"]),
            "timestamp": metadata["timestamp"], "source_timestamp": metadata["source_timestamp"],
            "source_task_index": int(metadata["source_task_index"]),
            "wait_ms": int(metadata["wait_ms"]), "effective_age_ms": int(metadata["effective_age_ms"]),
            "role": metadata["role"], "condition_index": self.conditions,
            "source_condition_index": int(metadata["condition_index"]),
            "start": self.cursor, "stop": self.cursor + count,
            "gt_start": self.gt_cursor, "gt_stop": self.gt_cursor + gt_count,
            "stale_iou_start": self.iou_cursor, "stale_iou_stop": self.iou_cursor + count * gt_count,
            "prop_iou_start": self.iou_cursor, "prop_iou_stop": self.iou_cursor + count * gt_count,
            "current_raw_path": value["current_raw_path"],
            "current_raw_sha256": value["current_raw_sha256"],
            "previous_raw_path": value["previous_raw_path"],
            "previous_raw_sha256": value["previous_raw_sha256"],
            "previous_task_index": value["previous_task_index"],
            "previous_timestamp": value["previous_timestamp"],
            "history_dt_s": value["history_dt_s"], "horizon_s": value["horizon_s"],
            "reconstruction_max_abs_error": value["reconstruction_max_abs_error"],
        }
        append(self.stream, record)
        for name in ("stale_boxes", "prop_boxes", "scores", "source_detection_indices",
                     "prop_valid", "association_features", "gt_boxes"):
            self.arrays[name].append(np.asarray(value[name]))
        self.arrays["stale_ious"].append(value["stale_ious"].reshape(-1))
        self.arrays["prop_ious"].append(value["prop_ious"].reshape(-1))
        self.cursor += count
        self.gt_cursor += gt_count
        self.iou_cursor += count * gt_count
        self.conditions += 1

    def finish(self) -> dict[str, Any]:
        self.stream.close()
        arrays = {name: np.concatenate(values, axis=0) for name, values in self.arrays.items()}
        np.savez_compressed(self.directory / "pools.npz", **arrays)
        summary = {
            "status": "complete", "schema": "v2v_legal_history_geometry_dataset_v1",
            "conditions": self.conditions, "stale_candidates": self.cursor,
            "gt_occurrences": self.gt_cursor, "iou_values_per_arm": self.iou_cursor,
            "association_feature_names": [
                "source_score", "source_x", "source_y", "source_z", "box_h", "box_w", "box_l", "source_yaw",
                "horizon_s", "history_dt_s", "velocity_x", "velocity_y", "association_distance_m",
                "association_distance_over_gate", "shift_x", "shift_y", "prop_valid",
            ],
            "conditions_sha256": sha256(self.directory / "conditions.jsonl"),
            "pools_sha256": sha256(self.directory / "pools.npz"),
        }
        atomic(self.directory / "summary.json", summary)
        return summary


def reachable(matrix: np.ndarray) -> set[int]:
    if not matrix.shape[0] or not matrix.shape[1]:
        return set()
    return set(np.flatnonzero(matrix.max(axis=0) >= 0.7).astype(int).tolist())


def native_value(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = common_mask(raw["pred_boxes"]) & (raw["pred_labels"].astype(int) == 0)
    boxes = raw["pred_boxes"][mask].astype(np.float32)
    scores = raw["pred_scores"][mask].astype(np.float64)
    gt = raw["gt_boxes"].astype(np.float32)[:, :7]
    return boxes, scores, gt


def native_covered(native_boxes: np.ndarray, native_scores: np.ndarray,
                   gt: np.ndarray) -> tuple[set[int], np.ndarray]:
    matrix = iou32(native_boxes, gt)
    order = np.argsort(-native_scores, kind="stable")
    matches = greedy(matrix, order, 0.7)
    return {int(value) for value in matches if value >= 0}, matrix


def update_support(target: dict[str, Any], stale_iou: np.ndarray, prop_iou: np.ndarray,
                   prop_valid: np.ndarray, native_set: set[int], gt: np.ndarray,
                   stale_boxes: np.ndarray, prop_boxes: np.ndarray) -> dict[str, Any]:
    stale_set, prop_set = reachable(stale_iou), reachable(prop_iou)
    dual = stale_set | prop_set
    universe = set(range(len(gt)))
    native_uncovered = universe - native_set
    stale_new, prop_new, dual_new = stale_set & native_uncovered, prop_set & native_uncovered, dual & native_uncovered
    unresolved = native_uncovered - dual_new
    candidate_boxes = np.concatenate((stale_boxes, prop_boxes[prop_valid]), axis=0)
    candidate_iou = np.concatenate((stale_iou, prop_iou[prop_valid]), axis=0)
    near_05 = near_03 = center_le2 = center_le3 = 0
    per_gt = []
    for gt_id in sorted(unresolved):
        if len(candidate_boxes):
            values = candidate_iou[:, gt_id]
            best = int(np.argmax(values)); best_iou = float(values[best])
            center_error = float(np.linalg.norm(candidate_boxes[best, :2] - gt[gt_id, :2]))
        else:
            best = -1; best_iou = 0.0; center_error = float("inf")
        near_05 += int(best_iou >= 0.5); near_03 += int(best_iou >= 0.3)
        center_le2 += int(center_error <= 2.0); center_le3 += int(center_error <= 3.0)
        per_gt.append({"gt_index": gt_id, "best_candidate_index": best,
                       "best_iou70_support": best_iou,
                       "best_center_error_m": None if not np.isfinite(center_error) else center_error})
    for key, value in {
        "conditions": 1, "gt": len(gt), "native_covered_gt": len(native_set),
        "native_uncovered_gt": len(native_uncovered), "stale_reachable_gt": len(stale_set),
        "prop_reachable_gt": len(prop_set), "dual_reachable_gt": len(dual),
        "stale_native_uncovered_gt": len(stale_new), "prop_native_uncovered_gt": len(prop_new),
        "dual_native_uncovered_gt": len(dual_new),
        "prop_added_over_stale_native_uncovered_gt": len(prop_new - stale_new),
        "stale_lost_by_prop_native_uncovered_gt": len(stale_new - prop_new),
        "native_uncovered_unresolved_by_dual": len(unresolved),
        "unresolved_best_iou_ge_05": near_05, "unresolved_best_iou_ge_03": near_03,
        "unresolved_center_error_le_2m": center_le2, "unresolved_center_error_le_3m": center_le3,
        "stale_candidates": len(stale_boxes), "prop_candidates": int(prop_valid.sum()),
    }.items():
        target[key] += int(value)
    return {"stale_reachable_gt": len(stale_set), "prop_reachable_gt": len(prop_set),
            "dual_reachable_gt": len(dual), "native_covered_gt": len(native_set),
            "native_uncovered_gt": len(native_uncovered), "stale_native_uncovered_gt": len(stale_new),
            "prop_native_uncovered_gt": len(prop_new), "dual_native_uncovered_gt": len(dual_new),
            "prop_added_over_stale_native_uncovered_gt": len(prop_new - stale_new),
            "native_uncovered_unresolved_by_dual": len(unresolved),
            "unresolved": per_gt}


def fixed_fp(scores: np.ndarray, tp: np.ndarray, budget: int) -> dict[str, int]:
    order = np.argsort(-scores, kind="stable")
    false = true = submitted = 0
    for index in order:
        value = int(tp[index])
        if false + (1 - value) > budget:
            break
        false += 1 - value; true += value; submitted += 1
    return {"tp": true, "fp": false, "submitted": submitted}


def evaluation_orders(native_boxes: np.ndarray, native_scores: np.ndarray,
                      geometry: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    native_order = np.argsort(-native_scores, kind="stable")
    native_rank = 4.0 + stable_unique(native_scores[native_order])
    stale_boxes = geometry["stale_boxes"]
    prop_boxes = geometry["prop_boxes"]
    source_scores = geometry["scores"]
    valid = geometry["prop_valid"]
    tail_order = np.argsort(-source_scores, kind="stable")
    stale_rank = stable_unique(source_scores[tail_order])
    replaced = stale_boxes.copy()
    replaced[valid] = prop_boxes[valid]
    dual_boxes = []
    dual_scores = []
    dual_type = []
    for source_id in tail_order:
        dual_boxes.append(stale_boxes[source_id]); dual_scores.append(source_scores[source_id]); dual_type.append(1)
        if bool(valid[source_id]):
            dual_boxes.append(prop_boxes[source_id]); dual_scores.append(source_scores[source_id]); dual_type.append(2)
    if dual_boxes:
        dual_boxes_array = np.asarray(dual_boxes, np.float32)
        dual_scores_array = stable_unique(np.asarray(dual_scores, np.float64))
        dual_types_array = np.asarray(dual_type, np.int8)
    else:
        dual_boxes_array = np.empty((0, 7), np.float32)
        dual_scores_array = np.empty(0, np.float64)
        dual_types_array = np.empty(0, np.int8)
    prefix_boxes = native_boxes[native_order]
    prefix_types = np.zeros(len(prefix_boxes), np.int8)
    return {
        "native": (prefix_boxes, native_rank, prefix_types),
        "native_prefix_stale_raw": (
            np.concatenate((prefix_boxes, stale_boxes[tail_order]), axis=0),
            np.r_[native_rank, stale_rank],
            np.r_[prefix_types, np.ones(len(tail_order), np.int8)],
        ),
        "native_prefix_cv_replace_raw": (
            np.concatenate((prefix_boxes, replaced[tail_order]), axis=0),
            np.r_[native_rank, stale_rank],
            np.r_[prefix_types, np.where(valid[tail_order], 2, 1).astype(np.int8)],
        ),
        "native_prefix_stale_plus_cv_raw": (
            np.concatenate((prefix_boxes, dual_boxes_array), axis=0),
            np.r_[native_rank, dual_scores_array],
            np.r_[prefix_types, dual_types_array],
        ),
    }


def evaluate_val(root: Path, dataset: PackedSource, geometries: dict[int, dict[str, Any]],
                 output: Path) -> dict[str, Any]:
    native_map = native_eval_map(root / NATIVE_EVAL)
    aggregate = defaultdict(lambda: {"scores": [], "tp50": [], "tp70": [], "gt": 0,
                                    "frames": 0, "native_candidates": 0, "stale_candidates": 0,
                                    "prop_candidates": 0, "duplicate_fp": 0,
                                    "localization_fp": 0, "background_fp": 0})
    output.mkdir(parents=True, exist_ok=False)
    records = 0
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for condition, metadata in enumerate(dataset.meta):
            assert metadata["role"] == "val21"
            key = (metadata["sequence"], int(metadata["global_index"]), int(metadata["wait_ms"]))
            row = native_map[key]
            native = {
                "pred_boxes": np.asarray(row["boxes_3d_lidar"], np.float32),
                "pred_scores": np.asarray(row["scores"], np.float32),
                "pred_labels": np.asarray(row["label_preds"], np.int32),
                "gt_boxes": np.asarray(row["gt_boxes"], np.float32)[:, :7],
            }
            native_boxes, native_scores, gt = native_value(native)
            geometry = geometries[condition]
            assert np.array_equal(gt, geometry["gt_boxes"])
            native_set, _ = native_covered(native_boxes, native_scores, gt)
            wait = int(metadata["wait_ms"])
            for method, (boxes, scores, hypothesis_type) in evaluation_orders(
                    native_boxes, native_scores, geometry).items():
                matrix = iou32(boxes, gt)
                order = np.arange(len(boxes), dtype=np.int32)
                match50 = greedy(matrix, order, 0.5)
                match70 = greedy(matrix, order, 0.7)
                errors = error_counts(matrix, order, match70)
                current = {int(value) for value in match70 if value >= 0}
                for group in ("all", metadata["sequence"]):
                    item = aggregate[(group, wait, method)]
                    item["scores"].append(scores); item["tp50"].append((match50 >= 0).astype(np.int8))
                    item["tp70"].append((match70 >= 0).astype(np.int8)); item["gt"] += len(gt)
                    item["frames"] += 1; item["native_candidates"] += int((hypothesis_type == 0).sum())
                    item["stale_candidates"] += int((hypothesis_type == 1).sum())
                    item["prop_candidates"] += int((hypothesis_type == 2).sum())
                    item["duplicate_fp"] += errors["duplicate_fp"]
                    item["localization_fp"] += errors["localization_fp"]
                    item["background_fp"] += errors["background_fp"]
                append(stream, {
                    "condition_index": condition, "sequence": metadata["sequence"],
                    "global_index": int(metadata["global_index"]), "timestamp": metadata["timestamp"],
                    "source_timestamp": metadata["source_timestamp"], "source_task_index": int(metadata["source_task_index"]),
                    "wait_ms": wait, "effective_age_ms": int(metadata["effective_age_ms"]),
                    "method": method, "boxes": boxes.astype(float).tolist(),
                    "scores": scores.astype(float).tolist(), "hypothesis_type": hypothesis_type.astype(int).tolist(),
                    "matched_gt50": match50.astype(int).tolist(), "matched_gt70": match70.astype(int).tolist(),
                    "gt_boxes": gt.astype(float).tolist(),
                    "rescued_vs_native": len(current - native_set), "destroyed_vs_native": len(native_set - current),
                    "errors": errors,
                })
                records += 1
    panels = []
    full_arrays = {}
    for (sequence, wait, method), item in sorted(aggregate.items()):
        scores = np.concatenate(item.pop("scores")); tp50 = np.concatenate(item.pop("tp50")); tp70 = np.concatenate(item.pop("tp70"))
        panel = dict(item, sequence=sequence, wait_ms=wait, method=method,
                     ap50=metrics(tp50, scores, item["gt"])["ap"],
                     ap70=metrics(tp70, scores, item["gt"])["ap"],
                     tp50=int(tp50.sum()), fp50=int((1 - tp50).sum()),
                     tp70=int(tp70.sum()), fp70=int((1 - tp70).sum()))
        panels.append(panel)
        if sequence == "all":
            full_arrays[(wait, method)] = (scores, tp70)
    fixed = {str(wait): {method: {str(budget): fixed_fp(*full_arrays[(wait, method)], budget)
                                         for budget in FP_BUDGETS} for method in METHODS}
             for wait in (0, 400)}
    summary = {
        "status": "complete", "schema": "v2v_legal_history_simple_geometry_evaluation_v1",
        "role": "val21_exposed", "methods": list(METHODS), "records": records,
        "panels": panels, "fixed_fp_posthoc_diagnostic": fixed,
        "predictions_sha256": sha256(output / "predictions.jsonl"),
        "boundary": "Raw-score controls only. Fixed-FP uses eval GT post hoc and is not a deployment threshold. CV association and selection use no GT.",
    }
    atomic(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve(); contract_path = args.contract.resolve(); output = args.output.resolve()
    assert "embodied-research.slice" in Path("/proc/self/cgroup").read_text()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic(); cpu_started = time.process_time()
    output.mkdir(parents=True)
    try:
        contract = json.loads(contract_path.read_text())
        assert contract["schema"] == "v2v_legal_history_geometry_diagnostic_contract_v1"
        assert contract["status"] == "frozen_before_execution"
        assert contract["task_id"] == TASK_ID and contract["owner"] == OWNER
        assert contract["output"] == str(output.relative_to(root))
        assert contract["producer_code_sha256"] == sha256(Path(__file__).resolve())
        for relative, digest in contract["input_sha256"].items():
            assert sha256(root / relative) == digest, relative
        snapshot = output / "code_snapshot"; snapshot.mkdir()
        for path in (Path(__file__).resolve(), contract_path,
                     root / "preexperiment/research_baselines/e1_receiver_value_core_v1_20261020.py",
                     root / "preexperiment/research_baselines/v2xseq_real_motion_packets.py"):
            shutil.copy2(path, snapshot / path.name)
        inventory = json.loads((root / INVENTORY / "inventory.json").read_text())
        assert inventory["legal_history"]["remote_outputs_with_saved_previous_output"] == 4138
        assert inventory["legal_history"]["scene_first_boundary_outputs"] == 17
        manifests = sensor_manifests(root)
        previous_map = {int(row["task_index"]): row for row in rows(root / INVENTORY / "legal_previous_source_map.jsonl")}
        assert len(previous_map) == 4155
        fit_native = native_fit_map(root)
        train_source = PackedSource(root / BASE / "adapted_train_dataset")
        val_source = PackedSource(root / BASE / "adapted_eval_dataset")
        assert len(train_source.meta) == 6252 and len(val_source.meta) == 2058
        totals: dict[tuple[str, str], Counter] = defaultdict(Counter)
        detailed_path = output / "condition_support.jsonl"
        train_writer = GeometryWriter(output / "train_history_dataset")
        val_writer = GeometryWriter(output / "val_history_dataset")
        geometries_val = {}
        max_reconstruction_error = 0.0
        legal_with_previous = legal_without_previous = 0
        with detailed_path.open("w", encoding="utf-8") as detail_stream:
            for split, dataset, writer in (("train", train_source, train_writer), ("val", val_source, val_writer)):
                native_map_eval = native_eval_map(root / NATIVE_EVAL) if split == "val" else None
                for condition, metadata in enumerate(dataset.meta):
                    pool = dataset.pool(condition)
                    geometry = condition_geometry(root, metadata, pool, manifests, previous_map)
                    writer.add(metadata, geometry)
                    max_reconstruction_error = max(max_reconstruction_error, geometry["reconstruction_max_abs_error"])
                    legal_with_previous += int(geometry["previous_task_index"] is not None)
                    legal_without_previous += int(geometry["previous_task_index"] is None)
                    if split == "val":
                        row = native_map_eval[(metadata["sequence"], int(metadata["global_index"]), int(metadata["wait_ms"]))]
                        native = {"pred_boxes": np.asarray(row["boxes_3d_lidar"], np.float32),
                                  "pred_scores": np.asarray(row["scores"], np.float32),
                                  "pred_labels": np.asarray(row["label_preds"], np.int32),
                                  "gt_boxes": np.asarray(row["gt_boxes"], np.float32)[:, :7]}
                        geometries_val[condition] = geometry
                    elif int(metadata["wait_ms"]) == 400:
                        native_row = fit_native[condition]
                        native = load_npz(local_artifact(root, native_row["raw_path"]))
                    else:
                        # Fit-side native cache is intentionally only wait400.  The wait0
                        # history construction remains complete, but native-uncovered
                        # support is not fabricated from a different receiver set.
                        native = None
                    if native is not None:
                        native_boxes, native_scores, native_gt = native_value(native)
                        assert np.array_equal(native_gt, geometry["gt_boxes"])
                        native_set, _ = native_covered(native_boxes, native_scores, native_gt)
                    else:
                        native_set = set()
                    support = update_support(totals[("all", "all")], geometry["stale_ious"], geometry["prop_ious"],
                                             geometry["prop_valid"], native_set, geometry["gt_boxes"],
                                             geometry["stale_boxes"], geometry["prop_boxes"])
                    group_keys = (("split", split), ("role", metadata["role"]),
                                  ("wait", str(int(metadata["wait_ms"]))),
                                  ("effective_age", str(int(metadata["effective_age_ms"]))))
                    for group_key in group_keys:
                        update_support(totals[group_key], geometry["stale_ious"], geometry["prop_ious"],
                                       geometry["prop_valid"], native_set, geometry["gt_boxes"],
                                       geometry["stale_boxes"], geometry["prop_boxes"])
                    append(detail_stream, {
                        "split": split, "condition_index": condition, "sequence": metadata["sequence"],
                        "global_index": int(metadata["global_index"]), "timestamp": metadata["timestamp"],
                        "source_timestamp": metadata["source_timestamp"], "source_task_index": int(metadata["source_task_index"]),
                        "wait_ms": int(metadata["wait_ms"]), "effective_age_ms": int(metadata["effective_age_ms"]),
                        "role": metadata["role"], "previous_task_index": geometry["previous_task_index"],
                        "previous_timestamp": geometry["previous_timestamp"], "history_dt_s": geometry["history_dt_s"],
                        "horizon_s": geometry["horizon_s"], "stale_candidates": len(geometry["stale_boxes"]),
                        "prop_candidates": int(geometry["prop_valid"].sum()), **support,
                    })
                    if (condition + 1) % 500 == 0:
                        atomic(output / "status.json", {"status": "building_legal_history", "split": split,
                               "conditions_completed_in_split": condition + 1, "time": time.time(),
                               "new_detector_or_backbone_forwards": 0, "head_updates": 0})
        train_summary = train_writer.finish(); val_summary = val_writer.finish()
        support_summary = {
            "status": "complete", "schema": "v2v_legal_history_geometry_support_v1",
            "groups": {f"{kind}:{name}": dict(value) for (kind, name), value in sorted(totals.items())},
            "condition_support_sha256": sha256(detailed_path),
            "legal_source_output_occurrences_with_previous": legal_with_previous,
            "legal_source_output_occurrences_without_previous": legal_without_previous,
            "unique_source_outputs_with_previous": 4138,
            "unique_scene_first_source_outputs": 17,
            "history_frame_period_s": 0.1,
            "maximum_stale_reconstruction_abs_error": max_reconstruction_error,
            "association": {"algorithm": "Hungarian center distance in current source frame",
                            "gate_m": "1 + 50*history_dt_s", "maximum_speed_mps": 50.0,
                            "future_or_GT_input": False},
            "boundary": "Reachability and residual-support counts are offline GT diagnostics, not inference inputs or AP effects.",
        }
        atomic(output / "fit_geometry_support.json", support_summary)
        evaluation = evaluate_val(root, val_source, geometries_val, output / "val_simple_controls")
        fit400 = support_summary["groups"]["role:fit24"]
        # fit24 includes wait0 and wait400.  Use the exact wait400 intersection
        # by recomputing the released decision from detailed per-condition rows.
        fit400_rows = [row for row in rows(detailed_path)
                       if row["role"] == "fit24" and int(row["wait_ms"]) == 400]
        fit400_decision = {
            "conditions": len(fit400_rows),
            "prop_adds_native_uncovered_iou70_gt": sum(row["prop_added_over_stale_native_uncovered_gt"] for row in fit400_rows),
            "dual_native_uncovered_iou70_gt": sum(row["dual_native_uncovered_gt"] for row in fit400_rows),
            "native_uncovered_unresolved_by_dual": sum(row["native_uncovered_unresolved_by_dual"] for row in fit400_rows),
        }
        fit400_decision["bounded_residual_qualification_recommended"] = (
            fit400_decision["native_uncovered_unresolved_by_dual"] > 0)
        cost = {
            "wall_seconds": time.monotonic() - started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "new_detector_or_backbone_forwards": 0, "detector_updates": 0,
            "head_updates": 0, "gpu_seconds": 0,
        }
        summary = {
            "status": "complete_review_pending",
            "schema": "v2v_legal_history_geometry_diagnostic_bundle_v1",
            "task_id": TASK_ID, "owner": OWNER,
            "train_history_dataset": train_summary, "val_history_dataset": val_summary,
            "fit400_decision": fit400_decision,
            "fit_geometry_support_sha256": sha256(output / "fit_geometry_support.json"),
            "val_simple_controls_sha256": sha256(output / "val_simple_controls/summary.json"),
            "val_predictions_sha256": evaluation["predictions_sha256"],
            "input_contract_sha256": sha256(contract_path),
            "code_sha256": sha256(Path(__file__).resolve()), "cost": cost,
            "strong_system_control": {
                "identity": "reviewed native-prefix plus source-U",
                "completion": str(DIRECT_QU / "completion_state_v1.json"),
                "completion_sha256": sha256(root / DIRECT_QU / "completion_state_v1.json"),
                "not_recomputed": True,
            },
            "boundaries": [
                "V2V4Real native 0/400ms protocol and exposed val1029 remain unchanged.",
                "The simple controls use raw scores; they do not replace the retained learned source-U system.",
                "No detector/backbone/head update or scientific forward is performed.",
                "System gain, history geometry support, ranking, and any later residual-head contribution remain separate.",
            ],
        }
        atomic(output / "summary.json", summary)
        atomic(output / "completion_state_v1.json", {
            "schema": summary["schema"], "status": summary["status"],
            "task_id": TASK_ID, "owner": OWNER, "output": str(output.relative_to(root)),
            "summary_sha256": sha256(output / "summary.json"), "actual_cost": cost,
            "fit400_decision": fit400_decision,
            "predictions": str((output / "val_simple_controls/predictions.jsonl").relative_to(root)),
            "predictions_sha256": evaluation["predictions_sha256"],
            "new_detector_or_backbone_forwards": 0, "head_updates": 0,
        })
        print(json.dumps({"status": summary["status"], "fit400_decision": fit400_decision,
                          "cost": cost}, sort_keys=True), flush=True)
    except BaseException as error:
        atomic(output / "failure.json", {
            "status": "failed_preserved", "task_id": TASK_ID, "owner": OWNER,
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
            "new_detector_or_backbone_forwards": 0, "head_updates": 0,
        })
        raise


if __name__ == "__main__":
    main()
