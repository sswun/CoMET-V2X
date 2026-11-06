"""Core utilities for the cached stale+one-propagated history selector."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from audit_v2xseq_strict_common_roi_20261010 import mask as common_mask
from e1_receiver_value_core_v1_20261020 import (
    atomic, feature_matrix, greedy, iou32, load_npz, read_rows,
    remote_indices, sha256, unique_ego_scores,
)
from v2xseq_real_motion_packets import transform_boxes7d


WAITS = (0, 100, 200, 400)
FEATURE_NAMES = (
    "actual_age_ms", "source_score", "receiver_x", "receiver_y", "receiver_z",
    "box_h", "box_w", "box_l", "receiver_sin_yaw", "receiver_cos_yaw",
    "receiver_range_xy", "receiver_abs_y", "source_x", "source_y", "source_z",
    "source_range_xy", "source_sin_yaw", "source_cos_yaw",
    "source_to_receiver_tx", "source_to_receiver_ty", "source_to_receiver_tz",
    "source_to_receiver_range_xy", "source_to_receiver_sin_yaw",
    "source_to_receiver_cos_yaw", "nearest_ego_center_distance",
    "nearest_ego_abs_dx", "nearest_ego_abs_dy", "max_ego_bev_iou",
    "nearest_ego_prediction_score", "ego_boxes_within_10m",
    "hypothesis_is_ego", "hypothesis_is_propagated", "has_causal_association",
    "horizon_s", "history_dt_s", "speed_mps", "association_distance_m",
    "association_distance_over_gate", "propagation_shift_m",
)
assert len(FEATURE_NAMES) == 39

BASE_METHODS = ("ego", "stale_raw", "prop_raw", "dual_raw")
LEARNED_METHODS = (
    "stale_quality_aug", "dual_quality_aug",
    "stale_utility_aug", "dual_utility_aug",
    "dual_quality_base", "dual_utility_base",
)
ALL_METHODS = BASE_METHODS + LEARNED_METHODS
BETAS = (1.0, 1.2, 1.4)


def append(stream, value: Any) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False) + "\n")


def source_record_from_npz(row: dict[str, Any], root: Path) -> dict[str, Any]:
    raw_path = Path(row["raw_path"])
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    return {
        "sequence": row["sequence"],
        "frame": row["source_frame"],
        "timestamp_us": int(row["source_timestamp_us"]),
        "pose": np.asarray(row["source_pose_lidar_to_world"], np.float64),
        "raw": load_npz(raw_path),
        "raw_path": str(raw_path),
        "raw_sha256": row.get("raw_sha256", sha256(raw_path)),
    }


def source_record_from_json(row: dict[str, Any], frame_key: str) -> dict[str, Any]:
    return {
        "sequence": row["sequence"],
        "frame": row[frame_key],
        "timestamp_us": int(row["source_timestamp_us"]),
        "pose": np.asarray(row["source_pose_lidar_to_world"], np.float64),
        "raw": {
            "pred_boxes": np.asarray(row["boxes_3d_lidar"], np.float32).reshape(-1, 7),
            "pred_scores": np.asarray(row["scores"], np.float32),
            "pred_labels": np.asarray(row["label_preds"], np.int32),
        },
        "raw_path": None,
        "raw_sha256": None,
    }


def previous_map(sources: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any] | None]:
    output: dict[tuple[str, str], dict[str, Any] | None] = {}
    for sequence in sorted({key[0] for key in sources}):
        ordered = sorted((row for key, row in sources.items() if key[0] == sequence),
                         key=lambda row: row["timestamp_us"])
        assert all(a["timestamp_us"] < b["timestamp_us"] for a, b in zip(ordered, ordered[1:]))
        for index, row in enumerate(ordered):
            output[(sequence, row["frame"])] = None if index == 0 else ordered[index - 1]
    return output


def propagate(current: dict[str, Any], previous: dict[str, Any] | None, horizon_s: float) -> dict[str, Any]:
    boxes = current["raw"]["pred_boxes"].astype(np.float32).copy()
    n = len(boxes)
    confirmed = np.zeros(n, bool)
    speed = np.zeros(n, np.float64)
    distance_value = np.zeros(n, np.float64)
    ratio = np.zeros(n, np.float64)
    shift = np.zeros(n, np.float64)
    history_dt = 0.0
    if previous is None or horizon_s <= 0 or not n or not len(previous["raw"]["pred_boxes"]):
        return dict(boxes=boxes, confirmed=confirmed, speed=speed, distance=distance_value,
                    ratio=ratio, shift=shift, history_dt_s=history_dt)
    history_dt = (current["timestamp_us"] - previous["timestamp_us"]) / 1e6
    if history_dt <= 0 or history_dt > .5:
        return dict(boxes=boxes, confirmed=confirmed, speed=speed, distance=distance_value,
                    ratio=ratio, shift=shift, history_dt_s=float(history_dt))
    aligned = transform_boxes7d(previous["raw"]["pred_boxes"], previous["pose"], current["pose"])
    distances = np.linalg.norm(boxes[:, None, :2] - aligned[None, :, :2], axis=2)
    rows, columns = linear_sum_assignment(distances)
    gate = 1.0 + 50.0 * history_dt
    for row, column in zip(rows, columns):
        value = float(distances[row, column])
        if value > gate:
            continue
        velocity = (boxes[row, :2] - aligned[column, :2]) / history_dt
        magnitude = float(np.linalg.norm(velocity))
        if magnitude > 50.0:
            velocity *= 50.0 / magnitude
            magnitude = 50.0
        displacement = velocity * horizon_s
        boxes[row, :2] += displacement.astype(np.float32)
        confirmed[row] = True
        speed[row] = magnitude
        distance_value[row] = value
        ratio[row] = value / gate
        shift[row] = float(np.linalg.norm(displacement))
    return dict(boxes=boxes, confirmed=confirmed, speed=speed, distance=distance_value,
                ratio=ratio, shift=shift, history_dt_s=float(history_dt))


def make_condition(
    ego: dict[str, np.ndarray], source: dict[str, Any] | None,
    previous: dict[str, Any] | None, receiver_pose: np.ndarray,
    receiver_timestamp_us: int, age_ms: float, protect_threshold: float,
) -> dict[str, Any]:
    ego_mask = common_mask(ego["pred_boxes"]) & (ego["pred_labels"].astype(int) == 0)
    ego_indices = np.flatnonzero(ego_mask).astype(np.int32)
    ego_boxes = ego["pred_boxes"][ego_mask].astype(np.float32)
    ego_scores = ego["pred_scores"][ego_mask].astype(np.float64)
    protected = ego_scores >= protect_threshold
    protected_boxes, protected_scores = ego_boxes[protected], ego_scores[protected]
    ego_base = feature_matrix(ego_boxes, ego_boxes, ego_scores, 0.0, np.eye(4),
                              protected_boxes, protected_scores)
    ego_extra = np.zeros((len(ego_boxes), 9), np.float64)
    ego_extra[:, 0] = 1.0

    history_boxes: list[np.ndarray] = []
    history_source_boxes: list[np.ndarray] = []
    history_scores: list[float] = []
    history_indices: list[int] = []
    history_types: list[int] = []
    history_extra: list[list[float]] = []
    diagnostic = {
        "raw_remote": 0, "common_roi_remote": 0, "remote_self_nms_removed": 0,
        "stale_hypotheses": 0, "propagated_hypotheses": 0,
    }
    if source is not None:
        horizon_s = max(0.0, (int(receiver_timestamp_us) - int(source["timestamp_us"])) / 1e6)
        assert abs(horizon_s * 1000.0 - age_ms) < .0011
        motion = propagate(source, previous, horizon_s)
        stale = transform_boxes7d(source["raw"]["pred_boxes"], source["pose"], receiver_pose)
        propagated = transform_boxes7d(motion["boxes"], source["pose"], receiver_pose)
        selected, nms = remote_indices(stale, source["raw"]["pred_scores"],
                                       source["raw"]["pred_labels"])
        diagnostic.update(nms)
        transform = np.linalg.solve(receiver_pose, source["pose"])
        for source_index in selected.astype(int):
            associated = bool(motion["confirmed"][source_index])
            common = [
                0.0, 0.0, float(associated), horizon_s,
                float(motion["history_dt_s"] if associated else 0.0),
                float(motion["speed"][source_index] if associated else 0.0),
                float(motion["distance"][source_index] if associated else 0.0),
                float(motion["ratio"][source_index] if associated else 0.0),
                float(motion["shift"][source_index] if associated else 0.0),
            ]
            history_boxes.append(stale[source_index]); history_source_boxes.append(source["raw"]["pred_boxes"][source_index])
            history_scores.append(float(source["raw"]["pred_scores"][source_index])); history_indices.append(source_index)
            history_types.append(1); history_extra.append(common)
            if associated and bool(common_mask(propagated[source_index:source_index + 1])[0]):
                propagated_extra = common.copy(); propagated_extra[1] = 1.0
                history_boxes.append(propagated[source_index]); history_source_boxes.append(source["raw"]["pred_boxes"][source_index])
                history_scores.append(float(source["raw"]["pred_scores"][source_index])); history_indices.append(source_index)
                history_types.append(2); history_extra.append(propagated_extra)
        diagnostic["stale_hypotheses"] = int(sum(kind == 1 for kind in history_types))
        diagnostic["propagated_hypotheses"] = int(sum(kind == 2 for kind in history_types))
        if history_boxes:
            hb = np.asarray(history_boxes, np.float32).reshape(-1, 7)
            hsb = np.asarray(history_source_boxes, np.float32).reshape(-1, 7)
            hs = np.asarray(history_scores, np.float64)
            history_base = feature_matrix(hb, hsb, hs, age_ms, transform,
                                          protected_boxes, protected_scores)
            history_features = np.column_stack((history_base, np.asarray(history_extra, np.float64)))
        else:
            hb = np.empty((0, 7), np.float32); hs = np.empty(0, np.float64)
            history_features = np.empty((0, 39), np.float64)
    else:
        hb = np.empty((0, 7), np.float32); hs = np.empty(0, np.float64)
        history_features = np.empty((0, 39), np.float64)

    boxes = np.concatenate((ego_boxes, hb), axis=0)
    raw_scores = np.r_[ego_scores, hs].astype(np.float64)
    hypothesis_type = np.r_[np.zeros(len(ego_boxes), np.int8), np.asarray(history_types, np.int8)]
    features = np.concatenate((np.column_stack((ego_base, ego_extra)), history_features), axis=0).astype(np.float32)
    is_protected = np.r_[protected, np.zeros(len(hb), bool)].astype(np.int8)
    source_indices = np.r_[ego_indices, np.asarray(history_indices, np.int32)]
    gt_mask = common_mask(ego["gt_boxes"]) & (ego["gt_labels"].astype(int) == 0)
    gt_indices = np.flatnonzero(gt_mask).astype(np.int32)
    gt_boxes = ego["gt_boxes"][gt_mask].astype(np.float32)
    matrix = iou32(boxes, gt_boxes)
    protected_ids = np.flatnonzero(is_protected)
    protected_order = protected_ids[np.argsort(-unique_ego_scores(raw_scores[protected_ids]), kind="stable")]
    protected_matches = greedy(matrix, protected_order, .7)
    covered = {int(value) for value in protected_matches if value >= 0}
    quality = (matrix.max(axis=1, initial=0.0) >= .7).astype(np.int8)
    uncovered = [index for index in range(len(gt_boxes)) if index not in covered]
    utility = (matrix[:, uncovered].max(axis=1, initial=0.0) >= .7).astype(np.int8)
    assert np.all(utility <= quality)
    return {
        "boxes": boxes, "raw_scores": raw_scores, "hypothesis_type": hypothesis_type,
        "is_protected": is_protected, "source_detection_indices": source_indices,
        "features": features, "quality_y": quality, "utility_y": utility,
        "gt_boxes": gt_boxes, "gt_original_indices": gt_indices, "ious": matrix,
        "diagnostic": diagnostic,
    }


class PackedDataset:
    def __init__(self, directory: Path):
        self.directory = directory
        self.meta = list(read_rows(directory / "conditions.jsonl"))
        self.data = load_npz(directory / "pools.npz")

    def pool(self, index: int) -> dict[str, np.ndarray]:
        meta = self.meta[index]
        a, b, ga, gb = meta["start"], meta["stop"], meta["gt_start"], meta["gt_stop"]
        value = {key: self.data[key][a:b] for key in (
            "boxes", "raw_scores", "hypothesis_type", "is_protected",
            "source_detection_indices", "features", "quality_y", "utility_y",
        )}
        value["gt_boxes"] = self.data["gt_boxes"][ga:gb]
        value["gt_original_indices"] = self.data["gt_original_indices"][ga:gb]
        value["ious"] = self.data["ious"][meta["iou_start"]:meta["iou_stop"]].reshape(b-a, gb-ga)
        return value

    def score_pool(self, index: int, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        meta = self.meta[index]
        return {key: value[meta["start"]:meta["stop"]] for key, value in scores.items()}


class PackWriter:
    def __init__(self, directory: Path):
        self.directory = directory; directory.mkdir(parents=True, exist_ok=False)
        self.stream = (directory / "conditions.jsonl").open("w")
        self.arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        self.cursor = self.gt_cursor = self.iou_cursor = self.conditions = 0
        self.started = time.monotonic()

    def add(self, metadata: dict[str, Any], value: dict[str, Any]) -> None:
        n, g = len(value["boxes"]), len(value["gt_boxes"])
        meta = dict(metadata, condition_index=self.conditions, start=self.cursor, stop=self.cursor+n,
                    gt_start=self.gt_cursor, gt_stop=self.gt_cursor+g,
                    iou_start=self.iou_cursor, iou_stop=self.iou_cursor+n*g,
                    protected_ids=np.flatnonzero(value["is_protected"]).astype(int).tolist(),
                    variable_ids=np.flatnonzero(value["is_protected"] == 0).astype(int).tolist(),
                    diagnostic=value["diagnostic"])
        append(self.stream, meta)
        for key, array in value.items():
            if key != "diagnostic":
                self.arrays[key].append(array.reshape(-1) if key == "ious" else array)
        self.cursor += n; self.gt_cursor += g; self.iou_cursor += n*g; self.conditions += 1

    def finish(self, provenance: dict[str, Any]) -> dict[str, Any]:
        self.stream.close()
        arrays = {key: np.concatenate(values, axis=0) for key, values in self.arrays.items()}
        np.savez_compressed(self.directory / "pools.npz", **arrays)
        summary = dict(provenance, status="complete", conditions=self.conditions,
                       candidates=self.cursor, gt_occurrences=self.gt_cursor,
                       iou_values=self.iou_cursor, wall_seconds=time.monotonic()-self.started,
                       pools_sha256=sha256(self.directory/"pools.npz"),
                       conditions_sha256=sha256(self.directory/"conditions.jsonl"))
        atomic(self.directory / "summary.json", summary)
        return summary


def learned_score_name(method: str) -> str:
    if "quality" in method:
        return "quality_aug" if method.endswith("aug") else "quality_base"
    if "utility" in method:
        return "utility_aug" if method.endswith("aug") else "utility_base"
    raise KeyError(method)


def pool_order(pool: dict[str, np.ndarray], scores: dict[str, np.ndarray], method: str,
               threshold: float | None = None, fallback: bool = False) -> tuple[np.ndarray, np.ndarray]:
    kind = pool["hypothesis_type"]
    ego = np.flatnonzero(kind == 0); stale = np.flatnonzero(kind == 1); prop = np.flatnonzero(kind == 2)
    if fallback or method == "ego":
        selected = ego; rank = unique_ego_scores(pool["raw_scores"][ego], stable=True)
    elif method in ("stale_raw", "prop_raw", "dual_raw"):
        remote = {"stale_raw": stale, "prop_raw": prop, "dual_raw": np.sort(np.r_[stale, prop])}[method]
        selected = np.r_[ego, remote]
        rank = np.r_[2.0 + unique_ego_scores(pool["raw_scores"][ego], stable=True),
                     unique_ego_scores(pool["raw_scores"][remote], stable=True)]
    else:
        fixed = np.flatnonzero(pool["is_protected"] == 1)
        allow_prop = method.startswith("dual_")
        variable = np.flatnonzero((pool["is_protected"] == 0) & ((kind != 2) | allow_prop))
        score_name = learned_score_name(method)
        variable_rank = unique_ego_scores(scores[score_name][variable], stable=True)
        if threshold is not None:
            keep = scores[score_name][variable] >= threshold
            variable, variable_rank = variable[keep], variable_rank[keep]
        selected = np.r_[fixed, variable]
        rank = np.r_[2.0 + unique_ego_scores(pool["raw_scores"][fixed], stable=True), variable_rank]
    order = np.argsort(-rank, kind="stable")
    return selected[order], rank[order].astype(np.float64)


def error_counts(matrix: np.ndarray, order: np.ndarray, matches: np.ndarray) -> dict[str, int]:
    maxima = matrix[order].max(axis=1, initial=0.0)
    wrong = matches < 0
    return {
        "tp": int((~wrong).sum()),
        "duplicate_fp": int((wrong & (maxima >= .7)).sum()),
        "localization_fp": int((wrong & (maxima >= .1) & (maxima < .7)).sum()),
        "background_fp": int((wrong & (maxima < .1)).sum()),
    }
