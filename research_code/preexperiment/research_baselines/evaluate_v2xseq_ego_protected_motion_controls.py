#!/usr/bin/env python3
"""No-learning ego-priority add-only controls over frozen V2X-Seq packets."""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evaluate_v2xseq_real_motion_controls import (
    add_stat,
    append,
    arrays,
    atomic,
    average_precision,
    blank,
    frame_match,
    iou_matrix,
    load_jsonl,
    load_packets,
    resolve,
    sha256,
)
from v2xseq_real_motion_packets import stale_and_motion_arms


ROOT = Path(__file__).resolve().parents[2]
AGES = (0, 100, 200, 400)
METHODS = (
    "ego_protected_stale",
    "ego_protected_constant_velocity",
    "ego_protected_two_hit_constant_velocity",
)


def ego_protected_merge(
    local: tuple[np.ndarray, np.ndarray, np.ndarray],
    remote: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Keep every local detection; add only range-valid nonconflicting remote boxes."""
    local_boxes, local_scores, local_labels = local
    remote_boxes, remote_scores, remote_labels = remote
    local_boxes = np.asarray(local_boxes, dtype=np.float32).reshape(-1, 7)
    local_scores = np.asarray(local_scores, dtype=np.float32).reshape(-1)
    local_labels = np.asarray(local_labels, dtype=np.int32).reshape(-1)
    remote_boxes = np.asarray(remote_boxes, dtype=np.float32).reshape(-1, 7)
    remote_scores = np.asarray(remote_scores, dtype=np.float32).reshape(-1)
    remote_labels = np.asarray(remote_labels, dtype=np.int32).reshape(-1)
    if not (len(local_boxes) == len(local_scores) == len(local_labels)):
        raise ValueError("local box/score/label counts differ")
    if not (len(remote_boxes) == len(remote_scores) == len(remote_labels)):
        raise ValueError("remote box/score/label counts differ")
    remote_input = len(remote_boxes)
    in_range = (
        (remote_boxes[:, 0] >= 0.0) & (remote_boxes[:, 0] <= 96.0)
        & (remote_boxes[:, 1] >= -40.0) & (remote_boxes[:, 1] <= 40.0)
    ) if len(remote_boxes) else np.empty(0, dtype=bool)
    range_dropped = int((~in_range).sum())
    remote_boxes = remote_boxes[in_range]
    remote_scores = remote_scores[in_range]
    remote_labels = remote_labels[in_range]
    local_remote = iou_matrix(remote_boxes, local_boxes)
    remote_remote = iou_matrix(remote_boxes, remote_boxes)
    kept: list[int] = []
    conflict_dropped = 0
    self_nms_dropped = 0
    for index in np.argsort(-remote_scores, kind="stable"):
        if len(local_boxes) and float(local_remote[index].max(initial=0.0)) > threshold:
            conflict_dropped += 1
            continue
        if kept and float(remote_remote[index, kept].max(initial=0.0)) > threshold:
            self_nms_dropped += 1
            continue
        kept.append(int(index))
    keep = np.asarray(kept, dtype=np.int64)
    boxes = np.concatenate((local_boxes, remote_boxes[keep]), axis=0)
    scores = np.concatenate((local_scores, remote_scores[keep]), axis=0)
    labels = np.concatenate((local_labels, remote_labels[keep]), axis=0)
    return boxes, scores, labels, {
        "local_count": int(len(local_boxes)),
        "remote_input": int(remote_input),
        "remote_retained": int(len(keep)),
        "remote_range_dropped": range_dropped,
        "remote_conflict_dropped": conflict_dropped,
        "remote_self_nms_dropped": self_nms_dropped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = resolve(args.protocol_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, spec in config["frozen_inputs"].items():
        if sha256(resolve(spec["path"])) != spec["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
    upstream = json.loads(resolve(config["upstream_review"]["path"]).read_text(encoding="utf-8"))
    if upstream.get("status") != "passed" or not all(upstream.get("checks", {}).values()):
        raise RuntimeError("offline repaired parent review is not fully passing")
    ego_rows = load_jsonl(resolve(config["paths"]["ego_geometry"]))
    if len(ego_rows) != int(config["dataset"]["frames"]):
        raise RuntimeError("ego geometry coverage changed")
    packets = load_packets(resolve(config["paths"]["infra_result"]))
    packet_keys = {
        (sequence, int(value["record"]["local_index"]))
        for sequence, values in packets.items() for value in values
    }
    ego_keys = {(str(row["sequence"]), int(row["local_index"])) for row in ego_rows}
    if packet_keys != ego_keys or len(ego_keys) != len(ego_rows):
        raise RuntimeError("ego/infra key sets differ")

    stats = {(age, method, threshold): blank() for age in AGES for method in METHODS for threshold in (0.5, 0.7)}
    stable = defaultdict(lambda: [0, 0])
    prediction_path = output / "frame_predictions.jsonl"
    metric_path = output / "frame_metrics.jsonl"
    threshold = float(config["fusion"]["remote_conflict_iou_threshold"])
    for ordinal, ego in enumerate(ego_rows, start=1):
        sequence = str(ego["sequence"])
        local_index = int(ego["local_index"])
        local_boxes, local_scores, local_labels, gt = arrays(ego)
        local = (local_boxes, local_scores, local_labels)
        ego_match = frame_match(*local, gt, 0.7)
        current_pose = np.asarray(
            packets[sequence][local_index]["record"]["receiver_pose_lidar_to_world"], dtype=np.float64
        )
        receiver_timestamp = int(packets[sequence][local_index]["record"]["receiver_timestamp_us"])
        source_rows = packets[sequence]
        source_times = [int(value["decoded"]["metadata"]["source_timestamp_us"]) for value in source_rows]
        for age in AGES:
            chosen_index = bisect.bisect_right(source_times, receiver_timestamp - age * 1000) - 1
            chosen = source_rows[chosen_index] if chosen_index >= 0 else None
            if chosen is None:
                arms = None
                packet_bytes = 0
                source_frame = None
                horizon_s = None
            else:
                previous = None
                if chosen_index > 0:
                    previous_decoded = source_rows[chosen_index - 1]["decoded"]
                    previous = {
                        "boxes": previous_decoded["boxes"],
                        "source_pose": np.asarray(previous_decoded["metadata"]["source_pose_lidar_to_world"]),
                        "source_timestamp_us": int(previous_decoded["metadata"]["source_timestamp_us"]),
                    }
                arms = stale_and_motion_arms(chosen["decoded"], previous, current_pose, receiver_timestamp)
                packet_bytes = int(chosen["bytes"])
                source_frame = chosen["decoded"]["metadata"]["source_frame"]
                horizon_s = float(arms["horizon_s"])
            for method in METHODS:
                if arms is None:
                    boxes, scores, labels = local
                    diagnostic = {
                        "local_count": len(local_boxes), "remote_input": 0, "remote_retained": 0,
                        "remote_range_dropped": 0, "remote_conflict_dropped": 0,
                        "remote_self_nms_dropped": 0,
                    }
                    charged_bytes = 0
                else:
                    remote_boxes = {
                        "ego_protected_stale": arms["stale_boxes"],
                        "ego_protected_constant_velocity": arms["constant_velocity_boxes"],
                        "ego_protected_two_hit_constant_velocity": arms["two_hit_constant_velocity_boxes"],
                    }[method]
                    mask = (
                        arms["two_hit_mask"]
                        if method == "ego_protected_two_hit_constant_velocity"
                        else np.ones(len(arms["scores"]), dtype=bool)
                    )
                    boxes, scores, labels, diagnostic = ego_protected_merge(
                        local,
                        (remote_boxes, arms["scores"][mask], arms["labels"][mask]),
                        threshold=threshold,
                    )
                    charged_bytes = packet_bytes
                matches = {value: frame_match(boxes, scores, labels, gt, value) for value in (0.5, 0.7)}
                for value in (0.5, 0.7):
                    add_stat(stats[(age, method, value)], matches[value])
                stable[(age, method)][0] += int(
                    np.logical_and(ego_match["matched_gt"], matches[0.7]["matched_gt"]).sum()
                )
                stable[(age, method)][1] += int(ego_match["matched_gt"].sum())
                prediction = {
                    "sequence": sequence,
                    "local_index": local_index,
                    "global_index": int(ego["global_index"]),
                    "age_ms": age,
                    "method": method,
                    "source_frame": source_frame,
                    "horizon_s": horizon_s,
                    "packet_bytes": charged_bytes,
                    **diagnostic,
                    "boxes_3d_lidar": boxes.tolist(),
                    "scores": scores.tolist(),
                    "label_preds": labels.tolist(),
                    "gt_boxes": gt.tolist(),
                }
                append(prediction_path, prediction)
                append(metric_path, {
                    **{
                        name: prediction[name]
                        for name in (
                            "sequence", "local_index", "global_index", "age_ms", "method",
                            "source_frame", "horizon_s", "packet_bytes", "local_count",
                            "remote_input", "remote_retained", "remote_range_dropped",
                            "remote_conflict_dropped", "remote_self_nms_dropped",
                        )
                    },
                    "detections": len(boxes),
                    "gt": int(sum(int(value[7]) == 0 for value in gt)),
                    "tp50": int(sum(matches[0.5]["tp"])),
                    "fp50": int(sum(matches[0.5]["fp"])),
                    "tp70": int(sum(matches[0.7]["tp"])),
                    "fp70": int(sum(matches[0.7]["fp"])),
                })
        if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(ego_rows):
            print(f"ego_protected_controls progress={ordinal}/{len(ego_rows)}", flush=True)

    method_rows = []
    for age in AGES:
        for method in METHODS:
            row: dict[str, Any] = {"age_ms": age, "method": method}
            for value in (0.5, 0.7):
                stat = stats[(age, method, value)]
                suffix = int(value * 100)
                row[f"ap{suffix}"] = average_precision(stat)
                row[f"recall{suffix}"] = float(sum(stat["tp"]) / stat["gt"])
                row[f"tp{suffix}"] = int(sum(stat["tp"]))
                row[f"fp{suffix}"] = int(sum(stat["fp"]))
            row["gt"] = int(stats[(age, method, 0.5)]["gt"])
            row["stable_ego_tp70_retention"] = float(
                stable[(age, method)][0] / max(stable[(age, method)][1], 1)
            )
            method_rows.append(row)
    summary = {
        "status": "completed",
        "role": config["role"],
        "frames": len(ego_rows),
        "sequences": len(packets),
        "conditions": len(ego_rows) * len(AGES) * len(METHODS),
        "methods": method_rows,
        "predictions_sha256": sha256(prediction_path),
        "frame_metrics_sha256": sha256(metric_path),
        "limits": config["interpretation_boundaries"],
    }
    atomic(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
