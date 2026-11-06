#!/usr/bin/env python3
"""Offline real-time stale/CV/two-hit controls from complete V2X-Seq artifacts."""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from shapely.geometry import Polygon

from v2xseq_real_motion_packets import decode_packet, stale_and_motion_arms


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("ego_only", "stale_box", "constant_velocity", "two_hit_constant_velocity")
AGES = (0, 100, 200, 400)


def resolve(path: str | os.PathLike[str]) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(row["boxes_3d_lidar"], dtype=np.float32).reshape(-1, 7)
    scores = np.asarray(row["scores"], dtype=np.float32).reshape(-1)
    labels = np.asarray(row["label_preds"], dtype=np.int32).reshape(-1)
    gt = np.asarray(row["gt_boxes"], dtype=np.float32)
    gt = gt.reshape(-1, gt.shape[-1] if gt.size else 8)
    if not (len(boxes) == len(scores) == len(labels)):
        raise RuntimeError("detection schema mismatch")
    return boxes, scores, labels, gt


def polygon(box: np.ndarray) -> Polygon:
    x, y, _, _, width, length, yaw = box[:7]
    local = np.asarray([[length/2, -width/2], [length/2, width/2], [-length/2, width/2], [-length/2, -width/2]])
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    corners = local @ np.asarray([[cosine, -sine], [sine, cosine]]).T
    corners += [x, y]
    return Polygon(corners)


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros((len(left), len(right)), dtype=np.float64)
    lp = [polygon(box) for box in left]
    rp = [polygon(box) for box in right]
    for i, a in enumerate(lp):
        for j, b in enumerate(rp):
            union = a.union(b).area
            result[i, j] = a.intersection(b).area / union if union else 0.0
    return result


def merge(local: tuple[np.ndarray, np.ndarray, np.ndarray], remote: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    local_boxes, local_scores, local_labels = local
    remote_boxes, remote_scores, remote_labels = remote
    boxes = np.concatenate((local_boxes, remote_boxes))
    scores = np.concatenate((local_scores, remote_scores))
    labels = np.concatenate((local_labels, remote_labels))
    origin = np.concatenate((np.zeros(len(local_boxes), dtype=np.int8), np.ones(len(remote_boxes), dtype=np.int8)))
    if not len(boxes):
        return boxes, scores, labels, {"remote_input": 0, "remote_retained": 0, "nms_suppressed": 0}
    order = np.argsort(-scores, kind="stable")
    pairwise = iou_matrix(boxes, boxes)
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for index in order:
        if suppressed[index]:
            continue
        keep.append(int(index))
        suppressed |= pairwise[index] > 0.15
        suppressed[index] = False
    keep_array = np.asarray(keep, dtype=np.int64)
    in_range = (
        (boxes[keep_array, 0] >= 0.0) & (boxes[keep_array, 0] <= 96.0)
        & (boxes[keep_array, 1] >= -40.0) & (boxes[keep_array, 1] <= 40.0)
    )
    keep_array = keep_array[in_range]
    return (
        boxes[keep_array], scores[keep_array], labels[keep_array],
        {
            "remote_input": int(len(remote_boxes)),
            "remote_retained": int(origin[keep_array].sum()),
            "nms_suppressed": int(len(boxes) - len(keep)),
        },
    )


def frame_match(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, gt_all: np.ndarray, threshold: float) -> dict[str, Any]:
    boxes = boxes[labels == 0]
    scores = scores[labels == 0]
    gt = gt_all[gt_all[:, 7].astype(np.int64) == 0, :7] if len(gt_all) else np.empty((0, 7), dtype=np.float32)
    ious = iou_matrix(boxes, gt)
    remaining = list(range(len(gt)))
    tp = np.zeros(len(boxes), dtype=np.int8)
    matched_gt = np.zeros(len(gt), dtype=bool)
    for detection in np.argsort(-scores, kind="stable"):
        if not remaining:
            break
        values = ious[detection, remaining]
        position = int(np.argmax(values))
        if float(values[position]) >= threshold:
            gt_index = remaining.pop(position)
            tp[detection] = 1
            matched_gt[gt_index] = True
    return {"tp": tp.tolist(), "fp": (1 - tp).tolist(), "score": scores.astype(float).tolist(), "gt": len(gt), "matched_gt": matched_gt}


def average_precision(stat: dict[str, Any]) -> float:
    score = np.asarray(stat["score"]); tp = np.asarray(stat["tp"]); fp = np.asarray(stat["fp"])
    if int(stat["gt"]) <= 0:
        return 0.0
    order = np.argsort(-score, kind="stable"); tp = np.cumsum(tp[order]); fp = np.cumsum(fp[order])
    recall = tp / int(stat["gt"]); precision = tp / np.maximum(tp + fp, 1)
    recall = np.r_[0, recall, 1]; precision = np.r_[0, precision, 0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def add_stat(target: dict[str, Any], value: dict[str, Any]) -> None:
    for key in ("tp", "fp", "score"):
        target[key].extend(value[key])
    target["gt"] += int(value["gt"])


def blank() -> dict[str, Any]:
    return {"tp": [], "fp": [], "score": [], "gt": 0}


def load_packets(run: Path) -> dict[str, list[dict[str, Any]]]:
    geometry = load_jsonl(run / "detection_geometry.jsonl")
    indices = load_jsonl(run / "packet_index.jsonl")
    if len(geometry) != len(indices):
        raise RuntimeError("infra geometry and packet index counts differ")
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (run / "packets.bin").open("rb") as stream:
        for record, index in zip(geometry, indices):
            if (record["sequence"], int(record["local_index"])) != (index["sequence"], int(index["local_index"])):
                raise RuntimeError("packet key order differs")
            stream.seek(int(index["offset"])); payload = stream.read(int(index["bytes"]))
            if hashlib.sha256(payload).hexdigest() != index["sha256"]:
                raise RuntimeError("packet payload hash mismatch")
            decoded = decode_packet(payload)
            output[str(record["sequence"])].append({"decoded": decoded, "bytes": len(payload), "record": record})
    for sequence, rows in output.items():
        rows.sort(key=lambda value: int(value["decoded"]["metadata"]["source_timestamp_us"]))
        timestamps = [int(value["decoded"]["metadata"]["source_timestamp_us"]) for value in rows]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise RuntimeError(f"non-increasing source timestamps in {sequence}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = resolve(args.protocol_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = resolve(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    for name, spec in config["frozen_inputs"].items():
        if sha256(resolve(spec["path"])) != spec["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
    review = json.loads(resolve(config["frozen_inputs"]["infra_review"]["path"]).read_text())
    if review.get("status") != "passed" or not all(review.get("checks", {}).values()):
        raise RuntimeError("infra full review is not fully passing")
    ego_rows = load_jsonl(resolve(config["frozen_inputs"]["ego_geometry"]["path"]))
    if len(ego_rows) != int(config["dataset"]["frames"]):
        raise RuntimeError("ego geometry coverage changed")
    packets = load_packets(resolve(config["paths"]["infra_result"]))
    packet_keys = {(sequence, int(value["record"]["local_index"])) for sequence, values in packets.items() for value in values}
    ego_keys = {(str(row["sequence"]), int(row["local_index"])) for row in ego_rows}
    if packet_keys != ego_keys or len(ego_keys) != len(ego_rows):
        raise RuntimeError("ego/infra key sets differ")

    stats = {(age, method, threshold): blank() for age in AGES for method in METHODS for threshold in (0.5, 0.7)}
    scene_stats = defaultdict(blank)
    stable = defaultdict(lambda: [0, 0])
    prediction_path = output / "frame_predictions.jsonl"
    metric_path = output / "frame_metrics.jsonl"
    for ordinal, ego in enumerate(ego_rows, start=1):
        sequence = str(ego["sequence"]); local_index = int(ego["local_index"])
        local_boxes, local_scores, local_labels, gt = arrays(ego)
        current_pose = np.asarray(packets[sequence][local_index]["record"]["receiver_pose_lidar_to_world"], dtype=np.float64)
        receiver_timestamp = int(packets[sequence][local_index]["record"]["receiver_timestamp_us"])
        source_rows = packets[sequence]
        source_times = [int(value["decoded"]["metadata"]["source_timestamp_us"]) for value in source_rows]
        local = (local_boxes, local_scores, local_labels)
        ego_match = {threshold: frame_match(*local, gt, threshold) for threshold in (0.5, 0.7)}
        for age in AGES:
            chosen_index = bisect.bisect_right(source_times, receiver_timestamp - age * 1000) - 1
            chosen = source_rows[chosen_index] if chosen_index >= 0 else None
            if chosen is None:
                arms = None; packet_bytes = 0; source_frame = None; horizon_s = None
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
                if method == "ego_only" or arms is None:
                    boxes, scores, labels = local
                    diagnostics = {"remote_input": 0, "remote_retained": 0, "nms_suppressed": 0}
                    charged_bytes = 0 if method == "ego_only" else packet_bytes
                else:
                    remote_boxes = {
                        "stale_box": arms["stale_boxes"],
                        "constant_velocity": arms["constant_velocity_boxes"],
                        "two_hit_constant_velocity": arms["two_hit_constant_velocity_boxes"],
                    }[method]
                    remote_mask = arms["two_hit_mask"] if method == "two_hit_constant_velocity" else np.ones(len(arms["scores"]), dtype=bool)
                    boxes, scores, labels, diagnostics = merge(
                        local,
                        (remote_boxes, arms["scores"][remote_mask], arms["labels"][remote_mask]),
                    )
                    charged_bytes = packet_bytes
                matches = {threshold: frame_match(boxes, scores, labels, gt, threshold) for threshold in (0.5, 0.7)}
                for threshold in (0.5, 0.7):
                    add_stat(stats[(age, method, threshold)], matches[threshold])
                    add_stat(scene_stats[(sequence, age, method, threshold)], matches[threshold])
                stable[(age, method)][0] += int(np.logical_and(ego_match[0.7]["matched_gt"], matches[0.7]["matched_gt"]).sum())
                stable[(age, method)][1] += int(ego_match[0.7]["matched_gt"].sum())
                prediction = {
                    "sequence": sequence, "local_index": local_index, "global_index": int(ego["global_index"]),
                    "age_ms": age, "method": method, "source_frame": source_frame, "horizon_s": horizon_s,
                    "packet_bytes": charged_bytes, **diagnostics,
                    "boxes_3d_lidar": boxes.tolist(), "scores": scores.tolist(), "label_preds": labels.tolist(), "gt_boxes": gt.tolist(),
                }
                append(prediction_path, prediction)
                append(metric_path, {
                    **{key: prediction[key] for key in ("sequence", "local_index", "global_index", "age_ms", "method", "source_frame", "horizon_s", "packet_bytes", "remote_input", "remote_retained", "nms_suppressed")},
                    "detections": len(boxes), "gt": len(gt),
                    "tp50": int(sum(matches[0.5]["tp"])), "fp50": int(sum(matches[0.5]["fp"])),
                    "tp70": int(sum(matches[0.7]["tp"])), "fp70": int(sum(matches[0.7]["fp"])),
                })
        if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(ego_rows):
            print(f"motion_controls progress={ordinal}/{len(ego_rows)}", flush=True)

    method_rows = []
    scene_rows = []
    for age in AGES:
        for method in METHODS:
            row = {"age_ms": age, "method": method}
            for threshold in (0.5, 0.7):
                value = stats[(age, method, threshold)]
                row[f"ap{int(threshold*100)}"] = average_precision(value)
                row[f"recall{int(threshold*100)}"] = float(sum(value["tp"]) / value["gt"])
                row[f"tp{int(threshold*100)}"] = int(sum(value["tp"])); row[f"fp{int(threshold*100)}"] = int(sum(value["fp"]))
            row["gt"] = stats[(age, method, 0.5)]["gt"]
            row["stable_ego_tp70_retention"] = float(stable[(age, method)][0] / max(stable[(age, method)][1], 1))
            method_rows.append(row)
            for sequence in sorted(packets):
                scene_rows.append({
                    "sequence": sequence, "age_ms": age, "method": method,
                    "ap50": average_precision(scene_stats[(sequence, age, method, 0.5)]),
                    "ap70": average_precision(scene_stats[(sequence, age, method, 0.7)]),
                })
    summary = {
        "status": "completed", "role": config["role"], "frames": len(ego_rows), "sequences": len(packets),
        "conditions": len(ego_rows) * len(AGES) * len(METHODS), "methods": method_rows,
        "predictions_sha256": sha256(prediction_path), "frame_metrics_sha256": sha256(metric_path),
        "limits": ["author21 exposed development", "actual serialized application payload bytes", "no GT/future input to prediction construction"],
    }
    atomic(output / "scene_metrics.json", {"rows": scene_rows})
    atomic(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
