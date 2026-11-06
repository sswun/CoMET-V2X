#!/usr/bin/env python3
"""CPU-only independent review of fixed60 adapted V2X-Seq raw outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon


EXPECTED_FRAMES = 3316
EXPECTED_DELAYS = (0, 100, 200, 400)
THRESHOLDS = (0.5, 0.7)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def polygons_hwl(boxes: np.ndarray) -> list[Polygon]:
    """Independent NumPy/Shapely equivalent of boxes_to_corners2d(..., hwl)."""
    polygons: list[Polygon] = []
    template = np.asarray([[1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=np.float64) / 2.0
    for box in np.asarray(boxes, dtype=np.float64):
        length, width, angle = float(box[5]), float(box[4]), float(box[6])
        corners = template * np.asarray([length, width], dtype=np.float64)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
        corners = corners @ rotation + np.asarray([box[0], box[1]], dtype=np.float64)
        polygons.append(Polygon(corners))
    return polygons


def match_frame(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    threshold: float,
) -> tuple[list[float], list[int], list[int], int]:
    detection_mask = labels.astype(np.int64) == 0
    gt_mask = gt_labels.astype(np.int64) == 0
    boxes = boxes[detection_mask]
    scores = scores[detection_mask]
    gt_boxes = gt_boxes[gt_mask]
    order = np.argsort(-scores)
    detection_polygons = polygons_hwl(boxes)
    remaining = polygons_hwl(gt_boxes)
    true_positive: list[int] = []
    false_positive: list[int] = []
    for index in order:
        polygon = detection_polygons[int(index)]
        ious = np.asarray(
            [polygon.intersection(target).area / polygon.union(target).area for target in remaining],
            dtype=np.float32,
        )
        if len(remaining) == 0 or float(np.max(ious)) < threshold:
            true_positive.append(0)
            false_positive.append(1)
        else:
            true_positive.append(1)
            false_positive.append(0)
            remaining.pop(int(np.argmax(ious)))
    return scores[order].astype(float).tolist(), true_positive, false_positive, int(gt_mask.sum())


def voc_ap(tp: list[int], fp: list[int], scores: list[float], gt: int, global_sort: bool) -> dict[str, float]:
    tp_array = np.asarray(tp, dtype=np.int64)
    fp_array = np.asarray(fp, dtype=np.int64)
    if global_sort and len(scores):
        order = np.argsort(-np.asarray(scores, dtype=np.float64))
        tp_array = tp_array[order]
        fp_array = fp_array[order]
    cumulative_tp = np.cumsum(tp_array)
    cumulative_fp = np.cumsum(fp_array)
    recall = cumulative_tp.astype(np.float64) / max(int(gt), 1)
    precision = cumulative_tp.astype(np.float64) / np.maximum(cumulative_tp + cumulative_fp, 1)
    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.r_[0.0, precision, 0.0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))
    return {
        "ap": ap,
        "final_recall": float(recall[-1]) if len(recall) else 0.0,
        "final_precision": float(precision[-1]) if len(precision) else 0.0,
        "tp": int(tp_array.sum()),
        "fp": int(fp_array.sum()),
        "gt": int(gt),
    }


def self_test() -> dict[str, Any]:
    boxes = np.asarray(
        [
            [0, 0, 0, 1, 2, 4, 0],
            [20, 20, 0, 1, 2, 4, 0],
            [0.2, 0, 0, 1, 2, 4, 0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    labels = np.zeros(3, dtype=np.int64)
    gt = np.asarray([[0, 0, 0, 1, 2, 4, 0]], dtype=np.float32)
    gt_labels = np.zeros(1, dtype=np.int64)
    ordered_scores, tp, fp, count = match_frame(boxes, scores, labels, gt, gt_labels, 0.7)
    metrics = voc_ap(tp, fp, ordered_scores, count, True)
    checks = {
        "one_to_one_matching": tp == [1, 0, 0] and fp == [0, 1, 1],
        "class_filter_and_gt_count": count == 1,
        "global_voc_ap": abs(metrics["ap"] - 1.0) < 1e-12,
        "rotation_preserves_area": abs(polygons_hwl(boxes[[0]])[0].area - 8.0) < 1e-9,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "metrics": metrics}


def review_arm(arm: Path) -> dict[str, Any]:
    status = json.loads((arm / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((arm / "summary.json").read_text(encoding="utf-8"))
    records = read_jsonl(arm / "frame_manifest.jsonl")
    delay = int(summary["delay_ms"])
    eval_range = [float(value) for value in summary["eval_range"]]
    checks: dict[str, bool] = {
        "runner_terminal_completed": status.get("status") == "completed" and summary.get("status") == "completed",
        "fixed_epoch60_not_ap_selected": summary.get("fixed_epoch") == 60 and summary.get("checkpoint_selected_by_AP") is False,
        "corrected_index": summary.get("index_semantics") == "corrected_index",
        "exposed_val_boundary": summary.get("dataset_split") == "public_labeled_val21_exposed_not_hidden_test",
        "allowed_delay": delay in EXPECTED_DELAYS,
        "exact_manifest_coverage": len(records) == EXPECTED_FRAMES,
        "unique_complete_global_indices": sorted(int(row["global_index"]) for row in records) == list(range(EXPECTED_FRAMES)),
        "twenty_one_sequences": len({row["sequence"] for row in records}) == 21,
        "checkpoint_hash_exact": Path(summary["checkpoint"]).is_file() and sha256(Path(summary["checkpoint"])) == summary["checkpoint_sha256"],
    }
    stats: dict[float, dict[str, Any]] = {
        threshold: {"scores": [], "tp": [], "fp": [], "gt": 0} for threshold in THRESHOLDS
    }
    prediction_digest = hashlib.sha256()
    gt_digest = hashlib.sha256()
    raw_hash_failures: list[str] = []
    shape_failures: list[str] = []
    finite_failures: list[str] = []
    count_failures: list[dict[str, Any]] = []
    delay_failures: list[int] = []
    communication_non_na: list[int] = []
    for record in sorted(records, key=lambda row: int(row["global_index"])):
        global_index = int(record["global_index"])
        raw_path = Path(record["raw_path"])
        if not raw_path.is_file() or sha256(raw_path) != record["raw_sha256"]:
            raw_hash_failures.append(str(raw_path))
            continue
        with np.load(raw_path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ("pred_boxes", "pred_scores", "pred_labels", "gt_boxes", "gt_labels")}
        if not (
            arrays["pred_boxes"].ndim == 2
            and arrays["pred_boxes"].shape[1:] == (7,)
            and arrays["gt_boxes"].ndim == 2
            and arrays["gt_boxes"].shape[1:] == (7,)
            and len(arrays["pred_boxes"]) == len(arrays["pred_scores"]) == len(arrays["pred_labels"])
            and len(arrays["gt_boxes"]) == len(arrays["gt_labels"])
        ):
            shape_failures.append(str(raw_path))
            continue
        if any(not np.isfinite(value).all() for value in arrays.values()):
            finite_failures.append(str(raw_path))
            continue
        prediction_digest.update(arrays["pred_boxes"].tobytes())
        prediction_digest.update(arrays["pred_scores"].tobytes())
        prediction_digest.update(arrays["pred_labels"].tobytes())
        gt_digest.update(arrays["gt_boxes"].tobytes())
        gt_digest.update(arrays["gt_labels"].tobytes())
        boxes = arrays["pred_boxes"]
        range_mask = (
            (boxes[:, 0] >= eval_range[0])
            & (boxes[:, 0] <= eval_range[3])
            & (boxes[:, 1] >= eval_range[1])
            & (boxes[:, 1] <= eval_range[4])
        )
        for threshold in THRESHOLDS:
            scores, tp, fp, gt = match_frame(
                boxes[range_mask],
                arrays["pred_scores"][range_mask],
                arrays["pred_labels"][range_mask],
                arrays["gt_boxes"],
                arrays["gt_labels"],
                threshold,
            )
            stats[threshold]["scores"].extend(scores)
            stats[threshold]["tp"].extend(tp)
            stats[threshold]["fp"].extend(fp)
            stats[threshold]["gt"] += gt
            suffix = int(threshold * 100)
            expected_counts = (int(record[f"tp{suffix}"]), int(record[f"fp{suffix}"]), int(record["gt_vehicle"]))
            actual_counts = (sum(tp), sum(fp), gt)
            if actual_counts != expected_counts:
                count_failures.append(
                    {"global_index": global_index, "iou": threshold, "actual": actual_counts, "runner": expected_counts}
                )
        actual_delay = [float(value) for value in record.get("actual_time_delay_frames", [])]
        if record.get("delay_ms") != delay or actual_delay[:2] != [0.0, float(delay // 100)]:
            delay_failures.append(global_index)
        if record.get("communication_statistic") is not None:
            communication_non_na.append(global_index)

    metrics: dict[str, Any] = {"global_score_sort": {}, "upstream_frame_order": {}}
    for threshold in THRESHOLDS:
        values = stats[threshold]
        suffix = int(threshold * 100)
        global_values = voc_ap(values["tp"], values["fp"], values["scores"], values["gt"], True)
        upstream_values = voc_ap(values["tp"], values["fp"], values["scores"], values["gt"], False)
        metrics["global_score_sort"][f"ap{suffix}"] = global_values["ap"]
        metrics["global_score_sort"][f"final_recall_at_{suffix}"] = global_values["final_recall"]
        metrics["global_score_sort"][f"final_precision_at_{suffix}"] = global_values["final_precision"]
        metrics["upstream_frame_order"][f"ap{suffix}"] = upstream_values["ap"]
        metrics["upstream_frame_order"][f"final_recall_at_{suffix}"] = upstream_values["final_recall"]
        metrics["upstream_frame_order"][f"final_precision_at_{suffix}"] = upstream_values["final_precision"]
        metrics[f"counts_iou{suffix}"] = {
            "tp": global_values["tp"],
            "fp": global_values["fp"],
            "gt": global_values["gt"],
        }

    runner_metrics = summary["metrics"]
    metric_differences: dict[str, float] = {}
    for ordering in ("global_score_sort", "upstream_frame_order"):
        for threshold in THRESHOLDS:
            suffix = int(threshold * 100)
            for key in (f"ap{suffix}", f"final_recall_at_{suffix}", f"final_precision_at_{suffix}"):
                metric_differences[f"{ordering}.{key}"] = metrics[ordering][key] - float(runner_metrics[ordering][key])
    checks.update(
        all_raw_hashes_exact=not raw_hash_failures,
        raw_shapes_valid=not shape_failures,
        all_arrays_finite=not finite_failures,
        frame_counts_independently_match=not count_failures,
        delay_and_current_ego_semantics=not delay_failures,
        communication_statistic_explicitly_na=not communication_non_na,
        prediction_digest_exact=prediction_digest.hexdigest() == summary["prediction_content_digest"],
        gt_digest_exact=gt_digest.hexdigest() == summary["gt_content_digest"],
        all_metrics_independently_match=all(abs(value) < 1e-9 for value in metric_differences.values()),
    )
    return {
        "schema": "traf_v2xseq_adapted_fixed60_raw_independent_review_v1",
        "reviewed_at": now(),
        "status": "passed" if all(checks.values()) else "failed",
        "arm": str(arm),
        "delay_ms": delay,
        "checks": checks,
        "metrics": metrics,
        "metric_differences": metric_differences,
        "prediction_content_digest": prediction_digest.hexdigest(),
        "gt_content_digest": gt_digest.hexdigest(),
        "failures": {
            "raw_hash": raw_hash_failures[:20],
            "shape": shape_failures[:20],
            "finite": finite_failures[:20],
            "counts": count_failures[:20],
            "delay": delay_failures[:20],
            "communication_non_na": communication_non_na[:20],
        },
        "claim_boundary": "Independent CPU recomputation from all saved raw arrays on exposed public val21; no model forward and no hidden-test claim.",
    }


def review_aggregate(root: Path) -> dict[str, Any]:
    reviews: dict[int, dict[str, Any]] = {}
    for delay in EXPECTED_DELAYS:
        path = root / f"delay_{delay:03d}ms" / "independent_review.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            reviews[delay] = value
    checks = {
        "all_four_arms_present": sorted(reviews) == list(EXPECTED_DELAYS),
        "all_arm_reviews_passed": len(reviews) == 4 and all(value.get("status") == "passed" for value in reviews.values()),
        "shared_current_gt_across_delays": len(reviews) == 4 and len({value.get("gt_content_digest") for value in reviews.values()}) == 1,
        "checkpoint_identity_shared": False,
    }
    checkpoint_hashes = set()
    if len(reviews) == 4:
        for delay in EXPECTED_DELAYS:
            summary = json.loads((root / f"delay_{delay:03d}ms" / "summary.json").read_text(encoding="utf-8"))
            checkpoint_hashes.add(summary.get("checkpoint_sha256"))
    checks["checkpoint_identity_shared"] = len(checkpoint_hashes) == 1 and None not in checkpoint_hashes
    return {
        "schema": "traf_v2xseq_adapted_fixed60_four_delay_aggregate_review_v1",
        "reviewed_at": now(),
        "status": "passed" if all(checks.values()) else "failed",
        "root": str(root),
        "checks": checks,
        "arms": {
            str(delay): {
                "status": value.get("status"),
                "metrics": value.get("metrics"),
                "prediction_content_digest": value.get("prediction_content_digest"),
                "gt_content_digest": value.get("gt_content_digest"),
            }
            for delay, value in sorted(reviews.items())
        },
        "claim_boundary": "Four-delay adapted same-loader comparison on exposed public validation; fixed epoch60 and no AP checkpoint selection.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arm-dir")
    group.add_argument("--aggregate-root")
    group.add_argument("--self-test", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.arm_dir:
        arm = Path(args.arm_dir).resolve()
        result = review_arm(arm)
        output = Path(args.output).resolve() if args.output else arm / "independent_review.json"
        atomic_json(output, result)
    else:
        root = Path(args.aggregate_root).resolve()
        result = review_aggregate(root)
        output = Path(args.output).resolve() if args.output else root / "aggregate_review.json"
        atomic_json(output, result)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
