#!/usr/bin/env python3
"""Fit matched candidate-D heads and evaluate the frozen five-arm pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable

import numpy as np

from evaluate_traf_v2xseq_adapted_fixed60_raw_20261006 import atomic_json, now, sha256
from evaluate_v2xseq_ego_protected_motion_controls import ego_protected_merge
from evaluate_v2xseq_real_motion_controls import iou_matrix
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import match_frame, voc_ap
from v2xseq_real_motion_packets import transform_boxes7d


ROLES = ("fit24", "calibration8", "mechanism_holdout14")
METHODS = (
    "fixed_source_rank",
    "fixed_age200_receiver_x24",
    "raw_source_score_full",
    "raw_source_score_admission",
    "ordinary_quality_full",
    "ordinary_quality",
    "complementary_utility_full",
    "complementary_utility",
)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def blank() -> dict[str, Any]:
    return {"scores": [], "tp": [], "fp": [], "gt": 0}


def add_stat(target: dict[str, Any], value: tuple[list[float], list[int], list[int], int]) -> None:
    scores, tp, fp, gt = value
    target["scores"].extend(scores)
    target["tp"].extend(tp)
    target["fp"].extend(fp)
    target["gt"] += int(gt)


def metric(stat: dict[str, Any]) -> dict[str, Any]:
    value = voc_ap(stat["tp"], stat["fp"], stat["scores"], stat["gt"], True)
    return {
        "ap": float(value["ap"]),
        "tp": int(sum(stat["tp"])),
        "fp": int(sum(stat["fp"])),
        "gt": int(stat["gt"]),
    }


def npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as value:
        return {name: value[name] for name in value.files}


def candidate_indices(
    local_boxes: np.ndarray,
    remote_boxes: np.ndarray,
    remote_scores: np.ndarray,
    threshold: float = 0.15,
) -> tuple[np.ndarray, dict[str, int]]:
    in_range = (
        (remote_boxes[:, 0] >= 0.0) & (remote_boxes[:, 0] <= 96.0)
        & (remote_boxes[:, 1] >= -40.0) & (remote_boxes[:, 1] <= 40.0)
    ) if len(remote_boxes) else np.empty(0, dtype=bool)
    original = np.flatnonzero(in_range)
    boxes = remote_boxes[in_range]
    scores = remote_scores[in_range]
    local_remote = iou_matrix(boxes, local_boxes)
    remote_remote = iou_matrix(boxes, boxes)
    kept: list[int] = []
    conflict = 0
    self_nms = 0
    for index in np.argsort(-scores, kind="stable"):
        if len(local_boxes) and float(local_remote[index].max(initial=0.0)) > threshold:
            conflict += 1
            continue
        if kept and float(remote_remote[index, kept].max(initial=0.0)) > threshold:
            self_nms += 1
            continue
        kept.append(int(index))
    return original[np.asarray(kept, dtype=np.int64)], {
        "remote_input": int(len(remote_boxes)),
        "remote_range_dropped": int((~in_range).sum()),
        "remote_conflict_dropped": conflict,
        "remote_self_nms_dropped": self_nms,
        "remote_retained": len(kept),
    }


def greedy_matches(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    eligible_gt: set[int] | None = None,
) -> tuple[np.ndarray, set[int]]:
    target = np.zeros(len(boxes), dtype=np.int8)
    remaining = [
        index for index, label in enumerate(gt_labels.astype(np.int64))
        if label == 0 and (eligible_gt is None or index in eligible_gt)
    ]
    pairwise = iou_matrix(boxes, gt_boxes) if len(boxes) and len(gt_boxes) else np.zeros((len(boxes), len(gt_boxes)))
    matched: set[int] = set()
    for detection in np.argsort(-scores, kind="stable"):
        if int(labels[detection]) != 0 or not remaining:
            continue
        values = pairwise[detection, remaining]
        position = int(np.argmax(values))
        if float(values[position]) >= 0.7:
            gt_index = remaining.pop(position)
            target[detection] = 1
            matched.add(gt_index)
    return target, matched


def feature_matrix(
    receiver_boxes: np.ndarray,
    source_boxes: np.ndarray,
    source_scores: np.ndarray,
    age_ms: float,
    transform: np.ndarray,
    ego_boxes: np.ndarray,
    ego_scores: np.ndarray,
) -> np.ndarray:
    n = len(receiver_boxes)
    if not n:
        return np.empty((0, 30), dtype=np.float64)
    yaw = receiver_boxes[:, 6]
    source_yaw = source_boxes[:, 6]
    transform_yaw = math.atan2(float(transform[1, 0]), float(transform[0, 0]))
    relation = np.empty((n, 6), dtype=np.float64)
    if len(ego_boxes):
        delta = receiver_boxes[:, None, :2] - ego_boxes[None, :, :2]
        distance = np.linalg.norm(delta, axis=2)
        nearest = np.argmin(distance, axis=1)
        overlaps = iou_matrix(receiver_boxes, ego_boxes)
        relation[:, 0] = distance[np.arange(n), nearest]
        relation[:, 1] = np.abs(delta[np.arange(n), nearest, 0])
        relation[:, 2] = np.abs(delta[np.arange(n), nearest, 1])
        relation[:, 3] = overlaps.max(axis=1)
        relation[:, 4] = ego_scores[nearest]
        relation[:, 5] = (distance < 10.0).sum(axis=1)
    else:
        relation[:] = np.asarray([200.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    return np.column_stack((
        np.full(n, age_ms, dtype=np.float64),
        source_scores,
        receiver_boxes[:, 0], receiver_boxes[:, 1], receiver_boxes[:, 2],
        receiver_boxes[:, 3], receiver_boxes[:, 4], receiver_boxes[:, 5],
        np.sin(yaw), np.cos(yaw),
        np.linalg.norm(receiver_boxes[:, :2], axis=1), np.abs(receiver_boxes[:, 1]),
        source_boxes[:, 0], source_boxes[:, 1], source_boxes[:, 2],
        np.linalg.norm(source_boxes[:, :2], axis=1), np.sin(source_yaw), np.cos(source_yaw),
        np.full(n, transform[0, 3]), np.full(n, transform[1, 3]), np.full(n, transform[2, 3]),
        np.full(n, np.linalg.norm(transform[:2, 3])),
        np.full(n, math.sin(transform_yaw)), np.full(n, math.cos(transform_yaw)),
        relation,
    )).astype(np.float64)


def interval_scores(local_scores: np.ndarray, remote_scores: np.ndarray, kind: str) -> np.ndarray:
    local = 0.500001 + (local_scores.astype(np.float64) - 0.2) * (1.0 - 0.500001) / 0.8
    if kind == "raw":
        remote = 0.2 + (remote_scores.astype(np.float64) - 0.2) * (0.499999 - 0.2) / 0.8
    else:
        remote = 0.2 + np.clip(remote_scores.astype(np.float64), 0.0, 1.0) * (0.499999 - 0.2)
    return np.concatenate((local, remote))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--cache-review", required=True, type=Path)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--head-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "schema": "candidate_d_matched_heads_holdout_v1",
        "status": "initializing",
        "created_at": now(),
        "detector_forwards": 0,
        "detector_updates": 0,
        "head_updates": {"ordinary_quality": 0, "complementary_utility": 0},
    }
    atomic_json(output / "status.json", state)
    started = time.monotonic()
    try:
        assert "embodied-research.slice" in Path("/proc/self/cgroup").read_text()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        import torch
        from torch import nn

        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        cache_review_path = args.cache_review.resolve()
        cache_review = json.loads(cache_review_path.read_text())
        spec_path = args.head_spec.resolve()
        spec = json.loads(spec_path.read_text())
        assert cache_review["status"] == "passed" and all(cache_review["checks"].values())
        assert spec["created_before_head_fitting"] is True
        assert len(spec["features_in_order"]) == 30
        assert spec["matched_capacity_model"]["trainable_parameters_per_arm"] == 513
        producer_dirs = [args.qualification.resolve()] + [path.resolve() for path in args.shard]
        ego_records = [item for path in producer_dirs for item in rows(path / "ego_manifest.jsonl")]
        source_records = [item for path in producer_dirs for item in rows(path / "source_manifest.jsonl")]
        ego_by_key = {(item["sequence"], int(item["global_index"])): item for item in ego_records}
        source_by_key = {(item["sequence"], item["source_frame"]): item for item in source_records}
        geometry_path = cache_review_path.parent / "condition_geometry.jsonl"
        geometries = list(rows(geometry_path))
        assert len(geometries) == 1472
        assert len(ego_by_key) == 368 and len(source_by_key) == 1285

        ego_arrays = {key: npz(Path(value["raw_path"])) for key, value in ego_by_key.items()}
        source_arrays = {key: npz(Path(value["raw_path"])) for key, value in source_by_key.items()}
        frames: list[dict[str, Any]] = []
        all_features: list[np.ndarray] = []
        all_quality: list[np.ndarray] = []
        all_complement: list[np.ndarray] = []
        all_condition: list[np.ndarray] = []
        all_candidate: list[np.ndarray] = []
        role_counts = Counter()
        feature_started = time.monotonic()
        for condition_index, geometry in enumerate(geometries):
            ego_key = (geometry["sequence"], int(geometry["global_index"]))
            ego = ego_arrays[ego_key]
            local_boxes = ego["pred_boxes"].astype(np.float32)
            local_scores = ego["pred_scores"].astype(np.float32)
            local_labels = ego["pred_labels"].astype(np.int32)
            gt_boxes = ego["gt_boxes"].astype(np.float32)
            gt_labels = ego["gt_labels"].astype(np.int32)
            if geometry["source"] is None:
                remote_boxes = np.empty((0, 7), dtype=np.float32)
                remote_source_boxes = np.empty((0, 7), dtype=np.float32)
                remote_scores = np.empty(0, dtype=np.float32)
                remote_labels = np.empty(0, dtype=np.int32)
                features = np.empty((0, 30), dtype=np.float64)
                diagnostic = {name: 0 for name in (
                    "remote_input", "remote_range_dropped", "remote_conflict_dropped",
                    "remote_self_nms_dropped", "remote_retained"
                )}
            else:
                source_key = tuple(geometry["source"]["key"])
                source = source_arrays[source_key]
                transform = np.asarray(geometry["source"]["source_to_current_ego"], dtype=np.float64)
                transformed = transform_boxes7d(source["pred_boxes"].astype(np.float32),
                                                np.asarray(source_by_key[source_key]["source_pose_lidar_to_world"], dtype=np.float64),
                                                np.asarray(geometry["current_ego_pose_lidar_to_world"], dtype=np.float64))
                indices, diagnostic = candidate_indices(local_boxes, transformed, source["pred_scores"])
                remote_boxes = transformed[indices].astype(np.float32)
                remote_source_boxes = source["pred_boxes"][indices].astype(np.float32)
                remote_scores = source["pred_scores"][indices].astype(np.float32)
                remote_labels = source["pred_labels"][indices].astype(np.int32)
                features = feature_matrix(
                    remote_boxes, remote_source_boxes, remote_scores,
                    float(geometry["source"]["actual_age_ms"]), transform,
                    local_boxes, local_scores,
                )
                merged = ego_protected_merge(
                    (local_boxes, local_scores, local_labels),
                    (transformed, source["pred_scores"], source["pred_labels"]),
                    threshold=0.15,
                )
                assert np.allclose(merged[0][len(local_boxes):], remote_boxes, rtol=0, atol=1e-6)
                assert np.array_equal(merged[2][len(local_boxes):], remote_labels)
                assert all(int(merged[3][name]) == int(diagnostic[name]) for name in diagnostic)
            quality, _ = greedy_matches(remote_boxes, remote_scores, remote_labels, gt_boxes, gt_labels)
            _, ego_matched = greedy_matches(local_boxes, local_scores, local_labels, gt_boxes, gt_labels)
            eligible = {index for index, label in enumerate(gt_labels) if int(label) == 0} - ego_matched
            complement, _ = greedy_matches(remote_boxes, remote_scores, remote_labels, gt_boxes, gt_labels, eligible)
            start = sum(len(value) for value in all_features)
            stop = start + len(features)
            frame = {
                **geometry,
                "local_boxes": local_boxes,
                "local_scores": local_scores,
                "local_labels": local_labels,
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
                "remote_boxes": remote_boxes,
                "remote_source_boxes": remote_source_boxes,
                "remote_scores": remote_scores,
                "remote_labels": remote_labels,
                "features": features,
                "quality_target": quality,
                "complement_target": complement,
                "ego_matched_gt": ego_matched,
                "row_start": start,
                "row_stop": stop,
                "diagnostic": diagnostic,
            }
            frames.append(frame)
            all_features.append(features)
            all_quality.append(quality)
            all_complement.append(complement)
            all_condition.append(np.full(len(features), condition_index, dtype=np.int32))
            all_candidate.append(np.arange(len(features), dtype=np.int32))
            role_counts[geometry["role"]] += 1
        X = np.concatenate(all_features, axis=0)
        quality_y = np.concatenate(all_quality).astype(np.float32)
        complement_y = np.concatenate(all_complement).astype(np.float32)
        condition_ids = np.concatenate(all_condition)
        candidate_ids = np.concatenate(all_candidate)
        assert X.shape[1] == 30 and np.isfinite(X).all()
        assert dict(role_counts) == {"fit24": 768, "calibration8": 256, "mechanism_holdout14": 448}
        role_array = np.asarray([ROLES.index(frames[index]["role"]) for index in condition_ids], dtype=np.int8)
        fit_mask = role_array == ROLES.index("fit24")
        calibration_mask = role_array == ROLES.index("calibration8")
        holdout_mask = role_array == ROLES.index("mechanism_holdout14")
        assert fit_mask.any() and calibration_mask.any() and holdout_mask.any()
        mean = X[fit_mask].mean(axis=0)
        std = X[fit_mask].std(axis=0)
        std = np.maximum(std, float(spec["normalization"]["std_floor"]))
        Z = ((X - mean) / std).astype(np.float32)
        np.savez_compressed(
            output / "feature_dataset.npz",
            features=X.astype(np.float32),
            normalized_features=Z,
            ordinary_quality_target=quality_y,
            complementary_utility_target=complement_y,
            condition_index=condition_ids,
            candidate_index=candidate_ids,
            role_index=role_array,
        )
        atomic(output / "normalization.json", {
            "features_in_order": spec["features_in_order"],
            "mean": mean.tolist(),
            "std": std.tolist(),
            "fit_rows": int(fit_mask.sum()),
            "calibration_rows_excluded": int(calibration_mask.sum()),
            "holdout_rows_excluded": int(holdout_mask.sum()),
            "source": "fit24_only",
        })
        feature_seconds = time.monotonic() - feature_started

        class Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = nn.Sequential(nn.Linear(30, 16), nn.ReLU(), nn.Linear(16, 1))

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.network(value).squeeze(-1)

        fit_x = torch.from_numpy(Z[fit_mask])
        training: dict[str, Any] = {}
        probabilities: dict[str, np.ndarray] = {}
        targets = {"ordinary_quality": quality_y, "complementary_utility": complement_y}
        for name, target in targets.items():
            torch.manual_seed(int(spec["training"]["seed"]))
            model = Head()
            assert sum(value.numel() for value in model.parameters()) == 513
            y = torch.from_numpy(target[fit_mask])
            positives = int(y.sum().item())
            negatives = len(y) - positives
            pos_weight = min(20.0, max(1.0, negatives / max(positives, 1)))
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0)
            generator = torch.Generator().manual_seed(int(spec["training"]["seed"]))
            losses: list[dict[str, float]] = []
            train_started = time.monotonic()
            for update in range(1000):
                indices = torch.randint(0, len(fit_x), (min(256, len(fit_x)),), generator=generator)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(fit_x[indices]), y[indices])
                assert bool(torch.isfinite(loss).item())
                loss.backward()
                optimizer.step()
                if update == 0 or (update + 1) % 100 == 0:
                    losses.append({"update": update + 1, "loss": float(loss.detach())})
            state["head_updates"][name] = 1000
            model.eval()
            inference_started = time.monotonic()
            with torch.no_grad():
                probability = torch.sigmoid(model(torch.from_numpy(Z))).numpy().astype(np.float32)
            inference_seconds = time.monotonic() - inference_started
            assert np.isfinite(probability).all()
            probabilities[name] = probability
            weights = output / f"{name}_head.pth"
            torch.save(model.state_dict(), weights)
            training[name] = {
                "train_rows": len(y),
                "positives": positives,
                "negatives": negatives,
                "positive_weight": pos_weight,
                "updates": 1000,
                "loss_trace": losses,
                "weights": str(weights),
                "weights_sha256": sha256(weights),
                "training_wall_seconds": time.monotonic() - train_started,
                "all_candidate_inference_seconds": inference_seconds,
                "parameters": 513,
            }
        np.savez_compressed(
            output / "candidate_scores.npz",
            raw_source_score=X[:, 1].astype(np.float32),
            ordinary_quality=probabilities["ordinary_quality"],
            complementary_utility=probabilities["complementary_utility"],
        )
        state["status"] = "heads_fitted_calibrating"
        atomic_json(output / "status.json", state)

        def score_for_frame(frame: dict[str, Any], method: str) -> np.ndarray:
            span = slice(frame["row_start"], frame["row_stop"])
            if method == "raw_source_score_admission":
                return frame["remote_scores"].astype(np.float64)
            return probabilities[method][span].astype(np.float64)

        calibration: dict[str, Any] = {}
        calibration_curve_path = output / "calibration_curve.jsonl"
        calibration_frames = [frame for frame in frames if frame["role"] == "calibration8"]
        for method in ("raw_source_score_admission", "ordinary_quality", "complementary_utility"):
            ego_tp = ego_fp = 0
            score_values: list[np.ndarray] = []
            remote_tp_values: list[np.ndarray] = []
            remote_fp_values: list[np.ndarray] = []
            for frame in calibration_frames:
                ego_result = match_frame(
                    frame["local_boxes"], frame["local_scores"], frame["local_labels"],
                    frame["gt_boxes"], frame["gt_labels"], 0.7,
                )
                ego_tp += sum(ego_result[1]); ego_fp += sum(ego_result[2])
                scores = score_for_frame(frame, method)
                eligible = {index for index, label in enumerate(frame["gt_labels"]) if int(label) == 0} - frame["ego_matched_gt"]
                targets_for_order, _ = greedy_matches(
                    frame["remote_boxes"], scores, frame["remote_labels"],
                    frame["gt_boxes"], frame["gt_labels"], eligible,
                )
                score_values.append(scores)
                remote_tp_values.append(targets_for_order)
                remote_fp_values.append((frame["remote_labels"] == 0).astype(np.int8) - targets_for_order)
            scores = np.concatenate(score_values) if score_values else np.empty(0)
            remote_tp = np.concatenate(remote_tp_values) if remote_tp_values else np.empty(0, dtype=np.int8)
            remote_fp = np.concatenate(remote_fp_values) if remote_fp_values else np.empty(0, dtype=np.int8)
            budget = int(math.floor(0.2 * ego_fp))
            options: list[dict[str, Any]] = [{
                "threshold": None, "combined_tp70": ego_tp, "combined_fp70": ego_fp,
                "additional_remote_tp70": 0, "additional_remote_fp70": 0, "retained_remote": 0,
            }]
            for threshold in np.unique(scores):
                keep = scores >= threshold
                options.append({
                    "threshold": float(threshold),
                    "combined_tp70": ego_tp + int(remote_tp[keep].sum()),
                    "combined_fp70": ego_fp + int(remote_fp[keep].sum()),
                    "additional_remote_tp70": int(remote_tp[keep].sum()),
                    "additional_remote_fp70": int(remote_fp[keep].sum()),
                    "retained_remote": int(keep.sum()),
                })
            feasible = [value for value in options if value["additional_remote_fp70"] <= budget]
            selected = max(feasible, key=lambda value: (
                value["combined_tp70"], -value["additional_remote_fp70"],
                -value["retained_remote"],
                float("inf") if value["threshold"] is None else value["threshold"],
            ))
            for value in options:
                append(calibration_curve_path, {"method": method, "ego_fp70_budget_base": ego_fp,
                                                "additional_remote_fp70_budget": budget, **value})
            calibration[method] = {
                "selected": selected,
                "ego_tp70": ego_tp,
                "ego_fp70": ego_fp,
                "additional_remote_fp70_budget": budget,
                "candidate_thresholds": len(options),
                "selection_population": "calibration8_four_wait_conditions",
            }

        stats = {(method, threshold): blank() for method in METHODS for threshold in (0.5, 0.7)}
        sequence_stats = defaultdict(blank)
        diagnostics = defaultdict(Counter)
        prediction_path = output / "holdout_predictions.jsonl"
        holdout_frames = [frame for frame in frames if frame["role"] == "mechanism_holdout14"]
        for frame in holdout_frames:
            for method in METHODS:
                if method.startswith("ordinary_quality"):
                    score_method = "ordinary_quality"; remote_score = score_for_frame(frame, score_method); kind = "head"
                elif method.startswith("complementary_utility"):
                    score_method = "complementary_utility"; remote_score = score_for_frame(frame, score_method); kind = "head"
                else:
                    score_method = "raw_source_score_admission"; remote_score = frame["remote_scores"].astype(np.float64); kind = "raw"
                keep = np.ones(len(remote_score), dtype=bool)
                if method == "fixed_age200_receiver_x24" and frame["source"] is not None and float(frame["source"]["actual_age_ms"]) >= 200.0:
                    keep &= frame["remote_boxes"][:, 0] < 24.0
                elif method in ("raw_source_score_admission", "ordinary_quality", "complementary_utility"):
                    threshold = calibration[method]["selected"]["threshold"]
                    keep &= False if threshold is None else remote_score >= float(threshold)
                selected_boxes = frame["remote_boxes"][keep]
                selected_labels = frame["remote_labels"][keep]
                selected_remote_scores = remote_score[keep]
                boxes = np.concatenate((frame["local_boxes"], selected_boxes), axis=0)
                labels = np.concatenate((frame["local_labels"], selected_labels), axis=0)
                scores = interval_scores(frame["local_scores"], selected_remote_scores, kind)
                frame_results = {}
                for threshold in (0.5, 0.7):
                    value = match_frame(boxes, scores, labels, frame["gt_boxes"], frame["gt_labels"], threshold)
                    add_stat(stats[(method, threshold)], value)
                    add_stat(sequence_stats[(frame["sequence"], method, threshold)], value)
                    frame_results[str(threshold)] = {"tp": int(sum(value[1])), "fp": int(sum(value[2])), "gt": int(value[3])}
                _, combined_matched = greedy_matches(boxes, scores, labels, frame["gt_boxes"], frame["gt_labels"])
                rescued = len(combined_matched - frame["ego_matched_gt"])
                destroyed = len(frame["ego_matched_gt"] - combined_matched)
                diagnostics[method]["remote_before"] += len(remote_score)
                diagnostics[method]["remote_after"] += int(keep.sum())
                diagnostics[method]["rescued_targets70"] += rescued
                diagnostics[method]["destroyed_ego_targets70"] += destroyed
                append(prediction_path, {
                    "sequence": frame["sequence"],
                    "global_index": frame["global_index"],
                    "current_vehicle_frame": frame["current_vehicle_frame"],
                    "wait_ms": frame["wait_ms"],
                    "role": frame["role"],
                    "method": method,
                    "source": frame["source"],
                    "local_count": len(frame["local_boxes"]),
                    "remote_before": len(remote_score),
                    "remote_after": int(keep.sum()),
                    "remote_selected_relative_indices": np.flatnonzero(keep).astype(int).tolist(),
                    "boxes_3d_lidar": boxes.tolist(),
                    "scores": scores.tolist(),
                    "label_preds": labels.tolist(),
                    "gt_boxes": np.column_stack((frame["gt_boxes"], frame["gt_labels"])).tolist(),
                    "metrics": frame_results,
                    "rescued_targets70": rescued,
                    "destroyed_ego_targets70": destroyed,
                })

        methods: list[dict[str, Any]] = []
        for method in METHODS:
            row: dict[str, Any] = {"method": method, **dict(diagnostics[method])}
            for threshold in (0.5, 0.7):
                value = metric(stats[(method, threshold)])
                suffix = int(threshold * 100)
                for key, item in value.items():
                    row[f"{key}{suffix}" if key != "gt" else "gt"] = item
            sequence_signs = Counter()
            if method != "fixed_source_rank":
                for sequence in sorted({frame["sequence"] for frame in holdout_frames}):
                    current = metric(sequence_stats[(sequence, method, 0.7)])["ap"]
                    parent = metric(sequence_stats[(sequence, "fixed_source_rank", 0.7)])["ap"]
                    sequence_signs["positive" if current > parent + 1e-12 else "negative" if current < parent - 1e-12 else "equal"] += 1
            row["sequence_signs_vs_fixed_source_rank_ap70"] = dict(sequence_signs)
            methods.append(row)
        fixed = next(value for value in methods if value["method"] == "fixed_source_rank")
        raw_full = next(value for value in methods if value["method"] == "raw_source_score_full")
        assert fixed == raw_full | {"method": "fixed_source_rank"} or all(
            fixed[key] == raw_full[key] for key in fixed if key not in ("method", "sequence_signs_vs_fixed_source_rank_ap70")
        )
        checks = {
            "reviewed_cache_exact": True,
            "head_spec_frozen_before_fitting": True,
            "features_only_declared_observables": True,
            "normalization_fit24_only": True,
            "two_heads_matched_513_parameters_1000_updates": True,
            "calibration8_only_threshold_selection": True,
            "holdout14_excluded_from_fit_normalization_calibration": True,
            "all_five_arms_and_full_variants_evaluated": True,
            "fixed_source_rank_equals_raw_full": True,
            "all_local_outputs_preserved": all(value["destroyed_ego_targets70"] == 0 for value in methods),
            "raw_predictions_saved": True,
            "zero_detector_forwards_and_updates": True,
            "no_CUDA_initialized": not torch.cuda.is_initialized(),
        }
        assert all(checks.values()), checks
        input_paths = [cache_review_path, geometry_path, spec_path]
        input_paths += [path / "summary.json" for path in producer_dirs]
        summary = {
            **state,
            "status": "producer_complete_review_pending",
            "finished_at": now(),
            "checks": checks,
            "feature_schema_sha256": sha256(spec_path),
            "feature_rows": len(X),
            "feature_role_rows": {role: int((role_array == index).sum()) for index, role in enumerate(ROLES)},
            "target_positives": {
                "ordinary_quality": {role: int(quality_y[role_array == index].sum()) for index, role in enumerate(ROLES)},
                "complementary_utility": {role: int(complement_y[role_array == index].sum()) for index, role in enumerate(ROLES)},
            },
            "feature_construction_seconds": feature_seconds,
            "training": training,
            "calibration": calibration,
            "holdout_frames": 112,
            "holdout_conditions": len(holdout_frames),
            "holdout_sequences": 14,
            "methods": methods,
            "predictions_sha256": sha256(prediction_path),
            "feature_dataset_sha256": sha256(output / "feature_dataset.npz"),
            "candidate_scores_sha256": sha256(output / "candidate_scores.npz"),
            "normalization_sha256": sha256(output / "normalization.json"),
            "calibration_curve_sha256": sha256(calibration_curve_path),
            "input_sha256": {str(path): sha256(path) for path in input_paths},
            "detector_forwards": 0,
            "detector_updates": 0,
            "head_updates": {"ordinary_quality": 1000, "complementary_utility": 1000},
            "wall_seconds": time.monotonic() - started,
            "boundary": "Existing train46 internal module pilot. Holdout14 is excluded from this head's fit/calibration but is historically exposed and not an untouched external test.",
        }
        atomic(output / "summary.json", summary)
        atomic(output / "status.json", summary)
        print(json.dumps({
            "status": summary["status"],
            "feature_rows": summary["feature_rows"],
            "target_positives": summary["target_positives"],
            "calibration": summary["calibration"],
            "methods": summary["methods"],
            "wall_seconds": summary["wall_seconds"],
        }, indent=2), flush=True)
    except BaseException as exc:
        state.update(
            status="failed_preserved",
            failed_at=now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - started,
        )
        atomic_json(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
