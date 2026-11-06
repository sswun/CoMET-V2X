#!/usr/bin/env python3
"""Readout attribution on the frozen TraF causal shared-offset geometry.

The detector and candidate identities are frozen.  This program rematerializes
the accepted one-CV-candidate-per-source population after the reviewed causal
KF translation, recomputes all corrected-coordinate geometry features and
targets, fits one small direct-utility head on fit24, freezes source maps and
FP-budget workpoints on calibration8, and evaluates hold14 plus val21.

Ground truth is used only for fit/cal targets and evaluation.  The causal KF,
corrected-coordinate features, and tolerance scores do not consume labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import resource
import shutil
import time
from typing import Any

import numpy as np
from scipy.special import erf

from audit_common_offset_peer_v1_20261026 import pairs, views
from e1_receiver_value_core_v1_20261020 import (
    atomic, greedy, iou32, sha256, unique_ego_scores,
)
from object_history_selector_core_v1_20261021 import FEATURE_NAMES, PackedDataset
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap
from run_traf_receiver_local_source_monotone_calibration_v1_20261021 import (
    apply_map, fit_positive_logistic,
)


WAITS = (0, 100, 200, 400)
BETAS = (1.0, 1.2, 1.4)
FULL_METHODS = (
    "ego",
    "all_ego_raw",
    "protected_raw",
    "protected_raw_drop_anchors",
    "protected_source_u",
    "protected_source_u_drop_anchors",
    "protected_direct_u",
    "protected_task_circle",
    "protected_task_numeric",
)
WORKPOINT_BASES = (
    "protected_raw",
    "protected_source_u",
    "protected_direct_u",
    "protected_task_circle",
    "protected_task_numeric",
)
MODEL_SEED = 20261026
MODEL_UPDATES = 2000
MODEL_BATCH_PER_SEQUENCE = 16
IOU_THRESHOLD = 0.7
QUADRATURE_POINTS = 48


def checked_dataset(path: Path) -> tuple[PackedDataset, dict[str, Any]]:
    summary = json.loads((path / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert sha256(path / "pools.npz") == summary["pools_sha256"]
    assert sha256(path / "conditions.jsonl") == summary["conditions_sha256"]
    return PackedDataset(path), summary


def cv_identity(pool: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ego rows, stale/source rows and the actual chosen CV row IDs."""
    kind = pool["hypothesis_type"]
    ego = np.flatnonzero(kind == 0)
    stale = np.flatnonzero(kind == 1)
    propagated = np.flatnonzero(kind == 2)
    lookup = {int(pool["source_detection_indices"][j]): int(j) for j in propagated}
    chosen = np.asarray([
        lookup.get(int(pool["source_detection_indices"][j]), int(j)) for j in stale
    ], np.int64)
    return ego, stale, chosen


def correction_step(
    cv: np.ndarray,
    protected: np.ndarray,
    state: tuple[np.ndarray, float],
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Exact reviewed causal_kf step; no GT or future input."""
    remote_id, ego_id = pairs(cv, protected, 2.5, True)
    residual = protected[ego_id, :2].astype(np.float64) - cv[remote_id, :2]
    mean, variance = state
    prior = float(variance) + 0.15 ** 2
    if len(residual):
        gain = prior / (prior + 0.6 ** 2 / len(residual))
        mean = mean + gain * (residual.mean(axis=0) - mean)
        variance = (1.0 - gain) * prior
    else:
        variance = prior
    return mean.astype(np.float64), float(variance), remote_id, ego_id, residual


def update_corrected_features(
    base: np.ndarray,
    corrected: np.ndarray,
    protected_boxes: np.ndarray,
    protected_scores: np.ndarray,
) -> np.ndarray:
    """Rebuild every receiver-coordinate/query-dependent feature in 39D."""
    result = base.astype(np.float32).copy()
    if not len(result):
        return result
    result[:, 2:5] = corrected[:, 0:3]
    result[:, 5:8] = corrected[:, 3:6]
    result[:, 8] = np.sin(corrected[:, 6])
    result[:, 9] = np.cos(corrected[:, 6])
    result[:, 10] = np.linalg.norm(corrected[:, :2], axis=1)
    result[:, 11] = np.abs(corrected[:, 1])
    if len(protected_boxes):
        delta = corrected[:, None, :2] - protected_boxes[None, :, :2]
        distance = np.linalg.norm(delta, axis=2)
        nearest = distance.argmin(axis=1)
        overlap = iou32(corrected, protected_boxes)
        rows = np.arange(len(corrected))
        result[:, 24] = distance[rows, nearest]
        result[:, 25] = np.abs(delta[rows, nearest, 0])
        result[:, 26] = np.abs(delta[rows, nearest, 1])
        result[:, 27] = overlap.max(axis=1)
        result[:, 28] = protected_scores[nearest]
        result[:, 29] = (distance < 10.0).sum(axis=1)
    else:
        result[:, 24:30] = np.asarray([200.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    return result


def tolerance_probabilities(
    boxes: np.ndarray,
    variance: float,
    threshold: float = IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-area-circle approximation and numerical true-region integral.

    Both calculations use the same idealized equal-size, equal-yaw rectangle
    model and isotropic zero-mean center uncertainty.  The circle result is not
    called exact.  The numerical result integrates the non-circular IoU region.
    """
    if not len(boxes):
        empty = np.empty(0, np.float64)
        return empty, empty, empty
    length = np.maximum(boxes[:, 5].astype(np.float64), 1e-6)
    width = np.maximum(boxes[:, 4].astype(np.float64), 1e-6)
    critical = 2.0 * threshold * length * width / (1.0 + threshold)
    xmax = np.maximum(length - critical / width, 0.0)
    quarter_area = width * xmax + critical * np.log(
        np.maximum(length - xmax, 1e-12) / length
    )
    region_area = np.maximum(4.0 * quarter_area, 0.0)
    sigma2 = max(float(variance), 1e-12)
    circle = 1.0 - np.exp(-region_area / (2.0 * math.pi * sigma2))

    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_POINTS)
    # Symmetry: integrate x in [0, xmax], then double it.
    x = 0.5 * xmax[:, None] * (nodes[None, :] + 1.0)
    ylimit = width[:, None] - critical[:, None] / np.maximum(
        length[:, None] - x, 1e-12
    )
    ylimit = np.maximum(ylimit, 0.0)
    density_x = np.exp(-0.5 * x * x / sigma2) / math.sqrt(2.0 * math.pi * sigma2)
    conditional_y = erf(ylimit / math.sqrt(2.0 * sigma2))
    numeric = xmax * np.sum(weights[None, :] * density_x * conditional_y, axis=1)
    numeric = np.clip(numeric, 0.0, 1.0)
    return circle.astype(np.float64), numeric.astype(np.float64), region_area


def materialize_role(
    dataset: PackedDataset,
    role: str,
    output: Path,
) -> dict[str, Any]:
    selected = [i for i, row in enumerate(dataset.meta) if row["role"] == role]
    selected.sort(key=lambda i: (
        dataset.meta[i]["sequence"], int(dataset.meta[i]["wait_ms"]),
        int(dataset.meta[i]["receiver_timestamp_us"]),
    ))
    states: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    previous_time: dict[tuple[str, int], int] = {}
    previous_yaw: dict[tuple[str, int], float] = {}
    previous_source: dict[tuple[str, int], str | None] = {}
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    condition_offsets = [0]
    gt_offsets = [0]
    iou_offsets = [0]
    trace = []
    sequence_values = []
    global_indices = []
    waits = []
    timestamps = []
    source_frames = []
    started = time.monotonic()
    for count, index in enumerate(selected):
        meta = dataset.meta[index]
        pool = dataset.pool(index)
        key = (str(meta["sequence"]), int(meta["wait_ms"]))
        timestamp = int(meta["receiver_timestamp_us"])
        assert timestamp > previous_time.get(key, -1)
        previous_time[key] = timestamp
        ego_id, stale_id, chosen_id = cv_identity(pool)
        ego_boxes = pool["boxes"][ego_id].copy()
        ego_scores = pool["raw_scores"][ego_id].astype(np.float64)
        protected_local = np.flatnonzero(pool["is_protected"][ego_id])
        protected_boxes = ego_boxes[protected_local]
        protected_scores = ego_scores[protected_local]
        cv_boxes = pool["boxes"][chosen_id].copy()
        mean, variance, anchor_remote, _, residual = correction_step(
            cv_boxes, protected_boxes, states.get(key, (np.zeros(2), 1.0))
        )
        states[key] = (mean.copy(), variance)
        corrected = cv_boxes.copy()
        corrected[:, :2] += mean.astype(np.float32)
        remote_feature = update_corrected_features(
            pool["features"][chosen_id], corrected, protected_boxes, protected_scores
        )
        boxes = np.concatenate((ego_boxes, corrected), axis=0)
        score = np.r_[ego_scores, pool["raw_scores"][stale_id]].astype(np.float64)
        source = np.r_[np.zeros(len(ego_id), np.int8), np.ones(len(stale_id), np.int8)]
        protected = np.r_[pool["is_protected"][ego_id], np.zeros(len(stale_id), np.int8)]
        anchor = np.zeros(len(boxes), np.int8)
        anchor[len(ego_id) + anchor_remote] = 1
        feature = np.concatenate((pool["features"][ego_id], remote_feature), axis=0)
        matrix = np.concatenate((pool["ious"][ego_id], iou32(corrected, pool["gt_boxes"])), axis=0)
        fixed = np.flatnonzero(protected)
        fixed_rank = 2.0 + unique_ego_scores(score[fixed], stable=True)
        fixed_order = fixed[np.argsort(-fixed_rank, kind="stable")]
        fixed_match = greedy(matrix, fixed_order, IOU_THRESHOLD)
        covered = {int(value) for value in fixed_match if value >= 0}
        uncovered = np.asarray([j for j in range(len(pool["gt_boxes"])) if j not in covered], np.int64)
        quality = (matrix.max(axis=1, initial=0.0) >= IOU_THRESHOLD).astype(np.int8)
        utility = (
            matrix[:, uncovered].max(axis=1, initial=0.0) >= IOU_THRESHOLD
        ).astype(np.int8)
        assert np.all(utility <= quality)
        circle_remote, numeric_remote, area_remote = tolerance_probabilities(corrected, variance)
        circle = np.r_[np.ones(len(ego_id)), circle_remote]
        numeric = np.r_[np.ones(len(ego_id)), numeric_remote]
        region_area = np.r_[np.full(len(ego_id), np.nan), area_remote]
        ego_overlap = np.zeros(len(boxes), np.float32)
        if len(corrected) and len(ego_boxes):
            ego_overlap[len(ego_id):] = iou32(corrected, ego_boxes).max(axis=1, initial=0.0)

        pose = np.asarray(meta["current_ego_pose_lidar_to_world"], np.float64)
        yaw = math.atan2(float(pose[1, 0]), float(pose[0, 0]))
        yaw_delta = 0.0 if key not in previous_yaw else math.atan2(
            math.sin(yaw - previous_yaw[key]), math.cos(yaw - previous_yaw[key])
        )
        previous_yaw[key] = yaw
        source_value = meta.get("source")
        source_frame = None if source_value is None else str(source_value.get("source_frame"))
        repeated_source = bool(source_frame is not None and source_frame == previous_source.get(key))
        previous_source[key] = source_frame
        sequence_values.append(str(meta["sequence"]))
        global_indices.append(int(meta["global_index"]))
        waits.append(int(meta["wait_ms"]))
        timestamps.append(timestamp)
        source_frames.append("" if source_frame is None else source_frame)
        trace.append([
            index, len(anchor_remote), variance, mean[0], mean[1],
            float(np.linalg.norm(residual, axis=1).mean()) if len(residual) else 0.0,
            yaw_delta, float(repeated_source), len(ego_id), len(stale_id),
        ])
        for name, value in {
            "boxes": boxes.astype(np.float32),
            "raw_scores": score.astype(np.float64),
            "source": source,
            "protected": protected.astype(np.int8),
            "anchor": anchor,
            "features": feature.astype(np.float32),
            "quality_y": quality,
            "utility_y": utility,
            "tolerance_circle": circle.astype(np.float64),
            "tolerance_numeric": numeric.astype(np.float64),
            "tolerance_region_area": region_area.astype(np.float64),
            "max_ego_iou": ego_overlap,
            "ious": matrix.astype(np.float32).reshape(-1),
            "gt_original_indices": pool["gt_original_indices"].astype(np.int32),
        }.items():
            arrays[name].append(value)
        condition_offsets.append(condition_offsets[-1] + len(boxes))
        gt_offsets.append(gt_offsets[-1] + len(pool["gt_boxes"]))
        iou_offsets.append(iou_offsets[-1] + matrix.size)
        if (count + 1) % 1000 == 0:
            print(json.dumps({"stage": f"materialize_{role}", "completed": count + 1,
                              "total": len(selected), "wall_s": time.monotonic() - started}), flush=True)
    archive = {name: np.concatenate(value) for name, value in arrays.items()}
    archive.update({
        "condition_offsets": np.asarray(condition_offsets, np.int64),
        "gt_offsets": np.asarray(gt_offsets, np.int64),
        "iou_offsets": np.asarray(iou_offsets, np.int64),
        "dataset_index": np.asarray(selected, np.int64),
        "sequence": np.asarray(sequence_values, dtype="U16"),
        "global_index": np.asarray(global_indices, np.int64),
        "wait_ms": np.asarray(waits, np.int16),
        "receiver_timestamp_us": np.asarray(timestamps, np.int64),
        "source_frame": np.asarray(source_frames, dtype="U32"),
        "trace_columns": np.asarray([
            "dataset_index", "anchors", "kf_variance", "kf_dx", "kf_dy",
            "anchor_residual_norm_mean", "ego_yaw_delta_rad", "repeated_source",
            "ego_candidates", "remote_candidates",
        ], dtype="U32"),
        "trace": np.asarray(trace, np.float64),
    })
    np.savez_compressed(output, **archive)
    return {
        "role": role,
        "conditions": len(selected),
        "candidates": int(condition_offsets[-1]),
        "gt": int(gt_offsets[-1]),
        "anchors": int(archive["anchor"].sum()),
        "remote": int(archive["source"].sum()),
        "sha256": sha256(output),
        "wall_seconds": time.monotonic() - started,
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def pool(materialized: dict[str, np.ndarray], condition: int) -> dict[str, np.ndarray]:
    a, b = materialized["condition_offsets"][condition:condition + 2].astype(int)
    ga, gb = materialized["gt_offsets"][condition:condition + 2].astype(int)
    ia, ib = materialized["iou_offsets"][condition:condition + 2].astype(int)
    value = {key: materialized[key][a:b] for key in (
        "boxes", "raw_scores", "source", "protected", "anchor", "features",
        "quality_y", "utility_y", "tolerance_circle", "tolerance_numeric",
        "tolerance_region_area", "max_ego_iou",
    )}
    value["gt_original_indices"] = materialized["gt_original_indices"][ga:gb]
    value["ious"] = materialized["ious"][ia:ib].reshape(b - a, gb - ga)
    value["absolute"] = np.arange(a, b, dtype=np.int64)
    return value


def balanced_training_schedule(
    materialized: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    groups: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    weights = np.zeros(len(materialized["boxes"]), np.float64)
    for condition in range(len(materialized["dataset_index"])):
        p = pool(materialized, condition)
        variable = p["absolute"][p["protected"] == 0]
        if len(variable):
            groups[str(materialized["sequence"][condition])][
                int(materialized["global_index"][condition])
            ].append(variable)
    names = sorted(groups)
    assert len(names) == 24
    for sequence in names:
        frames = groups[sequence]
        for waits in frames.values():
            for ids in waits:
                weights[ids] = 1.0 / (len(names) * len(frames) * len(waits) * len(ids))
    assert abs(float(weights.sum()) - 1.0) < 1e-10
    rng = np.random.default_rng(MODEL_SEED)
    batch = np.empty((MODEL_UPDATES, len(names) * MODEL_BATCH_PER_SEQUENCE), np.int32)
    for sequence_index, sequence in enumerate(names):
        frame_groups = [groups[sequence][key] for key in sorted(groups[sequence])]
        frames = rng.integers(len(frame_groups), size=(MODEL_UPDATES, MODEL_BATCH_PER_SEQUENCE))
        choose_wait = rng.random((MODEL_UPDATES, MODEL_BATCH_PER_SEQUENCE))
        choose_candidate = rng.random((MODEL_UPDATES, MODEL_BATCH_PER_SEQUENCE))
        for update in range(MODEL_UPDATES):
            for sample in range(MODEL_BATCH_PER_SEQUENCE):
                waits = frame_groups[int(frames[update, sample])]
                ids = waits[min(int(choose_wait[update, sample] * len(waits)), len(waits) - 1)]
                batch[update, sequence_index * MODEL_BATCH_PER_SEQUENCE + sample] = ids[
                    min(int(choose_candidate[update, sample] * len(ids)), len(ids) - 1)
                ]
    features = materialized["features"].astype(np.float64)
    mean = (features * weights[:, None]).sum(axis=0)
    std = np.maximum(np.sqrt(((features - mean) ** 2 * weights[:, None]).sum(axis=0)), 1e-6)
    return weights, batch, mean, std, names


def train_direct_u(materialized: dict[str, np.ndarray], output: Path) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(2)
    torch.manual_seed(MODEL_SEED)
    weights, batches, mean, std, sequences = balanced_training_schedule(materialized)
    normalized = ((materialized["features"] - mean) / std).astype(np.float32)
    target = materialized["utility_y"].astype(np.float32)
    model = nn.Sequential(nn.Linear(39, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    matrix = torch.from_numpy(normalized)
    labels = torch.from_numpy(target)
    trace = []
    recent = []
    started = time.monotonic()
    for update in range(MODEL_UPDATES):
        ids = torch.from_numpy(batches[update].astype(np.int64))
        logits = model(matrix[ids]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels[ids])
        assert bool(torch.isfinite(loss))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        recent.append(float(loss.detach()))
        if (update + 1) % 250 == 0:
            trace.append({"update": update + 1, "mean_loss_last250": float(np.mean(recent[-250:]))})
            print(json.dumps({"stage": "train_direct_u", **trace[-1]}), flush=True)
    torch.save(model.state_dict(), output / "direct_u.pth")
    np.savez_compressed(output / "training_schedule.npz", weights=weights, batches=batches,
                        mean=mean, std=std)
    result = {
        "schema": "shared_offset_corrected_coordinate_direct_u_training_v1",
        "fit_role": "fit24",
        "features": list(FEATURE_NAMES),
        "target": "utility_IoU70_relative_to_corrected_coordinate_protected_prefix",
        "model": "39x32_relu_x1",
        "seed": MODEL_SEED,
        "updates": MODEL_UPDATES,
        "batch_size": int(batches.shape[1]),
        "balanced_sequences": sequences,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trace": trace,
        "wall_seconds": time.monotonic() - started,
        "weights_sha256": sha256(output / "direct_u.pth"),
        "schedule_sha256": sha256(output / "training_schedule.npz"),
        "gpu_seconds": 0,
    }
    atomic(output / "training_summary.json", result)
    assert not torch.cuda.is_initialized()
    return result


def score_direct_u(materialized: dict[str, np.ndarray], output: Path) -> np.ndarray:
    import torch
    from torch import nn
    with np.load(output / "training_schedule.npz", allow_pickle=False) as archive:
        mean, std = archive["mean"], archive["std"]
    matrix = ((materialized["features"] - mean) / std).astype(np.float32)
    model = nn.Sequential(nn.Linear(39, 32), nn.ReLU(), nn.Linear(32, 1))
    model.load_state_dict(torch.load(output / "direct_u.pth", map_location="cpu"), strict=True)
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(matrix), 65536):
            rows.append(torch.sigmoid(model(torch.from_numpy(matrix[start:start + 65536])).squeeze(-1)).numpy())
    return np.concatenate(rows).astype(np.float64)


def fit_source_maps(cal: dict[str, np.ndarray], direct_score: np.ndarray) -> dict[str, Any]:
    variable = cal["protected"] == 0
    maps: dict[str, Any] = {
        "schema": "shared_offset_corrected_coordinate_source_maps_v1",
        "role": "calibration8",
        "population": "all_corrected_coordinate_variable_candidates",
        "maps": {},
    }
    inputs = {
        "raw_to_utility": (cal["raw_scores"], cal["utility_y"]),
        "raw_to_quality": (cal["raw_scores"], cal["quality_y"]),
        "direct_to_utility": (direct_score, cal["utility_y"]),
    }
    for method, (score, target) in inputs.items():
        maps["maps"][method] = {}
        for source_id, source_name in enumerate(("ego", "remote")):
            use = variable & (cal["source"] == source_id)
            maps["maps"][method][source_name] = fit_positive_logistic(score[use], target[use])
    return maps


def apply_source_map(score: np.ndarray, source: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    result = np.empty(len(score), np.float64)
    for source_id, source_name in enumerate(("ego", "remote")):
        use = source == source_id
        result[use] = apply_map(score[use], mapping[source_name])
    return result


def score_methods(
    materialized: dict[str, np.ndarray],
    direct: np.ndarray,
    maps: dict[str, Any],
) -> dict[str, np.ndarray]:
    source_u = apply_source_map(materialized["raw_scores"], materialized["source"],
                                maps["maps"]["raw_to_utility"])
    source_q = apply_source_map(materialized["raw_scores"], materialized["source"],
                                maps["maps"]["raw_to_quality"])
    direct_u = apply_source_map(direct, materialized["source"],
                                maps["maps"]["direct_to_utility"])
    task_circle = np.clip(source_q * materialized["tolerance_circle"], 0.0, 1.0)
    task_numeric = np.clip(source_q * materialized["tolerance_numeric"], 0.0, 1.0)
    return {
        "protected_raw": materialized["raw_scores"].astype(np.float64),
        "protected_source_u": source_u,
        "protected_direct_u": direct_u,
        "protected_task_circle": task_circle,
        "protected_task_numeric": task_numeric,
    }


def ordered(
    p: dict[str, np.ndarray],
    method: str,
    score: dict[str, np.ndarray],
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ego = np.flatnonzero(p["source"] == 0)
    remote = np.flatnonzero(p["source"] == 1)
    if method == "ego":
        rank = unique_ego_scores(p["raw_scores"][ego])
        order = np.argsort(-rank, kind="stable")
        return ego[order], rank[order]
    if method == "all_ego_raw":
        ego_rank = 2.0 + unique_ego_scores(p["raw_scores"][ego])
        remote_rank = unique_ego_scores(p["raw_scores"][remote], stable=True)
        eo = np.argsort(-ego_rank, kind="stable")
        ro = np.argsort(-remote_rank, kind="stable")
        return np.r_[ego[eo], remote[ro]], np.r_[ego_rank[eo], remote_rank[ro]]

    drop_anchor = method.endswith("_drop_anchors")
    base = method.removesuffix("_drop_anchors")
    fixed = np.flatnonzero(p["protected"] == 1)
    variable = np.flatnonzero(p["protected"] == 0)
    if drop_anchor:
        variable = variable[~((p["source"][variable] == 1) & (p["anchor"][variable] == 1))]
    variable_score = score[base][variable]
    if threshold is not None:
        keep = variable_score >= threshold
        variable, variable_score = variable[keep], variable_score[keep]
    fixed_rank = 2.0 + unique_ego_scores(p["raw_scores"][fixed], stable=True)
    variable_rank = unique_ego_scores(variable_score, stable=True)
    ids = np.r_[fixed, variable]
    rank = np.r_[fixed_rank, variable_rank]
    order = np.argsort(-rank, kind="stable")
    return ids[order], rank[order]


def calibrate_workpoints(
    materialized: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    collected = {name: {"fixed_tp": 0, "fixed_fp": 0, "score": [], "tp": [], "remote": []}
                 for name in WORKPOINT_BASES}
    ego_tp = ego_fp = 0
    for condition in range(len(materialized["dataset_index"])):
        p = pool(materialized, condition)
        a, b = materialized["condition_offsets"][condition:condition + 2].astype(int)
        local_scores = {name: value[a:b] for name, value in scores.items()}
        ego_order, _ = ordered(p, "ego", local_scores)
        ego_match = greedy(p["ious"], ego_order, IOU_THRESHOLD)
        ego_tp += int((ego_match >= 0).sum()); ego_fp += int((ego_match < 0).sum())
        for name in WORKPOINT_BASES:
            ids, rank = ordered(p, name, local_scores)
            matches = greedy(p["ious"], ids, IOU_THRESHOLD)
            fixed = p["protected"][ids] == 1
            row = collected[name]
            row["fixed_tp"] += int(((matches >= 0) & fixed).sum())
            row["fixed_fp"] += int(((matches < 0) & fixed).sum())
            # Freeze deployable thresholds on the actual pre-tie method score.
            # nextafter tie encodings are evaluator ordering details and must not
            # split an equal-score population at deployment.
            row["score"].append(local_scores[name][ids[~fixed]])
            row["tp"].append((matches[~fixed] >= 0).astype(np.int8))
            row["remote"].append(p["source"][ids[~fixed]])
    result = {
        "schema": "shared_offset_corrected_coordinate_workpoints_v1",
        "role": "calibration8",
        "betas": list(BETAS),
        "ego_tp": ego_tp,
        "ego_fp": ego_fp,
        "methods": {},
    }
    curves: dict[str, np.ndarray] = {}
    for name, row in collected.items():
        value = np.concatenate(row["score"])
        tp = np.concatenate(row["tp"])
        remote = np.concatenate(row["remote"])
        sorting = np.argsort(-value, kind="stable")
        value, tp, remote = value[sorting], tp[sorting], remote[sorting]
        ends = np.r_[np.flatnonzero(np.diff(value) != 0), len(value) - 1]
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(1 - tp)
        cumulative_remote = np.cumsum(remote)
        curves[name + "__score"] = value[ends]
        curves[name + "__tp"] = cumulative_tp[ends]
        curves[name + "__fp"] = cumulative_fp[ends]
        method = {"fixed_tp": row["fixed_tp"], "fixed_fp": row["fixed_fp"], "working_points": {}}
        for beta in BETAS:
            budget = int(math.floor(beta * ego_fp))
            chosen = {"strategy": "ego_fallback", "threshold": None,
                      "tp": ego_tp, "fp": ego_fp, "remote": 0}
            best = (ego_tp, -ego_fp, 0, 1, float("inf"))
            options = [(-1, None, row["fixed_tp"], row["fixed_fp"], 0)]
            options += [
                (int(endpoint), float(value[endpoint]),
                 int(row["fixed_tp"] + cumulative_tp[endpoint]),
                 int(row["fixed_fp"] + cumulative_fp[endpoint]),
                 int(cumulative_remote[endpoint]))
                for endpoint in ends
                if row["fixed_fp"] + cumulative_fp[endpoint] <= budget
            ]
            for endpoint, threshold, total_tp, total_fp, remote_count in options:
                if total_fp > budget:
                    continue
                key = (total_tp, -total_fp, -remote_count, 0,
                       float("inf") if threshold is None else threshold)
                if key > best:
                    best = key
                    chosen = {"strategy": "empty_variable" if endpoint < 0 else "threshold",
                              "threshold": threshold, "tp": total_tp, "fp": total_fp,
                              "remote": remote_count}
            chosen.update({"beta": beta, "total_fp_budget": budget,
                           "variable_fp_budget": budget - row["fixed_fp"]})
            method["working_points"][str(beta)] = chosen
        result["methods"][name] = method
    return result, curves


def variant_specs(calibration: dict[str, Any]) -> list[tuple[str, str, float | None]]:
    variants = [(name, name, None) for name in FULL_METHODS]
    for base in WORKPOINT_BASES:
        for beta in BETAS:
            point = calibration["methods"][base]["working_points"][str(beta)]
            # A calibration fallback is the complete ego detector output.  It is
            # not the protected-only fixed prefix.  In contrast, empty_variable
            # intentionally keeps only the protected prefix.
            if point["strategy"] == "ego_fallback":
                deployed_base, threshold = "ego", None
            elif point["strategy"] == "empty_variable":
                deployed_base, threshold = base, float("inf")
            else:
                deployed_base, threshold = base, float(point["threshold"])
            variants.append((f"{base}_beta{beta:.1f}", deployed_base, threshold))
    return variants


def evaluate(
    materialized: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    calibration: dict[str, Any],
    role: str,
    output: Path,
) -> dict[str, Any]:
    variants = variant_specs(calibration)
    stats = defaultdict(lambda: {
        "scores": [], "tp50": [], "tp70": [], "gt": 0, "outputs": 0,
        "remote": 0, "anchor_remote": 0, "remote_quality": 0, "remote_utility": 0,
        "remote_no_ego_overlap": 0, "remote_no_ego_overlap_utility": 0,
        "motion_confirmed_remote": 0, "motion_confirmed_utility": 0,
        "rescued": 0, "destroyed": 0,
    })
    saved: dict[str, dict[str, list[Any]]] = {
        name: {"candidate": [], "score": [], "offset": [0]}
        for name, _, _ in variants
    }
    started = time.monotonic()
    for condition in range(len(materialized["dataset_index"])):
        p = pool(materialized, condition)
        a, b = materialized["condition_offsets"][condition:condition + 2].astype(int)
        local_scores = {name: value[a:b] for name, value in scores.items()}
        ego_order, _ = ordered(p, "ego", local_scores)
        ego_match = greedy(p["ious"], ego_order, IOU_THRESHOLD)
        ego_set = {int(value) for value in ego_match if value >= 0}
        wait = int(materialized["wait_ms"][condition])
        for name, base, threshold in variants:
            ids, rank = ordered(p, base, local_scores, threshold)
            # Keep float64 tie encodings so the reviewed all-ego/raw reference is
            # reproduced exactly; the prediction artifact is the AP source.
            rank_saved = rank.astype(np.float64)
            match50 = greedy(p["ious"], ids, 0.5)
            match70 = greedy(p["ious"], ids, IOU_THRESHOLD)
            matched = {int(value) for value in match70 if value >= 0}
            remote = p["source"][ids] == 1
            no_overlap = p["max_ego_iou"][ids] < 0.1
            motion = p["features"][ids, 32] > 0.5
            row = stats[(wait, name)]
            row["scores"].append(rank_saved)
            row["tp50"].append((match50 >= 0).astype(np.int8))
            row["tp70"].append((match70 >= 0).astype(np.int8))
            row["gt"] += p["ious"].shape[1]
            row["outputs"] += len(ids)
            row["remote"] += int(remote.sum())
            row["anchor_remote"] += int((remote & (p["anchor"][ids] == 1)).sum())
            row["remote_quality"] += int((remote & (p["quality_y"][ids] == 1)).sum())
            row["remote_utility"] += int((remote & (p["utility_y"][ids] == 1)).sum())
            row["remote_no_ego_overlap"] += int((remote & no_overlap).sum())
            row["remote_no_ego_overlap_utility"] += int((remote & no_overlap & (p["utility_y"][ids] == 1)).sum())
            row["motion_confirmed_remote"] += int((remote & motion).sum())
            row["motion_confirmed_utility"] += int((remote & motion & (p["utility_y"][ids] == 1)).sum())
            row["rescued"] += len(matched - ego_set)
            row["destroyed"] += len(ego_set - matched)
            saved[name]["candidate"].append(p["absolute"][ids])
            saved[name]["score"].append(rank_saved)
            saved[name]["offset"].append(saved[name]["offset"][-1] + len(ids))
        if (condition + 1) % 1000 == 0:
            print(json.dumps({"stage": f"evaluate_{role}", "completed": condition + 1,
                              "total": len(materialized["dataset_index"]),
                              "wall_s": time.monotonic() - started}), flush=True)
    panels = []
    for (wait, name), row in sorted(stats.items()):
        score = np.concatenate(row["scores"])
        panel = {"wait_ms": wait, "method": name}
        for threshold in (50, 70):
            tp = np.concatenate(row["tp" + str(threshold)])
            metric = voc_ap(tp.tolist(), (1 - tp).tolist(), score.tolist(), row["gt"], True)
            panel["ap" + str(threshold)] = float(metric["ap"] * 100.0)
            if threshold == 70:
                panel.update({"tp": int(metric["tp"]), "fp": int(metric["fp"]), "gt": int(metric["gt"])})
        panel.update({key: int(value) for key, value in row.items() if not isinstance(value, list)})
        panels.append(panel)
    archive: dict[str, np.ndarray] = {}
    for method, rows in saved.items():
        archive[method + "__candidate"] = np.concatenate(rows["candidate"]).astype(np.int64)
        archive[method + "__score"] = np.concatenate(rows["score"]).astype(np.float64)
        archive[method + "__offset"] = np.asarray(rows["offset"], np.int64)
    prediction_path = output / f"{role}_predictions.npz"
    np.savez_compressed(prediction_path, **archive)
    result = {
        "role": role,
        "conditions": int(len(materialized["dataset_index"])),
        "variants": len(variants),
        "method_conditions": int(len(materialized["dataset_index"]) * len(variants)),
        "panels": panels,
        "predictions_sha256": sha256(prediction_path),
        "wall_seconds": time.monotonic() - started,
    }
    atomic(output / f"{role}_summary.json", result)
    return result


def condition_audit(materialized: dict[str, np.ndarray], role: str) -> dict[str, Any]:
    anchor_bins = defaultdict(lambda: {"conditions": 0, "remote": 0, "utility": 0, "no_ego_overlap_utility": 0})
    age_bins = defaultdict(lambda: {"remote": 0, "utility": 0})
    motion = defaultdict(lambda: {"remote": 0, "utility": 0})
    turning = defaultdict(lambda: {"conditions": 0, "remote": 0, "utility": 0})
    repeated = defaultdict(lambda: {"conditions": 0, "remote": 0, "utility": 0})
    trace = materialized["trace"]
    for condition in range(len(materialized["dataset_index"])):
        p = pool(materialized, condition)
        remote = p["source"] == 1
        utility = p["utility_y"] == 1
        anchors = int(trace[condition, 1])
        anchor_name = "0" if anchors == 0 else ("1" if anchors == 1 else ("2" if anchors == 2 else "3plus"))
        row = anchor_bins[anchor_name]
        row["conditions"] += 1; row["remote"] += int(remote.sum())
        row["utility"] += int((remote & utility).sum())
        row["no_ego_overlap_utility"] += int((remote & utility & (p["max_ego_iou"] < .1)).sum())
        remote_ids = np.flatnonzero(remote)
        for local in remote_ids:
            age = float(p["features"][local, 0])
            age_name = "lt100" if age < 100 else ("100to199" if age < 200 else ("200to399" if age < 400 else "ge400"))
            age_bins[age_name]["remote"] += 1
            age_bins[age_name]["utility"] += int(utility[local])
            motion_name = "confirmed" if p["features"][local, 32] > .5 else "unconfirmed"
            motion[motion_name]["remote"] += 1
            motion[motion_name]["utility"] += int(utility[local])
        turn_name = "turning_ge1deg" if abs(float(trace[condition, 6])) >= math.radians(1.0) else "turning_lt1deg"
        turning[turn_name]["conditions"] += 1
        turning[turn_name]["remote"] += int(remote.sum())
        turning[turn_name]["utility"] += int((remote & utility).sum())
        repeat_name = "repeated_source" if trace[condition, 7] > .5 else "fresh_or_none"
        repeated[repeat_name]["conditions"] += 1
        repeated[repeat_name]["remote"] += int(remote.sum())
        repeated[repeat_name]["utility"] += int((remote & utility).sum())
    return {
        "role": role,
        "anchor_bins": dict(anchor_bins),
        "actual_age_bins_ms": dict(age_bins),
        "motion_confirmation": dict(motion),
        "receiver_turning": dict(turning),
        "source_repetition": dict(repeated),
        "receiver_overlap_definition": "max IoU with any actual current ego detector box; <0.1 is no overlapping detection candidate, not physical invisibility",
    }


def reference_panels(path: Path) -> dict[tuple[int, str], float]:
    value = json.loads(path.read_text())
    return {(int(row["wait_ms"]), str(row["method"])): float(row["ap70"])
            for row in value["panels"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--val-dataset", required=True, type=Path)
    parser.add_argument("--hold-reference", required=True, type=Path)
    parser.add_argument("--val-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    train, train_summary = checked_dataset(args.train_dataset)
    val, val_summary = checked_dataset(args.val_dataset)
    contract = {
        "schema": "shared_offset_task_tolerance_readout_contract_v1",
        "status": "frozen_before_new_readout_outcomes",
        "task_id": "shared_offset_task_tolerance_readout_20261026",
        "owner": "research_autopilot_shared_offset_readout_v1",
        "geometry": "reviewed causal_kf additive correction on one CV-or-stale fallback per accepted source identity",
        "geometry_inputs": ["boxes", "scores", "protected flags", "source identity", "causal receiver timestamp"],
        "geometry_forbidden": ["GT", "official offset", "future frame"],
        "coordinate_contract": "Every remote receiver-coordinate field, protected-ego relation, IoU, target and query is regenerated after shared correction. No old-position patch/query is reused.",
        "fit_role": "fit24 direct-U head only",
        "calibration_role": "calibration8 sourcewise positive-slope maps and beta workpoints",
        "evaluation_roles": ["mechanism_holdout14", "val21"],
        "full_methods": list(FULL_METHODS),
        "workpoint_bases": list(WORKPOINT_BASES),
        "betas": list(BETAS),
        "anchor_removal": "separate candidate-selection control; never attributed to probability scoring",
        "task_probability": {
            "existence": "sourcewise calibrated detector quality probability; raw score is not called calibrated probability",
            "uncertainty": "reviewed KF scalar variance is a nominal score input, not claimed independent/correct posterior",
            "circle": "equal-area-circle approximation",
            "numeric": f"{QUADRATURE_POINTS}-point Gauss-Legendre integral over exact equal-size/equal-yaw translation IoU>=0.7 region",
            "limitations": "box size/yaw errors, non-Gaussian tails and association failures are not represented",
        },
        "direct_u": {"model": "39x32_relu_x1", "seed": MODEL_SEED,
                     "updates": MODEL_UPDATES, "target": "corrected-coordinate utility"},
        "no_new_detector_or_backbone_forward": True,
        "not_end_to_end_latency": True,
        "train_dataset": str(args.train_dataset),
        "val_dataset": str(args.val_dataset),
        "input_sha256": {
            "train_pools": train_summary["pools_sha256"],
            "train_conditions": train_summary["conditions_sha256"],
            "val_pools": val_summary["pools_sha256"],
            "val_conditions": val_summary["conditions_sha256"],
            "hold_reference": sha256(args.hold_reference),
            "val_reference": sha256(args.val_reference),
        },
    }
    atomic(args.output / "contract.json", contract)
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)

    material_paths = {
        "fit24": args.output / "fit24_corrected.npz",
        "calibration8": args.output / "calibration8_corrected.npz",
        "mechanism_holdout14": args.output / "mechanism_holdout14_corrected.npz",
        "val21": args.output / "val21_corrected.npz",
    }
    material_summaries = {
        role: materialize_role(train if role != "val21" else val, role, path)
        for role, path in material_paths.items()
    }
    atomic(args.output / "materialization_summary.json", {
        "schema": "shared_offset_corrected_coordinate_materialization_v1",
        "roles": material_summaries,
        "feature_names": list(FEATURE_NAMES),
        "new_detector_or_backbone_forwards": 0,
    })
    fit = load_npz(material_paths["fit24"])
    cal = load_npz(material_paths["calibration8"])
    hold = load_npz(material_paths["mechanism_holdout14"])
    val_material = load_npz(material_paths["val21"])
    training = train_direct_u(fit, args.output)
    direct = {
        "fit24": score_direct_u(fit, args.output),
        "calibration8": score_direct_u(cal, args.output),
        "mechanism_holdout14": score_direct_u(hold, args.output),
        "val21": score_direct_u(val_material, args.output),
    }
    np.savez_compressed(args.output / "direct_u_scores.npz", **direct)
    maps = fit_source_maps(cal, direct["calibration8"])
    atomic(args.output / "source_maps.json", maps)
    score_cal = score_methods(cal, direct["calibration8"], maps)
    calibration, curves = calibrate_workpoints(cal, score_cal)
    np.savez_compressed(args.output / "calibration_curves.npz", **curves)
    atomic(args.output / "calibration.json", calibration)

    hold_result = evaluate(hold, score_methods(hold, direct["mechanism_holdout14"], maps),
                           calibration, "mechanism_holdout14", args.output)
    val_result = evaluate(val_material, score_methods(val_material, direct["val21"], maps),
                          calibration, "val21", args.output)
    audits = {
        "mechanism_holdout14": condition_audit(hold, "mechanism_holdout14"),
        "val21": condition_audit(val_material, "val21"),
    }
    atomic(args.output / "condition_audit.json", audits)

    reference = {
        "mechanism_holdout14": reference_panels(args.hold_reference),
        "val21": reference_panels(args.val_reference),
    }
    exact = {}
    for role, result in (("mechanism_holdout14", hold_result), ("val21", val_result)):
        actual = {int(row["wait_ms"]): float(row["ap70"])
                  for row in result["panels"] if row["method"] == "all_ego_raw"}
        delta = {str(wait): actual[wait] - reference[role][(wait, "causal_kf")] for wait in WAITS}
        exact[role] = {"ap70_difference_pp": delta, "max_abs_pp": max(abs(x) for x in delta.values())}
        assert exact[role]["max_abs_pp"] == 0.0

    summary = {
        "schema": "shared_offset_task_tolerance_readout_summary_v1",
        "status": "complete_producer_review_pending",
        "task_id": contract["task_id"],
        "owner": contract["owner"],
        "materialization": material_summaries,
        "training": training,
        "hold": hold_result,
        "val": val_result,
        "reviewed_causal_kf_exact_reproduction": exact,
        "artifact_sha256": {
            name: sha256(args.output / name) for name in (
                "contract.json", "materialization_summary.json", "fit24_corrected.npz",
                "calibration8_corrected.npz", "mechanism_holdout14_corrected.npz",
                "val21_corrected.npz", "direct_u.pth", "training_schedule.npz",
                "direct_u_scores.npz", "source_maps.json", "calibration.json",
                "calibration_curves.npz", "mechanism_holdout14_predictions.npz",
                "val21_predictions.npz", "condition_audit.json",
            )
        },
        "script_sha256": sha256(Path(__file__)),
        "wall_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
        "detector_forwards": 0,
        "backbone_forwards": 0,
        "head_updates": MODEL_UPDATES,
        "gpu_seconds": 0,
        "cost_boundary": "CPU cached materialization/head fitting/evaluation only; not complete-system inference latency",
    }
    atomic(args.output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("hold", "val")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
