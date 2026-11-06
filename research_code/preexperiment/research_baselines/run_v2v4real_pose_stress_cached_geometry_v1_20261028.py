#!/usr/bin/env python3
"""Frozen V2V4Real cached-geometry pose stress with clean-fit controls.

The sender pose perturbation is applied consistently by sequence and sender
timestamp to the current and causal-history packets.  Detector outputs are
frozen.  The task head, 39D readout, utility maps, and clean fit-only Huber
constant are all reused without refitting or perturbation augmentation.

This remains a cached geometry/readout study, not a perturbed point-cloud or
feature-fusion system run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / "preexperiment/research_autopilot"
BASE = ROOT / "preexperiment/research_baselines"
sys.path[:0] = [str(AUTO), str(BASE)]

import run_strict60_task_supervised_shared_correction_head_v1_20261028 as task
import run_v2v4real_task_geometry_fresh39d_v1_20261028 as v2recipe
import run_v2xvit_seed2024_task_geometry_fresh39d_v1_20261028 as recipe
import run_shared_offset_optional_formal_v1_20261026 as formal
import run_freealign_official_box_same_pool_v1_20261027 as freealign
from diagnose_v2v_legal_history_geometry_v1_20261023 import (
    associate, common_mask, local_artifact, sensor_manifests,
)
from e1_receiver_value_core_v1_20261020 import greedy, iou32
from fit_evaluate_candidate_d_heads_v1_20261019 import feature_matrix
from materialize_v2v4real_shared_offset_39d_inputs_v1_20261027 import extras
from object_history_selector_core_v1_20261021 import PackedDataset
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap
from v2xseq_real_motion_packets import transform_boxes7d


ARM_SPECS: dict[str, dict[str, float | str]] = {
    "clean": {"kind": "constant", "tx_m": 0.0, "ty_m": 0.0, "yaw_deg": 0.0},
    "const_tx_plus_0p5m": {"kind": "constant", "tx_m": 0.5, "ty_m": 0.0, "yaw_deg": 0.0},
    "const_tx_minus_0p5m": {"kind": "constant", "tx_m": -0.5, "ty_m": 0.0, "yaw_deg": 0.0},
    "const_ty_plus_0p5m": {"kind": "constant", "tx_m": 0.0, "ty_m": 0.5, "yaw_deg": 0.0},
    "const_ty_minus_0p5m": {"kind": "constant", "tx_m": 0.0, "ty_m": -0.5, "yaw_deg": 0.0},
    "const_yaw_plus_1deg": {"kind": "constant", "tx_m": 0.0, "ty_m": 0.0, "yaw_deg": 1.0},
    "const_yaw_minus_1deg": {"kind": "constant", "tx_m": 0.0, "ty_m": 0.0, "yaw_deg": -1.0},
    "slow_drift_tx_yaw_positive": {
        "kind": "drift", "tx_per_100ms_m": 0.025, "tx_cap_m": 0.5,
        "yaw_per_100ms_deg": 0.05, "yaw_cap_deg": 1.0,
    },
}
FORMAL_ROLES = ("mechanism_holdout14", "val21")
GEOMETRIES = ("stale", "cv", "fit_huber_constant", "causal_kf", "current_task", "freealign")
SCORINGS = ("raw", "frozen39d")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def evidence(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": recipe.sha256(path),
            "bytes": path.stat().st_size}


def perturbation(arm: str, timestamp: str | int) -> tuple[float, float, float]:
    spec = ARM_SPECS[arm]
    if spec["kind"] == "constant":
        return float(spec["tx_m"]), float(spec["ty_m"]), math.radians(float(spec["yaw_deg"]))
    step = max(0, int(timestamp))
    tx = min(float(spec["tx_cap_m"]), step * float(spec["tx_per_100ms_m"]))
    yaw = min(float(spec["yaw_cap_deg"]), step * float(spec["yaw_per_100ms_deg"]))
    return tx, 0.0, math.radians(yaw)


def perturb_pose(pose: np.ndarray, arm: str, timestamp: str | int) -> np.ndarray:
    """Add x/y/yaw bias to lidar-to-world pose without rotating world position."""
    tx, ty, yaw = perturbation(arm, timestamp)
    output = np.asarray(pose, np.float64).copy()
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw), 0.0],
                           [math.sin(yaw), math.cos(yaw), 0.0],
                           [0.0, 0.0, 1.0]], np.float64)
    output[:3, :3] = rotation @ output[:3, :3]
    output[0, 3] += tx; output[1, 3] += ty
    return output


def angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def pose_matrix_from_freealign(value: np.ndarray) -> np.ndarray:
    yaw = math.radians(float(value[4]))
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = [[math.cos(yaw), -math.sin(yaw)],
                      [math.sin(yaw), math.cos(yaw)]]
    result[:2, 3] = value[:2]
    return result


class PerturbedDataset:
    def __init__(self, directory: Path, history_directory: Path, arm: str,
                 manifests: dict[int, dict[str, Any]]):
        self.directory = directory
        self.base = PackedDataset(directory)
        self.meta = [dict(row) for row in self.base.meta]
        self.history = list(formal.read_rows(history_directory / "conditions.jsonl")) \
            if hasattr(formal, "read_rows") else [json.loads(line) for line in
                (history_directory / "conditions.jsonl").read_text().splitlines() if line.strip()]
        assert len(self.meta) == len(self.history)
        self.arm = arm; self.manifests = manifests
        self.cache: dict[int, dict[str, np.ndarray]] = {}
        self.raw_cache: dict[str, dict[str, np.ndarray]] = {}
        self.clean_errors: Counter[str] = Counter()
        for index, row in enumerate(self.meta):
            row = dict(row)
            if row.get("previous_raw_path"):
                path = local_artifact(ROOT, row["previous_raw_path"])
                row["previous_source_raw_path"] = str(path.resolve())
            if row.get("source_raw_path"):
                row["source_raw_path"] = str(local_artifact(ROOT, row["source_raw_path"]).resolve())
            if row.get("source") is not None:
                source = dict(row["source"]); source["actual_age_ms"] = float(row["effective_age_ms"])
                row["source"] = source
            self.meta[index] = row

    def raw(self, path: Path) -> dict[str, np.ndarray]:
        key = str(path.resolve())
        if key not in self.raw_cache:
            self.raw_cache[key] = load_npz(path)
        return self.raw_cache[key]

    def pool(self, index: int) -> dict[str, np.ndarray]:
        if index in self.cache:
            return self.cache[index]
        meta = self.meta[index]; history = self.history[index]; base = self.base.pool(index)
        assert int(history["condition_index"]) == index
        kind = base["hypothesis_type"].astype(np.int8)
        ego = np.flatnonzero(kind == 0); stale_clean = np.flatnonzero(kind == 1)
        source_ids = base["source_detection_indices"][stale_clean].astype(np.int32)
        task_index = int(history["source_task_index"]); manifest = self.manifests[task_index]
        current_path = local_artifact(ROOT, history["current_raw_path"])
        current = self.raw(current_path)
        source_true = np.asarray(manifest["source_pose_lidar_to_world"], np.float64)
        source_pose = perturb_pose(source_true, self.arm, manifest["timestamp"])
        receiver_pose = np.asarray(meta["current_ego_pose_lidar_to_world"], np.float64)
        previous = None; previous_pose = None; dt_s = 0.0
        if history["previous_task_index"] is not None:
            previous_manifest = self.manifests[int(history["previous_task_index"])]
            previous = self.raw(local_artifact(ROOT, history["previous_raw_path"]))
            previous_pose = perturb_pose(
                np.asarray(previous_manifest["source_pose_lidar_to_world"], np.float64),
                self.arm, previous_manifest["timestamp"])
            dt_s = 0.1
        horizon_s = float(meta["effective_age_ms"]) / 1000.0
        motion = associate(current, previous, source_pose, previous_pose, dt_s, horizon_s)
        stale_all = transform_boxes7d(current["pred_boxes"], source_pose, receiver_pose)
        prop_all = transform_boxes7d(motion["boxes"], source_pose, receiver_pose)
        prop_valid = motion["confirmed"][source_ids] & common_mask(prop_all[source_ids])
        protected_ego = ego[base["is_protected"][ego].astype(bool)]
        protected_boxes = base["boxes"][protected_ego]
        protected_scores = base["raw_scores"][protected_ego]
        source_to_receiver = np.linalg.solve(receiver_pose, source_pose)
        stale_boxes = stale_all[source_ids]
        stale30 = feature_matrix(
            stale_boxes, current["pred_boxes"][source_ids], current["pred_scores"][source_ids],
            float(meta["effective_age_ms"]), source_to_receiver, protected_boxes, protected_scores)
        stale_extra = extras(motion, source_ids, horizon_s, dt_s)
        stale_features = np.column_stack((stale30, stale_extra)).astype(np.float32)
        prop_ids = source_ids[prop_valid]; prop_boxes = prop_all[prop_ids]
        prop30 = feature_matrix(
            prop_boxes, current["pred_boxes"][prop_ids], current["pred_scores"][prop_ids],
            float(meta["effective_age_ms"]), source_to_receiver, protected_boxes, protected_scores)
        prop_extra = stale_extra[prop_valid].copy(); prop_extra[:, 1] = 1.0
        prop_features = np.column_stack((prop30, prop_extra)).astype(np.float32)
        boxes = np.concatenate((base["boxes"][ego], stale_boxes, prop_boxes)).astype(np.float32)
        scores = np.r_[base["raw_scores"][ego], current["pred_scores"][source_ids],
                       current["pred_scores"][prop_ids]].astype(base["raw_scores"].dtype)
        types = np.r_[np.zeros(len(ego), np.int8), np.ones(len(source_ids), np.int8),
                      np.full(len(prop_ids), 2, np.int8)]
        ids = np.r_[base["source_detection_indices"][ego], source_ids, prop_ids].astype(np.int32)
        protected = np.r_[base["is_protected"][ego], np.zeros(len(source_ids) + len(prop_ids), np.int8)]
        features = np.concatenate((base["features"][ego], stale_features, prop_features)).astype(np.float32)
        gt = base["gt_boxes"].copy()
        result = {
            "boxes": boxes, "raw_scores": scores, "hypothesis_type": types,
            "is_protected": protected, "source_detection_indices": ids,
            "features": features, "gt_boxes": gt, "ious": iou32(boxes, gt),
        }
        if self.arm == "clean":
            for key in ("boxes", "raw_scores", "hypothesis_type", "is_protected",
                        "source_detection_indices", "features"):
                self.clean_errors[key] = max(self.clean_errors[key], float(np.max(
                    np.abs(np.asarray(result[key], np.float64) - np.asarray(base[key], np.float64)),
                    initial=0.0)))
        self.cache[index] = result
        return result


class FreeAlignPoseCache:
    def __init__(self):
        self.values: dict[tuple[str, int], dict[str, Any]] = {}

    def get(self, dataset: PerturbedDataset, index: int) -> dict[str, Any]:
        key = (str(dataset.directory), index)
        if key in self.values:
            return self.values[key]
        meta = dataset.meta[index]; base = dataset.base.pool(index)
        kind = base["hypothesis_type"].astype(np.int8)
        ego = np.flatnonzero(kind == 0); stale = np.flatnonzero(kind == 1)
        source_ids = base["source_detection_indices"][stale].astype(np.int32)
        raw = dataset.raw(local_artifact(ROOT, dataset.history[index]["current_raw_path"]))
        ego_boxes = base["boxes"][ego]
        source_boxes = raw["pred_boxes"][source_ids]
        ego_corners = freealign.boxes_to_corners_3d(ego_boxes, "hwl")
        source_corners = freealign.boxes_to_corners_3d(source_boxes, "hwl")
        pose, mcs_size, matches, error, _ = freealign.capture_official_pose(ego_corners, source_corners)
        if error is not None:
            status = "exception_stale_fallback"
        elif not np.all(np.isfinite(pose)):
            status = "nonfinite_stale_fallback"
        elif len(ego_boxes) <= 2 or len(source_boxes) <= 2:
            status = "insufficient_boxes_stale_fallback"
        elif mcs_size < 3:
            status = "no_common_subgraph_stale_fallback"
        else:
            status = "estimated"
        value = {"pose": pose, "mcs_size": int(mcs_size), "matches": matches,
                 "error": error, "status": status, "source_boxes": source_boxes,
                 "source_ids": source_ids}
        self.values[key] = value
        return value


def infer_task(feature: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    import torch
    from torch import nn
    protocol = load_npz(ROOT / contract["inputs"]["task_fit_protocol"]["path"])
    model = nn.Sequential(nn.Linear(len(task.FEATURE_NAMES), 32), nn.ReLU(), nn.Linear(32, 2))
    model.load_state_dict(torch.load(ROOT / contract["inputs"]["task_head"]["path"], map_location="cpu"), strict=True)
    model.eval(); values = []
    normalized = ((feature - protocol["mean"]) / protocol["std"]).astype(np.float32)
    scale = float(protocol["fit_scale"])
    with torch.no_grad():
        tensor = torch.from_numpy(normalized)
        for start in range(0, len(tensor), 65536):
            values.append((scale * torch.tanh(model(tensor[start:start + 65536]))).numpy())
    return np.concatenate(values).astype(np.float64)


def build_role(dataset: PerturbedDataset, role: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    old = recipe.source_metadata; recipe.source_metadata = v2recipe.source_metadata
    try:
        return recipe.build_role(dataset, role, False)
    finally:
        recipe.source_metadata = old


def direct_features(
    dataset: PerturbedDataset,
    data: dict[str, np.ndarray],
    frame: dict[str, np.ndarray],
    task_delta: np.ndarray,
    fit_constant: np.ndarray,
    official: FreeAlignPoseCache,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    boxes = {name: [] for name in GEOMETRIES}
    features = {name: [] for name in GEOMETRIES}
    freealign_status: Counter[str] = Counter()
    freealign_pose_xy, freealign_pose_yaw = [], []
    corrupted_pose_xy, corrupted_pose_yaw = [], []
    for condition, index_value in enumerate(data["dataset_index"]):
        index = int(index_value); pool = dataset.pool(index); meta = dataset.meta[index]
        ego, stale_ids, cv_ids = formal.identities(pool)
        ego_boxes = pool["boxes"][ego].copy(); stale_remote = pool["boxes"][stale_ids].copy()
        cv_remote = pool["boxes"][cv_ids].copy()
        protected_local = np.flatnonzero(pool["is_protected"][ego])
        protected_boxes = ego_boxes[protected_local]
        protected_scores = pool["raw_scores"][ego][protected_local]
        base_stale_features = pool["features"][stale_ids].copy()
        base_cv_features = pool["features"][cv_ids].copy()
        remote_by_method = {
            "stale": stale_remote,
            "cv": cv_remote,
            "fit_huber_constant": cv_remote.copy(),
            "causal_kf": cv_remote.copy(),
            "current_task": cv_remote.copy(),
        }
        remote_by_method["fit_huber_constant"][:, :2] += fit_constant.astype(np.float32)
        remote_by_method["causal_kf"][:, :2] += frame["kf_delta"][condition].astype(np.float32)
        remote_by_method["current_task"][:, :2] += task_delta[condition].astype(np.float32)
        for method in ("stale", "cv", "fit_huber_constant", "causal_kf", "current_task"):
            remote = remote_by_method[method]
            base = base_stale_features if method == "stale" else base_cv_features
            remote_feature = formal.update_features(base, remote, protected_boxes, protected_scores)
            boxes[method].append(np.concatenate((ego_boxes, remote)).astype(np.float32))
            features[method].append(np.concatenate((pool["features"][ego], remote_feature)).astype(np.float32))

        fa = official.get(dataset, index); freealign_status[fa["status"]] += 1
        if fa["status"] == "estimated":
            fa_remote = freealign.apply_absolute_pose(fa["source_boxes"], fa["pose"])
            transform = pose_matrix_from_freealign(fa["pose"])
            first30 = feature_matrix(
                fa_remote, fa["source_boxes"], pool["raw_scores"][stale_ids],
                float(meta["effective_age_ms"]), transform, protected_boxes, protected_scores)
            fa_features = np.column_stack((first30, base_stale_features[:, 30:])).astype(np.float32)
            nominal = np.linalg.solve(np.asarray(meta["current_ego_pose_lidar_to_world"], np.float64),
                                      np.asarray(meta["source"]["source_pose_lidar_to_world"], np.float64))
            freealign_pose_xy.append(float(np.linalg.norm(fa["pose"][:2] - nominal[:2, 3])))
            freealign_pose_yaw.append(abs(float(angle(math.radians(float(fa["pose"][4]))
                                                      - math.atan2(nominal[1, 0], nominal[0, 0])))))
        else:
            fa_remote = stale_remote.copy(); fa_features = base_stale_features.copy()
        boxes["freealign"].append(np.concatenate((ego_boxes, fa_remote)).astype(np.float32))
        features["freealign"].append(np.concatenate((pool["features"][ego], fa_features)).astype(np.float32))

        history = dataset.history[index]; manifest = dataset.manifests[int(history["source_task_index"])]
        true_source = np.asarray(manifest["source_pose_lidar_to_world"], np.float64)
        noisy_source = perturb_pose(true_source, dataset.arm, manifest["timestamp"])
        receiver = np.asarray(meta["current_ego_pose_lidar_to_world"], np.float64)
        true_relative = np.linalg.solve(receiver, true_source); noisy_relative = np.linalg.solve(receiver, noisy_source)
        corrupted_pose_xy.append(float(np.linalg.norm(noisy_relative[:2, 3] - true_relative[:2, 3])))
        corrupted_pose_yaw.append(abs(float(angle(math.atan2(noisy_relative[1, 0], noisy_relative[0, 0])
                                                     - math.atan2(true_relative[1, 0], true_relative[0, 0])))))
    box_arrays = {name: np.concatenate(value) for name, value in boxes.items()}
    feature_arrays = {name: np.concatenate(value) for name, value in features.items()}
    diagnostic = {
        "freealign_status_counts": dict(sorted(freealign_status.items())),
        "freealign_success_rate": freealign_status["estimated"] / max(len(data["dataset_index"]), 1),
        "freealign_success_pose_xy_error_m": summary_stats(freealign_pose_xy),
        "freealign_success_pose_yaw_error_deg": summary_stats(np.degrees(freealign_pose_yaw)),
        "corrupted_nominal_pose_xy_error_m": summary_stats(corrupted_pose_xy),
        "corrupted_nominal_pose_yaw_error_deg": summary_stats(np.degrees(corrupted_pose_yaw)),
    }
    return box_arrays, feature_arrays, diagnostic


def summary_stats(values: Any) -> dict[str, float | int | None]:
    array = np.asarray(values, np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def recovery_stats(boxes: dict[str, np.ndarray], clean: dict[str, np.ndarray],
                   source: np.ndarray) -> dict[str, Any]:
    remote = source == 1; result = {}
    for method in GEOMETRIES:
        reference = clean["stale" if method in ("stale", "freealign") else "cv"]
        delta = boxes[method][remote, :2].astype(np.float64) - reference[remote, :2].astype(np.float64)
        center = np.linalg.norm(delta, axis=1)
        yaw = np.abs(angle(boxes[method][remote, 6].astype(np.float64)
                           - reference[remote, 6].astype(np.float64)))
        result[method] = {"center_error_m": summary_stats(center),
                          "yaw_error_deg": summary_stats(np.degrees(yaw))}
    return result


def evaluate_role(
    dataset: PerturbedDataset,
    role: str,
    data: dict[str, np.ndarray],
    boxes: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    readout: Path,
    source_maps: dict[str, Any],
    output: Path,
    arm: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    direct = formal.direct_scores(features, readout)
    utility = {name: formal.source_map(score, data["source"], source_maps["maps"]["direct_to_utility"])
               for name, score in direct.items()}
    methods = [f"{geometry}_{scoring}" for geometry in GEOMETRIES for scoring in SCORINGS]
    aggregate = defaultdict(lambda: {"score": [], "tp50": [], "tp70": [], "gt": 0, "outputs": 0})
    saved = {method: {"candidate": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
             for method in methods}
    for condition, index_value in enumerate(data["dataset_index"]):
        index = int(index_value); pool = dataset.pool(index); gt = pool["gt_boxes"]
        a, b = data["condition_offsets"][condition:condition + 2].astype(int)
        raw = data["raw_scores"][a:b]; source = data["source"][a:b]
        protected = data["protected"][a:b]; wait = int(data["wait_ms"][condition])
        for geometry in GEOMETRIES:
            for scoring in SCORINGS:
                method = f"{geometry}_{scoring}"
                score_input = raw if scoring == "raw" else utility[geometry][a:b]
                ids, score = formal.ordered(raw, source, protected, score_input, None)
                matrix = iou32(boxes[geometry][a:b][ids], gt)
                match50 = greedy(matrix, np.arange(len(ids)), .5)
                match70 = greedy(matrix, np.arange(len(ids)), .7)
                row = aggregate[(wait, method)]; row["score"].append(score)
                row["tp50"].append((match50 >= 0).astype(np.int8))
                row["tp70"].append((match70 >= 0).astype(np.int8)); row["gt"] += len(gt)
                row["outputs"] += len(ids)
                record = saved[method]; record["candidate"].append((a + ids).astype(np.int64))
                record["score"].append(score.astype(np.float64))
                record["tp50"].append((match50 >= 0).astype(np.int8))
                record["tp70"].append((match70 >= 0).astype(np.int8)); record["offset"].append(record["offset"][-1] + len(ids))
    panels = []
    for (wait, method), row in sorted(aggregate.items()):
        panel = {"role": role, "arm": arm, "wait_ms": wait, "method": method,
                 "gt": int(row["gt"]), "outputs": int(row["outputs"])}
        scores = np.concatenate(row["score"])
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"])
            metric = voc_ap(tp.tolist(), (1 - tp).tolist(), scores.tolist(), row["gt"], True)
            panel[f"ap{threshold}"] = 100.0 * metric["ap"]
            panel[f"tp{threshold}"] = int(metric["tp"]); panel[f"fp{threshold}"] = int(metric["fp"])
        panels.append(panel)
    archive: dict[str, np.ndarray] = {
        "method_names": np.asarray(methods, dtype="U64"), "geometry_names": np.asarray(GEOMETRIES, dtype="U32"),
        "condition_offsets": data["condition_offsets"], "dataset_index": data["dataset_index"],
        "sequence": data["sequence"], "wait_ms": data["wait_ms"],
        "receiver_timestamp_us": data["receiver_timestamp_us"], "source": data["source"],
    }
    for geometry in GEOMETRIES:
        archive[f"{geometry}__boxes"] = boxes[geometry]
        archive[f"{geometry}__direct39d"] = direct[geometry]
    for method, record in saved.items():
        for field in ("candidate", "score", "tp50", "tp70"):
            archive[f"{method}__{field}"] = np.concatenate(record[field])
        archive[f"{method}__offset"] = np.asarray(record["offset"], np.int64)
    path = output / f"{arm}__{role}__predictions.npz"
    np.savez_compressed(path, **archive)
    return panels, evidence(path)


def clean_reference_panels(summary_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    summary = load_json(summary_path)
    result = {}
    for role, panels in summary["panels"].items():
        for panel in panels:
            if panel["method"] == "task_geometry_fresh39d_direct_u":
                result[(role, int(panel["wait_ms"]))] = panel
    return result


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    manifests = sensor_manifests(ROOT)
    fit = load_npz(ROOT / contract["inputs"]["clean_label_free_fit"]["path"])
    fit_constant = fit["huber"].astype(np.float64)
    readout = ROOT / contract["inputs"]["direct_readout_root"]
    maps = load_json(ROOT / contract["inputs"]["source_maps"]["path"])
    official = FreeAlignPoseCache()
    clean_boxes: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    all_panels, arm_summaries, predictions = [], {}, {}
    clean_checks: dict[str, Any] = {}
    task_reference = clean_reference_panels(ROOT / contract["inputs"]["clean_task_summary"]["path"])

    for arm in contract["arms"]:
        arm_summary = {"spec": ARM_SPECS[arm], "roles": {}}
        for role in FORMAL_ROLES:
            split = "train" if role == "mechanism_holdout14" else "val"
            dataset = PerturbedDataset(
                ROOT / contract["datasets"][split], ROOT / contract["history"][split], arm, manifests)
            data, frame = build_role(dataset, role)
            residual = infer_task(frame["feature"], contract)
            task_delta = frame["kf_delta"] + float(contract["task_alpha"]) * residual
            boxes, features, diagnostic = direct_features(
                dataset, data, frame, task_delta, fit_constant, official)
            key = (split, role)
            if arm == "clean":
                clean_boxes[key] = {name: value.copy() for name, value in boxes.items()}
                clean_checks[f"{role}_pool_max_abs"] = float(max(dataset.clean_errors.values(), default=0.0))
            recovery = recovery_stats(boxes, clean_boxes[key], data["source"])
            panels, prediction = evaluate_role(
                dataset, role, data, boxes, features, readout, maps, output, arm)
            all_panels.extend(panels); predictions[f"{arm}|{role}"] = prediction
            arm_summary["roles"][role] = {
                "conditions": int(len(data["dataset_index"])),
                "candidates": int(len(data["source"])),
                "pose_diagnostic": diagnostic,
                "geometry_recovery": recovery,
                "prediction": prediction,
            }
            if arm == "clean":
                table = {(int(row["wait_ms"]), row["method"]): row for row in panels}
                for wait in (0, 400):
                    observed = table[(wait, "current_task_frozen39d")]
                    expected = task_reference[(role, wait)]
                    clean_checks[f"{role}_{wait}ms_task_ap70_abs_error"] = abs(observed["ap70"] - expected["ap70"])
                    clean_checks[f"{role}_{wait}ms_task_ap50_abs_error"] = abs(observed["ap50"] - expected["ap50"])
        arm_summaries[arm] = arm_summary

    clean_pass = all(value <= 5e-6 for key, value in clean_checks.items() if key.endswith("pool_max_abs")) \
        and all(value <= 1e-12 for key, value in clean_checks.items() if key.endswith("abs_error"))
    if not clean_pass:
        raise RuntimeError(("clean reproduction failed", clean_checks))
    summary = {
        "schema": "v2v4real_pose_stress_cached_geometry_summary_v1",
        "status": "complete_producer_review_pending",
        "created_at": now(), "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": evidence(ROOT / contract["contract_path"]),
        "arms": contract["arms"], "arm_specs": ARM_SPECS,
        "methods": [f"{geometry}_{scoring}" for geometry in GEOMETRIES for scoring in SCORINGS],
        "clean_fit_huber_constant_xy_m": fit_constant.tolist(),
        "clean_reproduction": {"passed": clean_pass, "checks": clean_checks},
        "panels": all_panels, "arm_summaries": arm_summaries,
        "predictions": predictions,
        "freealign_unique_calls": len(official.values),
        "input_and_weight_identity": contract["inputs"],
        "actual_cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_process_seconds": time.process_time(),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "detector_or_backbone_forwards": 0, "head_updates": 0, "gpu_seconds": 0,
            "official_freealign_calls": len(official.values),
        },
        "boundary": "Full formal-role cached geometry/readout stress on frozen detector outputs. It is not a perturbed point-cloud/feature-fusion system run, hidden test, or complete-system latency measurement.",
    }
    recipe.atomic(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA must be disabled for cached stress")
    contract = load_json(args.contract)
    if contract.get("status") != "frozen_released_for_execution":
        raise RuntimeError("contract not released")
    if recipe.sha256(Path(__file__)) != contract["runner"]["sha256"]:
        raise RuntimeError("runner identity drift")
    for name, item in contract["inputs"].items():
        if isinstance(item, dict) and "sha256" in item:
            if recipe.sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(("input identity drift", name, item["path"]))
    summary = compute(contract, args.output)
    print(json.dumps({"status": summary["status"], "clean": summary["clean_reproduction"],
                      "actual_cost": summary["actual_cost"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
