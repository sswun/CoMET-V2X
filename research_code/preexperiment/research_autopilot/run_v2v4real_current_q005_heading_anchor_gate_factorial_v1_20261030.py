#!/usr/bin/env python3
"""V2V4Real current-recipe q005 motion/observation 2x2.

The state key is always the full U96 sequence from the reviewed CV material.
The old U16 frame archives are deliberately not inputs.  Candidate/GT
evaluation follows the historical V2 protocol and never re-applies common ROI
after moving a frozen candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / "preexperiment/research_autopilot"
BASE = ROOT / "preexperiment/research_baselines"
sys.path[:0] = [str(AUTO), str(BASE)]

import run_crosscheckpoint_current_temporal_fixed_readout_controls_v1_20261029 as fixed
import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
import run_crosscheckpoint_temporal_q005_task_refit_v1_20261029 as refit
import run_strict60_task_supervised_shared_correction_head_v1_20261028 as task
import run_v2v4real_task_geometry_fresh39d_v1_20261028 as v2
import run_v2xvit_seed2024_task_geometry_fresh39d_v1_20261028 as task_recipe
import run_shared_offset_optional_formal_v1_20261026 as formal
from object_history_selector_core_v1_20261021 import PackedDataset


TASK = "v2v4real_current_q005_heading_anchor_gate_factorial_v1_20261030"
PARENT = "autonomous_research_continuation_20261024"
OWNER = "research_autopilot_v2v4real_current_q005_heading_anchor_gate_factorial_v1"
CONTRACT_SCHEMA = "v2v4real_current_q005_heading_anchor_gate_factorial_contract_recovery_v5"
STATUS = "frozen_released_for_queue_owned_cpu_execution"
ROLES = ("fit24", "calibration8", "mechanism_holdout14", "val21")
FORMAL = ("mechanism_holdout14", "val21")
WAITS = (0, 400)
METHODS = ("cv_q005_base", "heading_q005_base", "cv_anchor_gate", "heading_anchor_gate")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def ev(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def exact(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA \
            or contract.get("task_id") != TASK \
            or contract.get("parent_task_id") != PARENT \
            or contract.get("owner") != OWNER \
            or contract.get("status") != STATUS:
        raise RuntimeError("contract identity/status")
    if sha(Path(__file__).resolve()) != contract["runner"]["sha256"]:
        raise RuntimeError("runner drift")
    forbidden = tuple(contract["input_policy"]["forbidden_old_u16_frame_archives"])
    input_paths = {item["path"] for item in contract["inputs"].values()}
    if any(path in input_paths for path in forbidden):
        raise RuntimeError("old U16 frame archive entered the input set")
    for item in contract["inputs"].values():
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise RuntimeError(("input drift", item["path"]))


def preflight(contract: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for role in ROLES:
        paths = contract["materials"][role]
        archives = {name: load_npz(ROOT / value) for name, value in paths.items()}
        cv = archives["cv"]
        checks[f"{role}_sequence_dtype_u96"] = cv["sequence"].dtype.str.endswith("U96")
        checks[f"{role}_sequence_count"] = int(len(np.unique(cv["sequence"])))
        checks[f"{role}_conditions"] = int(len(cv["dataset_index"]))
        checks[f"{role}_candidates"] = int(len(cv["source"]))
        checks[f"{role}_waits"] = sorted(map(int, np.unique(cv["wait_ms"])))
        common = ("raw_scores", "source", "protected", "anchor", "condition_offsets",
                  "dataset_index", "sequence", "wait_ms", "receiver_timestamp_us")
        for geometry, archive in archives.items():
            checks[f"{role}_{geometry}_identity"] = all(exact(cv[key], archive[key]) for key in common)
        expected = contract["expected_roles"][role]
        if (checks[f"{role}_sequence_count"] != expected["full_sequences"]
                or checks[f"{role}_conditions"] != expected["conditions"]
                or checks[f"{role}_candidates"] != expected["candidates"]
                or checks[f"{role}_waits"] != list(WAITS)):
            raise RuntimeError((role, "preflight population drift", checks))
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    if not all(boolean_checks):
        raise RuntimeError(("preflight identity", checks))
    return {
        "status": "passed_no_experiment_started",
        "task_id": TASK,
        "checks": checks,
        "old_u16_frame_archives_consumed": 0,
        "detector_or_backbone_forwards": 0,
        "optimizer_updates": 0,
        "gpu_seconds": 0,
    }


def merge_material(role: str, dataset: PackedDataset, data: dict[str, np.ndarray],
                   frame: dict[str, np.ndarray], spec: dict[str, str],
                   expected: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cv = load_npz(ROOT / spec["cv"])
    kf = load_npz(ROOT / spec["kf"])
    stale = load_npz(ROOT / spec["stale"])
    common = ("raw_scores", "source", "protected", "anchor", "condition_offsets",
              "dataset_index", "wait_ms", "receiver_timestamp_us")
    identity = {f"data_{key}_exact": exact(data[key], cv[key]) for key in common}
    for geometry, archive in (("kf", kf), ("stale", stale)):
        for key in (*common, "sequence"):
            identity[f"{geometry}_{key}_exact"] = exact(cv[key], archive[key])
    meta_sequence = np.asarray(
        [str(dataset.meta[int(index)]["sequence"]) for index in cv["dataset_index"]], dtype="U96")
    identity.update({
        "full_sequence_from_dataset_exact": exact(meta_sequence, cv["sequence"]),
        "full_sequence_dtype_u96": cv["sequence"].dtype.str.endswith("U96"),
        "full_sequence_count_exact": int(len(np.unique(cv["sequence"]))) == expected["full_sequences"],
        "conditions_exact": len(cv["dataset_index"]) == expected["conditions"],
        "candidates_exact": len(cv["source"]) == expected["candidates"],
        "gt_exact": len(data["gt_boxes"]) == expected["gt"],
        "cv_boxes_max_error": float(np.max(np.abs(data["cv_boxes"] - cv["boxes"]), initial=0.0)),
        "kf_boxes_max_error": float(np.max(np.abs(data["kf_boxes"] - kf["boxes"]), initial=0.0)),
    })
    anchors = np.asarray([
        int(cv["anchor"][a:b].sum())
        for a, b in zip(cv["condition_offsets"][:-1], cv["condition_offsets"][1:])
    ], np.int32)
    identity["anchor_present_exact"] = int(np.sum(anchors > 0)) == expected["anchor_present"]
    identity["no_anchor_exact"] = int(np.sum(anchors == 0)) == expected["no_anchor"]
    booleans = [value for value in identity.values() if isinstance(value, bool)]
    errors = [value for key, value in identity.items() if key.endswith("_max_error")]
    if not all(booleans) or max(errors, default=0.0) > 5e-6:
        raise RuntimeError((role, "authoritative material identity", identity))
    material = {
        **{key: cv[key].copy() for key in common},
        "sequence": cv["sequence"].copy(),
        "gt_boxes": data["gt_boxes"].copy(),
        "gt_offsets": data["gt_offsets"].copy(),
        "cv_boxes": cv["boxes"].copy(),
        "cv_features": cv["features"].copy(),
        "kf_boxes": kf["boxes"].copy(),
        "kf_features": kf["features"].copy(),
        "stale_boxes": stale["boxes"].copy(),
        "stale_features": stale["features"].copy(),
    }
    # The frame sequence is intentionally unsafe U16.  Only its non-state-key
    # feature matrix survives; the material U96 sequence owns every state key.
    if frame["sequence"].dtype.str.endswith("U96"):
        raise RuntimeError("expected historical in-memory U16 negative control")
    return material, identity


def q005_frame(feature_source: np.ndarray, material: dict[str, np.ndarray],
               make_targets: bool) -> dict[str, np.ndarray]:
    feature = feature_source.astype(np.float64).copy()
    anchors = np.rint(np.expm1(feature[:, 14])).astype(np.int64)
    measurements = feature[:, 5:7].copy()
    state: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    delta = np.zeros((len(feature), 2), np.float64)
    target = np.full((len(feature), 2), np.nan, np.float64)
    stats_array = np.zeros((len(feature), 8), np.float64)
    prior_mean_array = np.zeros((len(feature), 2), np.float64)
    prior_variance_array = np.zeros(len(feature), np.float64)
    posterior_variance_array = np.zeros(len(feature), np.float64)
    for condition, (sequence_value, wait_value) in enumerate(
            zip(material["sequence"], material["wait_ms"])):
        key = (str(sequence_value), int(wait_value))
        count = int(anchors[condition])
        mean, variance = state.get(key, (np.zeros(2, np.float64), 1.0))
        mean = mean.copy()
        prior_mean = mean.copy()
        prior = float(variance) + .05 ** 2
        feature[condition, :2] = prior_mean
        feature[condition, 2] = np.log(prior)
        prior_mean_array[condition] = prior_mean
        prior_variance_array[condition] = prior
        if count:
            gain = prior / (prior + .6 ** 2 / count)
            mean += gain * (measurements[condition] - mean)
            variance = (1.0 - gain) * prior
        else:
            variance = prior
        feature[condition, 3:5] = mean
        delta[condition] = mean
        posterior_variance_array[condition] = variance
        state[key] = (mean.copy(), float(variance))
        if make_targets:
            a, b = material["condition_offsets"][condition:condition + 2].astype(int)
            ga, gb = material["gt_offsets"][condition:condition + 2].astype(int)
            anchor_set = set(np.flatnonzero(material["anchor"][a:b] == 1).tolist())
            target[condition], stats = task.task_target(
                material["cv_boxes"][a:b], material["gt_boxes"][ga:gb],
                material["raw_scores"][a:b], material["source"][a:b],
                material["protected"][a:b], anchor_set, mean, prior_mean, 2.0)
            stats_array[condition] = [
                stats["ego_uncovered_targets"], stats["nonanchor_candidates"],
                stats["proposals"], stats["baseline_tp70"], stats["target_tp70"],
                stats["rescued"], stats["damaged"], stats["target_matched_iou"],
            ]
    return {
        "feature": feature,
        "q005_delta": delta,
        "task_target": target,
        "task_stats": stats_array,
        "prior_mean": prior_mean_array,
        "prior_variance": prior_variance_array,
        "posterior_variance": posterior_variance_array,
        "sequence": material["sequence"].copy(),
        "wait_ms": material["wait_ms"].copy(),
        "dataset_index": material["dataset_index"].copy(),
        "receiver_timestamp_us": material["receiver_timestamp_us"].copy(),
    }


def heading_geometry(material: dict[str, np.ndarray], delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes = material["cv_boxes"].copy()
    features = material["cv_features"].copy()
    for condition, move in enumerate(delta):
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        source = material["source"][a:b]
        protected = material["protected"][a:b]
        remote = a + np.flatnonzero(source == 1)
        anchors = a + np.flatnonzero((source == 0) & (protected == 1))
        stale = material["stale_boxes"][remote]
        cv = material["cv_boxes"][remote]
        yaw = stale[:, 6].astype(np.float64)
        unit = np.column_stack((np.cos(yaw), np.sin(yaw)))
        displacement = cv[:, :2].astype(np.float64) - stale[:, :2].astype(np.float64)
        along = np.einsum("ij,ij->i", displacement, unit)
        projected = cv.copy()
        projected[:, :2] = (stale[:, :2].astype(np.float64)
                            + unit * along[:, None] + move[None, :]).astype(np.float32)
        boxes[remote] = projected
        features[remote] = formal.update_features(
            material["cv_features"][remote], projected, material["cv_boxes"][anchors],
            material["raw_scores"][anchors])
    return boxes, features


def evaluate_v2(dataset: PackedDataset, material: dict[str, np.ndarray],
                boxes_by_method: dict[str, np.ndarray],
                scores_by_method: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Historical V2 evaluator: frozen population/GT, no post-shift ROI mask."""
    stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {"score": [], "tp50": [], "tp70": [], "gt": 0, "outputs": 0, "remote": 0})
    saved = {method: {"candidate": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
             for method in METHODS}
    for condition, dataset_index in enumerate(material["dataset_index"]):
        gt = dataset.pool(int(dataset_index))["gt_boxes"]
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        raw = material["raw_scores"][a:b]
        source = material["source"][a:b]
        protected = material["protected"][a:b]
        wait = int(material["wait_ms"][condition])
        if wait not in WAITS:
            raise RuntimeError(("unexpected V2 wait", wait))
        for method in METHODS:
            ids, rank = formal.ordered(raw, source, protected, scores_by_method[method][a:b], None)
            matrix = task_recipe.iou32(boxes_by_method[method][a:b][ids], gt)
            match50 = task_recipe.greedy(matrix, np.arange(len(ids)), .5)
            match70 = task_recipe.greedy(matrix, np.arange(len(ids)), .7)
            row = stats[(wait, method)]
            row["score"].append(rank.astype(np.float64))
            row["tp50"].append((match50 >= 0).astype(np.int8))
            row["tp70"].append((match70 >= 0).astype(np.int8))
            row["gt"] += len(gt)
            row["outputs"] += len(ids)
            row["remote"] += int((source[ids] == 1).sum())
            record = saved[method]
            record["candidate"].append((a + ids).astype(np.int64))
            record["score"].append(rank.astype(np.float64))
            record["tp50"].append((match50 >= 0).astype(np.int8))
            record["tp70"].append((match70 >= 0).astype(np.int8))
            record["offset"].append(record["offset"][-1] + len(ids))
    panels = []
    for (wait, method), row in sorted(stats.items()):
        score = np.concatenate(row["score"])
        panel = {"wait_ms": wait, "method": method, "gt": int(row["gt"]),
                 "outputs": int(row["outputs"]), "remote": int(row["remote"]),
                 "post_geometry_common_mask_applied": False}
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"])
            metric = task_recipe.voc_ap(
                tp.tolist(), (1 - tp).tolist(), score.tolist(), row["gt"], True)
            panel[f"ap{threshold}"] = 100.0 * float(metric["ap"])
            panel[f"tp{threshold}"] = int(metric["tp"])
            panel[f"fp{threshold}"] = int(metric["fp"])
        panels.append(panel)
    archive: dict[str, np.ndarray] = {
        "method_names": np.asarray(METHODS, dtype="U32"),
        "condition_wait_ms": material["wait_ms"].copy(),
        "condition_dataset_index": material["dataset_index"].copy(),
        "condition_sequence": material["sequence"].copy(),
    }
    for method, record in saved.items():
        for field in ("candidate", "score", "tp50", "tp70"):
            archive[f"{method}__{field}"] = np.concatenate(record[field])
        archive[f"{method}__offset"] = np.asarray(record["offset"], np.int64)
    return panels, archive


def same_fp(predictions: dict[str, np.ndarray], panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = {(int(row["wait_ms"]), row["method"]): row for row in panels}
    waits = predictions["condition_wait_ms"]
    rows: list[dict[str, Any]] = []
    for wait in WAITS:
        condition_ids = np.flatnonzero(waits == wait)
        for observation, cv_method, heading_method in (
                ("q005_base", "cv_q005_base", "heading_q005_base"),
                ("anchor_gate", "cv_anchor_gate", "heading_anchor_gate")):
            budget = int(table[(wait, cv_method)]["fp70"])
            for method in (cv_method, heading_method):
                offsets = predictions[f"{method}__offset"].astype(np.int64)
                selected = np.concatenate([
                    np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
                    for index in condition_ids])
                score = predictions[f"{method}__score"]
                tp = predictions[f"{method}__tp70"].astype(np.int64)
                ranked = selected[np.argsort(-score[selected], kind="stable")]
                cumulative_tp = np.cumsum(tp[ranked])
                cumulative_fp = np.cumsum(1 - tp[ranked])
                eligible = np.flatnonzero(cumulative_fp <= budget)
                last = int(eligible[-1]) if len(eligible) else -1
                rows.append({
                    "wait_ms": wait, "observation": observation, "method": method,
                    "reference": cv_method, "fp_budget": budget,
                    "achieved_fp": int(cumulative_fp[last]) if last >= 0 else 0,
                    "tp70_at_same_fp": int(cumulative_tp[last]) if last >= 0 else 0,
                    "outputs_at_same_fp": last + 1,
                })
    return rows


def effects(role: str, panels: list[dict[str, Any]], same_fp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = {(int(row["wait_ms"]), row["method"]): row for row in panels}
    sfp = {(int(row["wait_ms"]), row["method"]): row for row in same_fp_rows}
    rows = []
    for wait in WAITS:
        cb, hb = table[(wait, "cv_q005_base")], table[(wait, "heading_q005_base")]
        cg, hg = table[(wait, "cv_anchor_gate")], table[(wait, "heading_anchor_gate")]
        heading_base = float(hb["ap70"] - cb["ap70"])
        heading_gate = float(hg["ap70"] - cg["ap70"])
        rows.append({
            "role": role, "wait_ms": wait,
            "heading_at_base_ap50_pp": float(hb["ap50"] - cb["ap50"]),
            "heading_at_base_ap70_pp": heading_base,
            "heading_at_gate_ap50_pp": float(hg["ap50"] - cg["ap50"]),
            "heading_at_gate_ap70_pp": heading_gate,
            "gate_at_cv_ap70_pp": float(cg["ap70"] - cb["ap70"]),
            "gate_at_heading_ap70_pp": float(hg["ap70"] - hb["ap70"]),
            "interaction_ap70_pp": float(heading_gate - heading_base),
            "heading_at_base_tp70": int(hb["tp70"] - cb["tp70"]),
            "heading_at_base_fp70": int(hb["fp70"] - cb["fp70"]),
            "heading_at_gate_tp70": int(hg["tp70"] - cg["tp70"]),
            "heading_at_gate_fp70": int(hg["fp70"] - cg["fp70"]),
            "heading_at_base_samefp_tp70": int(
                sfp[(wait, "heading_q005_base")]["tp70_at_same_fp"]
                - sfp[(wait, "cv_q005_base")]["tp70_at_same_fp"]),
            "heading_at_gate_samefp_tp70": int(
                sfp[(wait, "heading_anchor_gate")]["tp70_at_same_fp"]
                - sfp[(wait, "cv_anchor_gate")]["tp70_at_same_fp"]),
        })
    return rows


def sign_summary(values: list[float]) -> dict[str, Any]:
    count = Counter("positive" if value > 0 else "negative" if value < 0 else "zero"
                    for value in values)
    return {"cells": len(values), "positive": int(count["positive"]),
            "negative": int(count["negative"]), "zero": int(count["zero"]),
            "mean_pp": float(np.mean(values)), "minimum_pp": float(np.min(values)),
            "maximum_pp": float(np.max(values))}


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic()
    cpu_started = time.process_time()
    output.mkdir(parents=True, exist_ok=False)
    train = PackedDataset(ROOT / contract["datasets"]["train"])
    val = PackedDataset(ROOT / contract["datasets"]["val"])
    role_data, role_frame = v2.build_roles(train, val)
    materials: dict[str, dict[str, np.ndarray]] = {}
    qframes: dict[str, dict[str, np.ndarray]] = {}
    input_identity: dict[str, Any] = {}
    state_audit: dict[str, Any] = {}
    for role in ROLES:
        dataset = train if role != "val21" else val
        material, identity = merge_material(
            role, dataset, role_data[role], role_frame[role], contract["materials"][role],
            contract["expected_roles"][role])
        materials[role] = material
        correct_q015, q015_errors = fixed.run_state(
            role_frame[role], material["sequence"], material["wait_ms"], "q015", {})
        unsafe_q015, _ = fixed.run_state(
            role_frame[role], role_frame[role]["sequence"], material["wait_ms"], "q015", {})
        correct_error = float(np.max(np.abs(correct_q015 - role_frame[role]["kf_delta"]), initial=0.0))
        unsafe_error = float(np.max(np.abs(unsafe_q015 - role_frame[role]["kf_delta"]), initial=0.0))
        if correct_error != 0.0 or max(q015_errors.values()) != 0.0:
            raise RuntimeError((role, "U96 q015 replay failed", correct_error, q015_errors))
        if unsafe_error < float(contract["acceptance"]["unsafe_u16_q015_min_error_m"]):
            raise RuntimeError((role, "U16 negative control unexpectedly safe", unsafe_error))
        qframes[role] = q005_frame(role_frame[role]["feature"], material, role == "fit24")
        difference = np.linalg.norm(qframes[role]["q005_delta"] - correct_q015, axis=1)
        input_identity[role] = identity
        state_audit[role] = {
            "state_key_dtype": str(material["sequence"].dtype),
            "full_sequences": int(len(np.unique(material["sequence"]))),
            "u96_q015_replay_max_error_m": correct_error,
            "unsafe_u16_q015_replay_max_error_m": unsafe_error,
            "q005_minus_q015_mean_norm_m": float(np.mean(difference)),
            "q005_minus_q015_max_norm_m": float(np.max(difference, initial=0.0)),
        }

    task_dir = output / "q005_task_head"
    task_dir.mkdir()
    # The predecessor fit protocol is schedule-only evidence here.  Do not
    # consume its q015 means, targets, residuals, or truncated state material.
    with np.load(ROOT / contract["training"]["task_batch_schedule"], allow_pickle=False) as archive:
        batches = archive["batches"].astype(np.int32).copy()
    fit = qframes["fit24"]
    if batches.shape != tuple(contract["training"]["expected_task_batch_shape"]) \
            or int(batches.max()) >= len(fit["feature"]):
        raise RuntimeError("task schedule/population drift")
    mean = fit["feature"].mean(axis=0)
    std = np.maximum(fit["feature"].std(axis=0), 1e-6)
    residual_target = fit["task_target"] - fit["q005_delta"]
    quantile = float(np.quantile(np.abs(residual_target.ravel()), .95))
    scale = float(min(2.0, max(.25, quantile)))
    protocol_path = task_dir / "fit_protocol.npz"
    np.savez_compressed(
        protocol_path, feature_names=np.asarray(task.FEATURE_NAMES), mean=mean, std=std,
        batches=batches, fit_scale=np.asarray(scale), task_residual=residual_target,
        q005_delta=fit["q005_delta"], task_target=fit["task_target"],
        state_key_sequence=fit["sequence"])
    task_training = task.train_head(
        "task_head", (fit["feature"] - mean) / std, residual_target, batches, scale,
        int(contract["training"]["task_seed"]), task_dir)
    residuals = {role: task.infer_head(
        "task_head", (qframes[role]["feature"] - mean) / std, scale, task_dir)
                 for role in ROLES}

    alpha_rows = []
    for alpha in contract["calibration"]["alpha_grid"]:
        panels = task_recipe.raw_panels(
            materials["calibration8"],
            {"q005_task_refit": qframes["calibration8"]["q005_delta"]
             + float(alpha) * residuals["calibration8"]}, WAITS)
        alpha_rows.append({"alpha": float(alpha),
                           "mean_ap70": float(np.mean([row["ap70"] for row in panels])),
                           "panels": panels})
    selected = max(alpha_rows, key=lambda row: (row["mean_ap70"], -row["alpha"]))
    calibration = {
        "role": "calibration8", "waits_ms": list(WAITS),
        "criterion": "unweighted mean historical-V2 raw AP70 over 0/400ms",
        "selected_alpha": selected["alpha"], "tables": alpha_rows,
    }
    atomic(task_dir / "calibration.json", calibration)

    q005_geometries: dict[str, dict[str, np.ndarray]] = {}
    for role in ROLES:
        boxes, features = refit.task_geometry(materials[role], qframes[role]["q005_delta"])
        q005_geometries[role] = matched.geometry_material(materials[role], boxes, features)
    readout_head = output / "q005_cv_matched39d_head"
    readout_training = matched.train_head(
        q005_geometries["fit24"], ROOT / contract["training"]["readout_schedule"], readout_head)
    direct_calibration = matched.score(q005_geometries["calibration8"], readout_head)
    matched.source_map(q005_geometries["calibration8"], direct_calibration, readout_head)
    old_readout_schedule = load_npz(ROOT / contract["training"]["readout_schedule"])
    new_readout_schedule = load_npz(readout_head / "training_schedule.npz")

    role_results: dict[str, Any] = {}
    all_effects: list[dict[str, Any]] = []
    for role in ROLES:
        counts = np.asarray([
            int(materials[role]["anchor"][a:b].sum())
            for a, b in zip(materials[role]["condition_offsets"][:-1],
                            materials[role]["condition_offsets"][1:])], np.int32)
        base_delta = qframes[role]["q005_delta"]
        refit_delta = base_delta + float(selected["alpha"]) * residuals[role]
        gate_delta = np.where((counts > 0)[:, None], refit_delta, base_delta)
        state_path = output / f"{role}_q005_state.npz"
        np.savez_compressed(
            state_path, q005_feature=qframes[role]["feature"], q005_delta=base_delta,
            q005_prior_mean=qframes[role]["prior_mean"],
            q005_prior_variance=qframes[role]["prior_variance"],
            q005_posterior_variance=qframes[role]["posterior_variance"],
            task_target=qframes[role]["task_target"], task_residual=residuals[role],
            q005_refit_delta=refit_delta, anchor_present_gate_delta=gate_delta,
            anchor_count=counts, condition_offsets=materials[role]["condition_offsets"],
            dataset_index=materials[role]["dataset_index"],
            state_key_sequence=materials[role]["sequence"], wait_ms=materials[role]["wait_ms"],
            receiver_timestamp_us=materials[role]["receiver_timestamp_us"])
        role_results[role] = {
            "conditions": int(len(counts)), "candidates": int(len(materials[role]["source"])),
            "anchor_present": int(np.sum(counts > 0)), "no_anchor": int(np.sum(counts == 0)),
            "state": ev(state_path),
        }
        if role not in FORMAL:
            continue
        cv_base_box, cv_base_feature = refit.task_geometry(materials[role], base_delta)
        cv_gate_box, cv_gate_feature = refit.task_geometry(materials[role], gate_delta)
        heading_base_box, heading_base_feature = heading_geometry(materials[role], base_delta)
        heading_gate_box, heading_gate_feature = heading_geometry(materials[role], gate_delta)
        boxes = {
            "cv_q005_base": cv_base_box, "heading_q005_base": heading_base_box,
            "cv_anchor_gate": cv_gate_box, "heading_anchor_gate": heading_gate_box,
        }
        features = {
            "cv_q005_base": cv_base_feature, "heading_q005_base": heading_base_feature,
            "cv_anchor_gate": cv_gate_feature, "heading_anchor_gate": heading_gate_feature,
        }
        scores = {method: refit.score_with_head(materials[role], value, readout_head)
                  for method, value in features.items()}
        dataset = train if role == "mechanism_holdout14" else val
        panels, predictions = evaluate_v2(dataset, materials[role], boxes, scores)
        prediction_path = output / f"{role}_factorial_predictions.npz"
        np.savez_compressed(prediction_path, **predictions)
        same_fp_rows = same_fp(predictions, panels)
        role_effects = effects(role, panels, same_fp_rows)
        all_effects.extend(role_effects)
        local = materials[role]["source"] == 0
        role_results[role].update({
            "panels": panels, "same_fp": same_fp_rows, "effects": role_effects,
            "predictions": ev(prediction_path),
            "checks": {
                "four_fixed_arms": {row["method"] for row in panels} == set(METHODS),
                "only_0_400": {int(row["wait_ms"]) for row in panels} == set(WAITS),
                "historical_v2_no_post_geometry_roi": all(
                    row["post_geometry_common_mask_applied"] is False for row in panels),
                "candidate_population_fixed": all(
                    len(value) == len(materials[role]["source"]) for value in boxes.values()),
                "protected_local_base_exact": exact(cv_base_box[local], heading_base_box[local]),
                "protected_local_gate_exact": exact(cv_gate_box[local], heading_gate_box[local]),
                "all_finite": all(np.isfinite(value).all()
                                  for value in (*boxes.values(), *scores.values())),
            },
        })

    checks = {
        "full_u96_state_keys_all_roles": all(
            value["state_key_dtype"] == "<U96" for value in state_audit.values()),
        "unsafe_u16_negative_control_nonzero_all_roles": all(
            value["unsafe_u16_q015_replay_max_error_m"]
            >= float(contract["acceptance"]["unsafe_u16_q015_min_error_m"])
            for value in state_audit.values()),
        "u96_q015_replay_exact_all_roles": all(
            value["u96_q015_replay_max_error_m"] == 0.0 for value in state_audit.values()),
        "q005_materially_differs_from_q015": all(
            value["q005_minus_q015_max_norm_m"]
            >= float(contract["acceptance"]["q005_q015_min_max_difference_m"])
            for value in state_audit.values()),
        "task_batch_schedule_exact": batches.shape == tuple(
            contract["training"]["expected_task_batch_shape"]),
        "task_structure_seed_updates_exact": task_training["parameters"] == 866
            and task_training["updates"] == 2000
            and task_training["seed"] == int(contract["training"]["task_seed"]),
        "single_alpha_calibration8": selected["alpha"] in contract["calibration"]["alpha_grid"],
        "readout_weights_schedule_exact": np.array_equal(
            old_readout_schedule["weights"], new_readout_schedule["weights"]),
        "readout_batches_schedule_exact": np.array_equal(
            old_readout_schedule["batches"], new_readout_schedule["batches"]),
        "readout_geometry_specific_normalization": True,
        "formal_roles_complete": set(role_results) == set(ROLES)
            and all("panels" in role_results[role] for role in FORMAL),
        "sixteen_primary_panels": sum(len(role_results[role]["panels"]) for role in FORMAL) == 16,
        "all_formal_checks": all(all(role_results[role]["checks"].values()) for role in FORMAL),
        "historical_v2_evaluator_primary": True,
        "zero_old_u16_frame_archive_inputs": True,
        "zero_detector_or_backbone_forwards": True,
        "zero_gpu": True,
    }
    if not all(checks.values()):
        raise RuntimeError(("acceptance checks", checks))
    summary = {
        "schema": "v2v4real_current_q005_heading_anchor_gate_factorial_producer_v1",
        "status": "complete_producer_review_pending", "created_at": contract["frozen_at"],
        "task_id": TASK, "contract": ev(ROOT / contract["contract_path"]),
        "recipe": {
            "state_key": "(full U96 sequence, wait_ms)", "process_variance": .05 ** 2,
            "measurement_variance": ".6^2/anchor_count", "no_anchor": "predict only",
            "heading": "stale_xy + h(stale_yaw)*dot(h,cv_xy-stale_xy) + observation_delta",
            "gate": "anchor_count>0 ? q005_refit_delta : q005_base_delta",
            "primary_readout": "one q005 CV-geometry-matched 39D head for all four arms",
            "evaluation": "historical V2 frozen candidate/GT protocol; no post-geometry common_mask",
        },
        "input_identity": input_identity, "state_audit": state_audit,
        "task_training": task_training,
        "task_fit_protocol": ev(protocol_path), "fit_scale_m": scale,
        "fit_scale_raw_q95_m": quantile, "calibration": calibration,
        "matched39d_training": readout_training,
        "matched39d_head": {
            "weights": ev(readout_head / "direct_u.pth"),
            "schedule": ev(readout_head / "training_schedule.npz"),
            "source_maps": ev(readout_head / "source_maps.json"),
        },
        "roles": role_results,
        "aggregate": {
            "heading_at_base_ap70": sign_summary(
                [row["heading_at_base_ap70_pp"] for row in all_effects]),
            "heading_at_gate_ap70": sign_summary(
                [row["heading_at_gate_ap70_pp"] for row in all_effects]),
            "gate_at_cv_ap70": sign_summary(
                [row["gate_at_cv_ap70_pp"] for row in all_effects]),
            "gate_at_heading_ap70": sign_summary(
                [row["gate_at_heading_ap70_pp"] for row in all_effects]),
            "factorial_interaction_ap70": sign_summary(
                [row["interaction_ap70_pp"] for row in all_effects]),
        },
        "checks": checks,
        "actual_cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_process_seconds": time.process_time() - cpu_started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "task_head_updates": 2000, "matched39d_head_updates": 2000,
            "detector_or_backbone_forwards": 0, "gpu_seconds": 0,
        },
        "boundary": (
            "Exposed V2V4Real cached-module factorial on frozen 0/400ms fit/cal/hold/val roles. "
            "No detector/backbone forward, hidden confirmation, per-cell selection, post-shift ROI "
            "filter, complete-system cost, or universal-domain claim."),
    }
    atomic(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "alpha": selected["alpha"],
                      "aggregate": summary["aggregate"], "actual_cost": summary["actual_cost"]},
                     sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA must be hidden")
    contract = load(args.contract)
    validate_contract(contract)
    if args.preflight_only:
        result = preflight(contract)
        result["contract"] = rel(args.contract)
        result["contract_sha256"] = sha(args.contract)
        print(json.dumps(result, sort_keys=True))
        return
    if args.output is None:
        raise RuntimeError("--output is required unless --preflight-only")
    compute(contract, args.output)


if __name__ == "__main__":
    main()
