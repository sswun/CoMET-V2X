#!/usr/bin/env python3
"""Fit and evaluate the current task-frame geometry recipe on V2X-ViT seed2024.

The detector, packed candidate datasets and 39D direct-utility readout are
frozen.  A new task head with the reviewed 24D structure is fit on fit24 only,
one global alpha is selected on calibration8 endpoints, and all four waits are
evaluated on hold14 and val21.  No detector/backbone forward is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / "preexperiment/research_autopilot"
BASELINES = ROOT / "preexperiment/research_baselines"
sys.path[:0] = [str(AUTO), str(BASELINES)]

import run_strict60_task_supervised_shared_correction_head_v1_20261028 as task
import run_shared_offset_optional_formal_v1_20261026 as formal
from e1_receiver_value_core_v1_20261020 import greedy, iou32
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap
from v2xvit_receiver_timestamp_adapter_v1_20261026 import TimestampedPackedDataset


ROLES = ("fit24", "calibration8", "mechanism_holdout14", "val21")
FORMAL_ROLES = ("mechanism_holdout14", "val21")
WAITS = (0, 100, 200, 400)
METHODS = (
    "kf_direct_u",
    "task_geometry_fixed_kf_readout",
    "task_geometry_fresh39d_direct_u",
)


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def evidence(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def exact(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def source_metadata(dataset: TimestampedPackedDataset) -> dict[str, tuple[int, int]]:
    """Map frozen sender-cache paths to (timestamp_us, packet record count)."""
    result: dict[str, tuple[int, int]] = {}
    for row in dataset.meta:
        path = row.get("source_raw_path")
        source = row.get("source")
        if not path:
            if source is not None:
                raise RuntimeError("source metadata without frozen source path")
            continue
        value = (int(source["source_timestamp_us"]), int(row["diagnostic"]["raw_remote"]))
        if path in result and result[path] != value:
            raise RuntimeError((path, result[path], value))
        result[path] = value
    missing = {str(row["previous_source_raw_path"]) for row in dataset.meta
               if row.get("previous_source_raw_path") and
               str(row["previous_source_raw_path"]) not in result}
    if missing:
        raise RuntimeError(("unresolved legal previous sender packets", sorted(missing)[:5]))
    return result


def build_role(dataset: TimestampedPackedDataset, role: str,
               make_targets: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    selected = [index for index, row in enumerate(dataset.meta) if row["role"] == role]
    selected.sort(key=lambda index: (
        str(dataset.meta[index]["sequence"]), int(dataset.meta[index]["wait_ms"]),
        int(dataset.meta[index]["receiver_timestamp_us"]),
    ))
    sender = source_metadata(dataset)
    states: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    previous_time: dict[tuple[str, int], int] = {}
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    offsets = [0]; gt_offsets = [0]
    features = np.zeros((len(selected), len(task.FEATURE_NAMES)), np.float64)
    kf_delta = np.zeros((len(selected), 2), np.float64)
    geometry_target = np.full((len(selected), 2), np.nan, np.float64)
    task_target = np.full((len(selected), 2), np.nan, np.float64)
    task_stats = np.zeros((len(selected), 8), np.float64)
    sequences, waits, timestamps = [], [], []
    max_age_error = 0.0
    for condition, index in enumerate(selected):
        meta = dataset.meta[index]; pool = dataset.pool(index)
        key = (str(meta["sequence"]), int(meta["wait_ms"]))
        timestamp = int(meta["receiver_timestamp_us"])
        if timestamp <= previous_time.get(key, -1):
            raise RuntimeError((key, "non-monotonic receiver timestamp"))
        previous_time[key] = timestamp
        ego, stale_id, cv_id = formal.identities(pool)
        ego_boxes = pool["boxes"][ego].copy()
        cv_remote = pool["boxes"][cv_id].copy()
        cv_boxes = np.concatenate((ego_boxes, cv_remote)).astype(np.float32)
        raw = np.r_[pool["raw_scores"][ego], pool["raw_scores"][stale_id]].astype(np.float64)
        source = np.r_[np.zeros(len(ego), np.int8), np.ones(len(stale_id), np.int8)]
        protected = np.r_[pool["is_protected"][ego], np.zeros(len(stale_id), np.int8)]
        protected_local = np.flatnonzero(pool["is_protected"][ego])
        protected_boxes = ego_boxes[protected_local]
        remote_anchor, protected_anchor = task.pairs(cv_remote, protected_boxes, 2.5, True)
        anchor = np.zeros(len(cv_boxes), np.int8); anchor[len(ego) + remote_anchor] = 1
        residual = (protected_boxes[protected_anchor, :2].astype(np.float64)
                    - cv_remote[remote_anchor, :2].astype(np.float64))
        prior_mean, variance = states.get(key, (np.zeros(2, np.float64), 1.0))
        prior_mean = prior_mean.copy(); prior_variance = float(variance) + .15 ** 2
        update = prior_mean.copy()
        if len(residual):
            gain = prior_variance / (prior_variance + .6 ** 2 / len(residual))
            update += gain * (residual.mean(axis=0) - update)
            variance = (1.0 - gain) * prior_variance
        else:
            variance = prior_variance
        states[key] = (update.copy(), float(variance)); kf_delta[condition] = update
        kf_boxes = cv_boxes.copy(); kf_boxes[len(ego):, :2] += update.astype(np.float32)

        mean = residual.mean(axis=0) if len(residual) else np.zeros(2)
        median = np.median(residual, axis=0) if len(residual) else np.zeros(2)
        std = residual.std(axis=0) if len(residual) else np.zeros(2)
        rms = float(np.sqrt(np.square(residual).sum(axis=1).mean())) if len(residual) else 0.0
        huber = task.robust_huber(residual)
        current_path = meta.get("source_raw_path")
        previous_path = meta.get("previous_source_raw_path")
        if current_path:
            current_timestamp, current_records = sender[str(current_path)]
            current_age = max(0.0, (timestamp - current_timestamp) / 1000.0)
            max_age_error = max(max_age_error, abs(current_age - float(meta["source"]["actual_age_ms"])))
        else:
            current_timestamp, current_records, current_age = 0, 0, 0.0
            if len(stale_id) or previous_path:
                raise RuntimeError((index, "empty sender packet has remote/history records"))
        if previous_path:
            history_timestamp, history_records = sender[str(previous_path)]
            history_age = max(0.0, (timestamp - history_timestamp) / 1000.0)
            interval = max(0.0, (current_timestamp - history_timestamp) / 1000.0)
        else:
            history_records = 0; history_age = 0.0; interval = 0.0
        remote_score = raw[source == 1]
        features[condition] = np.asarray([
            prior_mean[0], prior_mean[1], np.log(max(prior_variance, 1e-12)),
            update[0], update[1], mean[0], mean[1], median[0], median[1],
            huber[0], huber[1], std[0], std[1], rms,
            np.log1p(len(remote_anchor)), np.log1p(len(stale_id) - len(remote_anchor)),
            np.log1p(len(protected_local)), np.log1p(current_records),
            np.log1p(history_records), np.log1p(current_age), np.log1p(history_age),
            np.log1p(interval),
            float(remote_score.mean()) if len(remote_score) else 0.0,
            float(remote_score.std()) if len(remote_score) else 0.0,
        ])
        gt = pool["gt_boxes"].astype(np.float32)
        if make_targets:
            geometry_target[condition] = task.geometry_target(
                cv_boxes, np.flatnonzero(source == 1), gt, update, 2.0)
            task_target[condition], stats = task.task_target(
                cv_boxes, gt, raw, source, protected,
                set((len(ego) + remote_anchor).tolist()), update, prior_mean, 2.0)
            task_stats[condition] = [
                stats["ego_uncovered_targets"], stats["nonanchor_candidates"],
                stats["proposals"], stats["baseline_tp70"], stats["target_tp70"],
                stats["rescued"], stats["damaged"], stats["target_matched_iou"],
            ]
        for name, value in {
            "cv_boxes": cv_boxes, "kf_boxes": kf_boxes, "raw_scores": raw,
            "source": source, "protected": protected, "anchor": anchor,
            "gt_boxes": gt,
        }.items():
            arrays[name].append(value)
        offsets.append(offsets[-1] + len(cv_boxes)); gt_offsets.append(gt_offsets[-1] + len(gt))
        sequences.append(key[0]); waits.append(key[1]); timestamps.append(timestamp)
    flat = {name: np.concatenate(value) for name, value in arrays.items()}
    flat.update(
        condition_offsets=np.asarray(offsets, np.int64),
        gt_offsets=np.asarray(gt_offsets, np.int64),
        dataset_index=np.asarray(selected, np.int64),
        sequence=np.asarray(sequences, dtype="U16"),
        wait_ms=np.asarray(waits, np.int16),
        receiver_timestamp_us=np.asarray(timestamps, np.int64),
    )
    frame = {
        "feature": features, "kf_delta": kf_delta,
        "geometry_target": geometry_target, "task_target": task_target,
        "task_stats": task_stats, "dataset_index": flat["dataset_index"].copy(),
        "sequence": flat["sequence"].copy(), "wait_ms": flat["wait_ms"].copy(),
        "receiver_timestamp_us": flat["receiver_timestamp_us"].copy(),
        "max_source_age_error_ms": np.asarray(max_age_error),
    }
    if not np.isfinite(features).all() or max_age_error > 1e-9:
        raise RuntimeError(("invalid task frame", max_age_error))
    return flat, frame


def raw_panels(data: dict[str, np.ndarray], corrections: dict[str, np.ndarray],
               waits: tuple[int, ...] = (0, 400)) -> list[dict[str, Any]]:
    rows: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {"score": [], "tp": [], "gt": 0, "outputs": 0})
    for condition, wait in enumerate(data["wait_ms"]):
        if int(wait) not in waits:
            continue
        a, b = data["condition_offsets"][condition:condition + 2].astype(int)
        ga, gb = data["gt_offsets"][condition:condition + 2].astype(int)
        raw = data["raw_scores"][a:b]; source = data["source"][a:b]
        protected = data["protected"][a:b]; remote = np.flatnonzero(source == 1)
        gt = data["gt_boxes"][ga:gb]
        for method, delta in corrections.items():
            boxes = data["cv_boxes"][a:b].copy(); boxes[remote, :2] += delta[condition].astype(np.float32)
            ids, score = formal.ordered(raw, source, protected, raw, None)
            matched = greedy(iou32(boxes[ids], gt), np.arange(len(ids)), .7)
            row = rows[(int(wait), method)]; row["score"].append(score)
            row["tp"].append((matched >= 0).astype(np.int8)); row["gt"] += len(gt)
            row["outputs"] += len(ids)
    panels = []
    for (wait, method), row in sorted(rows.items()):
        score = np.concatenate(row["score"]); tp = np.concatenate(row["tp"])
        metric = voc_ap(tp.tolist(), (1 - tp).tolist(), score.tolist(), row["gt"], True)
        panels.append({"wait_ms": wait, "method": method, "ap70": 100.0 * metric["ap"],
                       "tp70": int(metric["tp"]), "fp70": int(metric["fp"]),
                       "gt": int(row["gt"]), "outputs": int(row["outputs"])})
    return panels


def formal_identity(data: dict[str, np.ndarray], material: dict[str, np.ndarray]) -> tuple[dict[str, bool], dict[str, float]]:
    checks = {}
    for key in ("condition_offsets", "dataset_index", "sequence", "wait_ms",
                "receiver_timestamp_us", "raw_scores", "source", "protected", "anchor", "cv_boxes"):
        checks[key + "_exact"] = exact(data[key], material[key])
    errors = {
        "kf_boxes": float(np.max(np.abs(data["kf_boxes"] - material["kf_boxes"]), initial=0.0)),
    }
    columns = {str(name): index for index, name in enumerate(material["trace_columns"])}
    trace_delta = material["trace"][:, [columns["kf_dx"], columns["kf_dy"]]]
    errors["kf_trace"] = float(np.max(np.abs(trace_delta - (
        data["kf_boxes"][data["condition_offsets"][:-1], :2] * 0.0)), initial=0.0)) if False else 0.0
    return checks, errors


def evaluate_formal(dataset: TimestampedPackedDataset, data: dict[str, np.ndarray],
                    frame: dict[str, np.ndarray], material: dict[str, np.ndarray],
                    residual: np.ndarray, alpha: float, readout: Path,
                    maps: dict[str, Any], output: Path, role: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, float]]:
    delta = frame["kf_delta"] + alpha * residual
    task_boxes = material["cv_boxes"].copy(); task_features = material["cv_features"].copy()
    for condition in range(len(material["dataset_index"])):
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        source = material["source"][a:b]; protected = material["protected"][a:b]
        remote = np.flatnonzero(source == 1); fixed = np.flatnonzero(protected == 1)
        task_boxes[a:b][remote, :2] += delta[condition].astype(np.float32)
        task_features[a:b][remote] = formal.update_features(
            material["cv_features"][a:b][remote], task_boxes[a:b][remote],
            material["cv_boxes"][a:b][fixed], material["raw_scores"][a:b][fixed])
    direct = formal.direct_scores({"kf": material["kf_features"], "task": task_features}, readout)
    mapping = maps["maps"]["direct_to_utility"]
    kf_score = formal.source_map(direct["kf"], material["source"], mapping)
    task_score = formal.source_map(direct["task"], material["source"], mapping)
    score_by_method = {
        "kf_direct_u": kf_score,
        "task_geometry_fixed_kf_readout": kf_score,
        "task_geometry_fresh39d_direct_u": task_score,
    }
    boxes_by_method = {
        "kf_direct_u": material["kf_boxes"],
        "task_geometry_fixed_kf_readout": task_boxes,
        "task_geometry_fresh39d_direct_u": task_boxes,
    }
    stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {"score": [], "tp50": [], "tp70": [], "gt": 0,
                 "outputs": 0, "remote": 0})
    saved = {method: {"candidate": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
             for method in METHODS}
    for condition, index in enumerate(material["dataset_index"]):
        pool = dataset.pool(int(index)); gt = pool["gt_boxes"]
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        raw = material["raw_scores"][a:b]; source = material["source"][a:b]
        protected = material["protected"][a:b]; wait = int(material["wait_ms"][condition])
        for method in METHODS:
            score = score_by_method[method][a:b]
            ids, rank = formal.ordered(raw, source, protected, score, None)
            matrix = iou32(boxes_by_method[method][a:b][ids], gt)
            match50 = greedy(matrix, np.arange(len(ids)), .5)
            match70 = greedy(matrix, np.arange(len(ids)), .7)
            row = stats[(wait, method)]; row["score"].append(rank)
            row["tp50"].append((match50 >= 0).astype(np.int8))
            row["tp70"].append((match70 >= 0).astype(np.int8)); row["gt"] += len(gt)
            row["outputs"] += len(ids); row["remote"] += int((source[ids] == 1).sum())
            record = saved[method]; record["candidate"].append((a + ids).astype(np.int64))
            record["score"].append(rank.astype(np.float64))
            record["tp50"].append((match50 >= 0).astype(np.int8))
            record["tp70"].append((match70 >= 0).astype(np.int8))
            record["offset"].append(record["offset"][-1] + len(ids))
    panels = []
    for (wait, method), row in sorted(stats.items()):
        panel = {"wait_ms": wait, "method": method, "gt": int(row["gt"]),
                 "outputs": int(row["outputs"]), "remote": int(row["remote"])}
        score = np.concatenate(row["score"])
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"])
            metric = voc_ap(tp.tolist(), (1 - tp).tolist(), score.tolist(), row["gt"], True)
            panel[f"ap{threshold}"] = 100.0 * metric["ap"]
            panel[f"tp{threshold}"] = int(metric["tp"]); panel[f"fp{threshold}"] = int(metric["fp"])
        panels.append(panel)
    archive: dict[str, np.ndarray] = {"method_names": np.asarray(METHODS, dtype="U48")}
    for method, record in saved.items():
        for field in ("candidate", "score", "tp50", "tp70"):
            archive[f"{method}__{field}"] = np.concatenate(record[field])
        archive[f"{method}__offset"] = np.asarray(record["offset"], np.int64)
    material_path = output / f"{role}_task_material.npz"
    np.savez_compressed(material_path, task_delta=delta, task_boxes=task_boxes,
                        task_features=task_features, kf_direct_u=kf_score,
                        task_direct_u=task_score, condition_offsets=material["condition_offsets"],
                        dataset_index=material["dataset_index"], sequence=material["sequence"],
                        wait_ms=material["wait_ms"],
                        receiver_timestamp_us=material["receiver_timestamp_us"])
    return panels, archive, {
        "task_delta_max_abs_m": float(np.max(np.abs(delta), initial=0.0)),
        "conditional_readout_max_abs": float(np.max(np.abs(task_score - kf_score), initial=0.0)),
        "material_bytes": float(material_path.stat().st_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA must be disabled")
    contract = load_json(args.contract)
    if contract.get("status") != "frozen_released_for_execution":
        raise RuntimeError("contract is not released")
    if sha256(Path(__file__)) != contract["runner"]["sha256"]:
        raise RuntimeError("runner identity drift")
    for item in contract["inputs"].values():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(("input identity drift", item["path"]))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    train = TimestampedPackedDataset(ROOT / contract["datasets"]["train"],
                                     ROOT / contract["vehicle_data_info"])
    val = TimestampedPackedDataset(ROOT / contract["datasets"]["val"],
                                   ROOT / contract["vehicle_data_info"])
    role_data: dict[str, dict[str, np.ndarray]] = {}
    role_frame: dict[str, dict[str, np.ndarray]] = {}
    for role in ROLES:
        role_data[role], role_frame[role] = build_role(
            train if role != "val21" else val, role, make_targets=role == "fit24")
        np.savez_compressed(args.output / f"{role}_frame_inputs.npz", **role_frame[role])

    fit = role_frame["fit24"]
    mean = fit["feature"].mean(axis=0); std = np.maximum(fit["feature"].std(axis=0), 1e-6)
    geometry_residual = fit["geometry_target"] - fit["kf_delta"]
    task_residual = fit["task_target"] - fit["kf_delta"]
    quantile = float(np.quantile(np.abs(np.r_[geometry_residual.ravel(), task_residual.ravel()]), .95))
    scale = float(min(2.0, max(.25, quantile)))
    seed = int(contract["training"]["seed"]); rng = np.random.default_rng(seed)
    batches = rng.integers(0, len(fit["feature"]), size=(
        int(contract["training"]["updates"]), int(contract["training"]["batch_size"])),
        dtype=np.int32)
    np.savez_compressed(args.output / "fit_protocol.npz",
                        feature_names=np.asarray(task.FEATURE_NAMES), mean=mean, std=std,
                        batches=batches, fit_scale=np.asarray(scale),
                        geometry_residual=geometry_residual, task_residual=task_residual)
    training = task.train_head("task_head", (fit["feature"] - mean) / std,
                               task_residual, batches, scale, seed, args.output)
    residuals: dict[str, np.ndarray] = {}
    for role in ROLES:
        residuals[role] = task.infer_head(
            "task_head", (role_frame[role]["feature"] - mean) / std,
            scale, args.output)
        np.savez_compressed(args.output / f"{role}_head_residuals.npz",
                            task_head=residuals[role])

    cal = role_data["calibration8"]; cal_frame = role_frame["calibration8"]
    calibration_rows = []
    for alpha in contract["calibration"]["alpha_grid"]:
        panels = raw_panels(cal, {"task_head": cal_frame["kf_delta"] + float(alpha) * residuals["calibration8"]})
        calibration_rows.append({"alpha": float(alpha),
                                 "mean_ap70": float(np.mean([row["ap70"] for row in panels])),
                                 "panels": panels})
    selected = max(calibration_rows, key=lambda row: (row["mean_ap70"], -row["alpha"]))
    calibration = {"schema": "v2xvit_seed2024_task_geometry_calibration_v1",
                   "role": "calibration8", "criterion": "mean raw-readout AP70 over 0/400ms",
                   "alpha_grid": contract["calibration"]["alpha_grid"],
                   "selected_alpha": selected["alpha"], "tables": calibration_rows,
                   "weight_updates": 0}
    atomic(args.output / "calibration.json", calibration)

    maps = load_json(ROOT / contract["inputs"]["fresh39d_source_maps"]["path"])
    formal_summary = load_json(ROOT / contract["inputs"]["formal_summary"]["path"])
    panels_by_role = {}; comparisons = []; identity_checks = {}; maximum_errors = {}
    output_evidence = {}
    for role in FORMAL_ROLES:
        material = load_npz(ROOT / contract["inputs"][f"{role}_formal_material"]["path"])
        checks, errors = formal_identity(role_data[role], material)
        identity_checks.update({f"{role}_{key}": value for key, value in checks.items()})
        maximum_errors.update({f"{role}_{key}": value for key, value in errors.items()})
        columns = {str(name): index for index, name in enumerate(material["trace_columns"])}
        trace_delta = material["trace"][:, [columns["kf_dx"], columns["kf_dy"]]]
        maximum_errors[f"{role}_kf_trace"] = float(np.max(np.abs(
            trace_delta - role_frame[role]["kf_delta"]), initial=0.0))
        panels, prediction, diagnostic = evaluate_formal(
            train if role == "mechanism_holdout14" else val,
            role_data[role], role_frame[role], material, residuals[role],
            float(selected["alpha"]), ROOT / contract["readout"], maps, args.output, role)
        panels_by_role[role] = panels
        prediction_path = args.output / f"{role}_predictions.npz"
        np.savez_compressed(prediction_path, **prediction)
        output_evidence[role] = {
            "frame_inputs": evidence(args.output / f"{role}_frame_inputs.npz"),
            "residuals": evidence(args.output / f"{role}_head_residuals.npz"),
            "task_material": evidence(args.output / f"{role}_task_material.npz"),
            "predictions": evidence(prediction_path), "diagnostic": diagnostic,
        }
        table = {(row["wait_ms"], row["method"]): row for row in panels}
        original = {(row["wait_ms"], row["method"]): row
                    for row in formal_summary["roles"][role]["panels"]}
        for wait in WAITS:
            baseline = table[(wait, "kf_direct_u")]
            old = original[(wait, "kf_direct_u")]
            maximum_errors[f"{role}_{wait}_baseline_ap70"] = abs(baseline["ap70"] - old["ap70"])
            fixed = table[(wait, "task_geometry_fixed_kf_readout")]
            fresh = table[(wait, "task_geometry_fresh39d_direct_u")]
            comparisons.append({
                "role": role, "wait_ms": wait,
                "system_delta_ap50_pp": fresh["ap50"] - baseline["ap50"],
                "system_delta_ap70_pp": fresh["ap70"] - baseline["ap70"],
                "geometry_fixed_readout_delta_ap70_pp": fixed["ap70"] - baseline["ap70"],
                "conditional_readout_delta_ap70_pp": fresh["ap70"] - fixed["ap70"],
                "system_delta_tp70": fresh["tp70"] - baseline["tp70"],
                "system_delta_fp70": fresh["fp70"] - baseline["fp70"],
            })

    target_stats = fit["task_stats"]
    checks = {
        "all_frozen_inputs_exact": True,
        "formal_candidate_identity_exact": all(identity_checks.values()),
        "formal_kf_geometry_exact": max(maximum_errors.values(), default=0.0) <= 5e-6,
        "only_fit24_supervised": all(
            (role == "fit24") == bool(np.isfinite(role_frame[role]["task_target"]).all())
            for role in ROLES),
        "single_global_calibration_alpha": selected["alpha"] in contract["calibration"]["alpha_grid"],
        "bounded_task_residual": all(float(np.max(np.abs(value), initial=0.0)) <= scale + 1e-6
                                     for value in residuals.values()),
        "all_four_formal_waits": all({row["wait_ms"] for row in panels_by_role[role]} == set(WAITS)
                                     for role in FORMAL_ROLES),
        "zero_detector_backbone_forward": True,
        "zero_gpu": True,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "identity": identity_checks,
                            "maximum_errors": maximum_errors})
    artifacts = {
        "fit_protocol": evidence(args.output / "fit_protocol.npz"),
        "task_head_weights": evidence(args.output / "task_head.pth"),
        "calibration": evidence(args.output / "calibration.json"),
    }
    for role in ("fit24", "calibration8"):
        artifacts[f"{role}_frame_inputs"] = evidence(args.output / f"{role}_frame_inputs.npz")
        artifacts[f"{role}_residuals"] = evidence(args.output / f"{role}_head_residuals.npz")
    summary = {
        "schema": "v2xvit_seed2024_task_geometry_fresh39d_producer_v1",
        "status": "complete_producer_review_pending", "created_at": now(),
        "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": evidence(args.contract), "detector": contract["detector"],
        "feature_names": list(task.FEATURE_NAMES), "training": training,
        "fit_scale_m": scale, "fit_scale_raw_q95_m": quantile,
        "calibration": calibration,
        "target_summary": {
            "conditions": int(len(target_stats)),
            "baseline_tp70_condition_sum": int(target_stats[:, 3].sum()),
            "target_tp70_condition_sum": int(target_stats[:, 4].sum()),
            "rescued_condition_sum": int(target_stats[:, 5].sum()),
            "damaged_condition_sum": int(target_stats[:, 6].sum()),
            "positive_conditions": int(np.count_nonzero(target_stats[:, 4] > target_stats[:, 3])),
            "negative_conditions": int(np.count_nonzero(target_stats[:, 4] < target_stats[:, 3])),
        },
        "panels": panels_by_role, "comparisons": comparisons,
        "checks": checks, "identity_checks": identity_checks,
        "maximum_errors": maximum_errors, "artifacts": artifacts,
        "formal_outputs": output_evidence,
        "actual_cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_process_seconds": time.process_time(),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "cpu_threads": 2, "task_head_updates": int(contract["training"]["updates"]),
            "detector_or_backbone_forwards": 0, "gpu_seconds": 0,
        },
        "boundary": (
            "V2X-ViT seed2024 cached formal pipeline on exposed SPD roles. The 24D head is "
            "refit on ViT fit24; this is a same-structure cross-backbone confirmation, not a "
            "TraF-weight transfer, hidden-test result, actual network packet experiment, complete-system "
            "cost, second detector seed, or final core claim."
        ),
    }
    atomic(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "selected_alpha": selected["alpha"],
                      "comparisons": comparisons, "actual_cost": summary["actual_cost"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
