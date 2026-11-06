#!/usr/bin/env python3
"""Fit-only geometry/task-supervised bounded residuals on actual strict60 packets.

The learned output is a frame-level XY residual added to the already effective
causal KF correction.  It is neither a per-box optional selector nor a CV/KF
frame gate.  GT is consumed only while materializing fit24 supervision and in
ordinary evaluation; every inference feature is reconstructed from arrived
packet detections, protected ego detections, timestamps, and causal KF state.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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
sys.path.insert(0, str(ROOT / "preexperiment/research_baselines"))

from audit_common_offset_peer_v1_20261026 import pairs  # noqa: E402
from audit_v2xseq_strict_common_roi_20261010 import mask as common_mask  # noqa: E402
from e1_receiver_value_core_v1_20261020 import greedy, iou32, unique_ego_scores  # noqa: E402
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap  # noqa: E402


ROLES = ("fit24", "calibration8", "mechanism_holdout14", "val21")
FORMAL_ROLES = ("mechanism_holdout14", "val21")
WAITS = (0, 400)
FEATURE_NAMES = (
    "prior_dx", "prior_dy", "log_prior_variance", "kf_dx", "kf_dy",
    "anchor_mean_dx", "anchor_mean_dy", "anchor_median_dx", "anchor_median_dy",
    "anchor_huber_dx", "anchor_huber_dy", "anchor_std_dx", "anchor_std_dy",
    "anchor_residual_rms", "log1p_anchor_count", "log1p_nonanchor_count",
    "log1p_protected_count", "log1p_current_packet_records",
    "log1p_history_packet_records", "log1p_current_age_ms",
    "log1p_history_age_ms", "log1p_source_interval_ms",
    "remote_score_mean", "remote_score_std",
)
ARMS = ("fit_task_constant", "geometry_head", "task_head")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def packet_records(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                digest = str(row["payload_sha256"])
                old = result.get(digest)
                identity = (int(row["records"]), int(row["source_timestamp_us"]))
                if old is not None and identity != (
                        int(old["records"]), int(old["source_timestamp_us"])):
                    raise RuntimeError("packet digest maps to inconsistent metadata")
                result[digest] = row
    return result


def robust_huber(residual: np.ndarray) -> np.ndarray:
    if not len(residual):
        return np.zeros(2, np.float64)
    center = np.median(residual, axis=0).astype(np.float64)
    for _ in range(4):
        difference = residual - center
        norm = np.linalg.norm(difference, axis=1)
        weight = np.minimum(1.0, 1.0 / np.maximum(norm, 1e-12))
        center = (residual * weight[:, None]).sum(axis=0) / weight.sum()
    return center


def raw_prediction(boxes: np.ndarray, gt: np.ndarray, raw: np.ndarray,
                   source: np.ndarray, protected: np.ndarray,
                   ego_only: bool = False) -> dict[str, np.ndarray]:
    if ego_only:
        candidate = np.flatnonzero(source == 0)
        score = unique_ego_scores(raw[candidate], stable=True)
    else:
        fixed = np.flatnonzero(protected == 1)
        variable = np.flatnonzero(protected == 0)
        candidate = np.r_[fixed, variable]
        score = np.r_[2.0 + unique_ego_scores(raw[fixed], stable=True),
                      unique_ego_scores(raw[variable], stable=True)]
    order = np.argsort(-score, kind="stable")
    candidate, score = candidate[order], score[order]
    keep = common_mask(boxes[candidate])
    candidate, score = candidate[keep], score[keep]
    gt_roi = gt[common_mask(gt)]
    matrix = iou32(boxes[candidate], gt_roi)
    match50 = greedy(matrix, np.arange(len(candidate), dtype=np.int64), .5)
    match70 = greedy(matrix, np.arange(len(candidate), dtype=np.int64), .7)
    return {
        "candidate": candidate.astype(np.int64), "score": score.astype(np.float64),
        "tp50": (match50 >= 0).astype(np.int8),
        "tp70": (match70 >= 0).astype(np.int8),
        "matched70": match70.astype(np.int32),
    }


def nonanchor_task_matches(boxes: np.ndarray, score: np.ndarray,
                           gt: np.ndarray, target_ids: np.ndarray) -> tuple[set[int], float]:
    if not len(boxes) or not len(target_ids):
        return set(), 0.0
    keep = common_mask(boxes)
    boxes, score = boxes[keep], score[keep]
    order = np.argsort(-unique_ego_scores(score, stable=True), kind="stable")
    boxes = boxes[order]
    matrix = iou32(boxes, gt[target_ids])
    matched = greedy(matrix, np.arange(len(boxes), dtype=np.int64), .7)
    valid = np.flatnonzero(matched >= 0)
    identities = {int(target_ids[int(matched[index])]) for index in valid}
    quality = float(sum(float(matrix[index, int(matched[index])]) for index in valid))
    return identities, quality


def geometry_target(cv: np.ndarray, remote: np.ndarray, gt_roi: np.ndarray,
                    kf_delta: np.ndarray, residual_bound: float) -> np.ndarray:
    a, b = pairs(cv[remote], gt_roi, 3.0, False)
    if not len(a):
        return kf_delta.copy()
    target = (gt_roi[b, :2].astype(np.float64)
              - cv[remote[a], :2].astype(np.float64)).mean(axis=0)
    return kf_delta + np.clip(target - kf_delta, -residual_bound, residual_bound)


def task_target(cv: np.ndarray, gt_roi: np.ndarray, raw: np.ndarray,
                source: np.ndarray, protected: np.ndarray, anchors: set[int],
                kf_delta: np.ndarray, prior_delta: np.ndarray,
                residual_bound: float) -> tuple[np.ndarray, dict[str, int | float]]:
    ego = raw_prediction(cv, gt_roi, raw, source, protected, ego_only=True)
    covered = {int(value) for value in ego["matched70"] if int(value) >= 0}
    uncovered = np.asarray([index for index in range(len(gt_roi)) if index not in covered],
                           np.int64)
    remote = np.flatnonzero(source == 1)
    nonanchor = np.asarray([index for index in remote if int(index) not in anchors], np.int64)
    baseline_boxes = cv[nonanchor].copy()
    baseline_boxes[:, :2] += kf_delta.astype(np.float32)
    baseline, baseline_quality = nonanchor_task_matches(
        baseline_boxes, raw[nonanchor], gt_roi, uncovered)
    proposals = [kf_delta.copy(), prior_delta.copy(), np.zeros(2, np.float64)]
    if len(nonanchor) and len(uncovered):
        a, b = pairs(cv[nonanchor], gt_roi[uncovered], 4.0, False)
        residual = (gt_roi[uncovered[b], :2].astype(np.float64)
                    - cv[nonanchor[a], :2].astype(np.float64))
        proposals.extend(value.copy() for value in residual)
        if len(residual):
            proposals.extend((residual.mean(axis=0), np.median(residual, axis=0)))
    bounded, seen = [], set()
    for proposal in proposals:
        proposal = kf_delta + np.clip(np.asarray(proposal) - kf_delta,
                                     -residual_bound, residual_bound)
        key = tuple(np.round(proposal, 7).tolist())
        if key not in seen:
            seen.add(key); bounded.append(proposal)
    best = None
    for ordinal, proposal in enumerate(bounded):
        boxes = cv[nonanchor].copy(); boxes[:, :2] += proposal.astype(np.float32)
        matched, quality = nonanchor_task_matches(boxes, raw[nonanchor], gt_roi, uncovered)
        preserved = len(matched & baseline)
        rescued = len(matched - baseline)
        damaged = len(baseline - matched)
        criterion = (len(matched), preserved, quality,
                     -float(np.square(proposal - kf_delta).sum()), -ordinal)
        if best is None or criterion > best[0]:
            best = (criterion, proposal.copy(), matched, rescued, damaged, quality)
    assert best is not None
    return best[1], {
        "ego_uncovered_targets": int(len(uncovered)),
        "nonanchor_candidates": int(len(nonanchor)),
        "proposals": int(len(bounded)),
        "baseline_tp70": int(len(baseline)),
        "target_tp70": int(len(best[2])),
        "rescued": int(best[3]), "damaged": int(best[4]),
        "baseline_matched_iou": baseline_quality,
        "target_matched_iou": float(best[5]),
    }


def role_material(path: Path, packets: dict[str, dict[str, Any]],
                  residual_bound: float, make_targets: bool) -> dict[str, np.ndarray]:
    data = load(path)
    conditions = len(data["dataset_index"])
    feature = np.zeros((conditions, len(FEATURE_NAMES)), np.float64)
    kf_delta = np.zeros((conditions, 2), np.float64)
    geometry = np.full((conditions, 2), np.nan, np.float64)
    task = np.full((conditions, 2), np.nan, np.float64)
    task_stats = np.zeros((conditions, 8), np.float64)
    state: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    last_time: dict[tuple[str, int], int] = {}
    max_kf_error = 0.0
    for condition in range(conditions):
        a, b = data["condition_offsets"][condition:condition + 2].astype(int)
        ga, gb = data["gt_offsets"][condition:condition + 2].astype(int)
        cv, saved_kf = data["cv_boxes"][a:b], data["kf_boxes"][a:b]
        raw, source = data["raw_scores"][a:b], data["source"][a:b]
        protected = data["protected"][a:b]
        gt_roi = data["gt_boxes"][ga:gb][common_mask(data["gt_boxes"][ga:gb])]
        sequence, wait = str(data["sequence"][condition]), int(data["wait_ms"][condition])
        key = (sequence, wait)
        timestamp = int(data["receiver_timestamp_us"][condition])
        if timestamp <= last_time.get(key, -1):
            raise RuntimeError((key, "non-monotonic receiver time"))
        last_time[key] = timestamp
        remote, fixed = np.flatnonzero(source == 1), np.flatnonzero(protected == 1)
        ar, ae = pairs(cv[remote], cv[fixed], 2.5, True)
        anchor_indices = set(remote[ar].tolist())
        residual = (cv[fixed[ae], :2].astype(np.float64)
                    - cv[remote[ar], :2].astype(np.float64))
        prior_mean, variance = state.get(key, (np.zeros(2, np.float64), 1.0))
        prior_mean = prior_mean.copy(); prior_variance = float(variance) + .15 ** 2
        update = prior_mean.copy()
        if len(residual):
            gain = prior_variance / (prior_variance + .6 ** 2 / len(residual))
            update += gain * (residual.mean(axis=0) - update)
            variance = (1.0 - gain) * prior_variance
        else:
            variance = prior_variance
        state[key] = (update.copy(), float(variance)); kf_delta[condition] = update
        if len(remote):
            max_kf_error = max(max_kf_error, float(np.max(np.abs(
                cv[remote, :2].astype(np.float64) + update - saved_kf[remote, :2]))))
        mean = residual.mean(axis=0) if len(residual) else np.zeros(2)
        median = np.median(residual, axis=0) if len(residual) else np.zeros(2)
        std = residual.std(axis=0) if len(residual) else np.zeros(2)
        rms = float(np.sqrt(np.square(residual).sum(axis=1).mean())) if len(residual) else 0.0
        huber = robust_huber(residual)
        current_digest = str(data["packet_sha256"][condition])
        current = packets.get(current_digest)
        history = packets.get(str(data["history_packet_sha256"][condition]))
        current_records = int(data["packet_records"][condition])
        if current is None:
            if current_digest or current_records != 0:
                raise RuntimeError((condition, "unresolved nonempty current packet"))
            current_age = 0.0
        else:
            if int(current["records"]) != current_records:
                raise RuntimeError((condition, "current packet record count drift"))
            current_age = max(0.0, (timestamp - int(current["source_timestamp_us"])) / 1000.0)
        history_age = (max(0.0, (timestamp - int(history["source_timestamp_us"])) / 1000.0)
                       if history is not None else 0.0)
        interval = (max(0.0, (int(current["source_timestamp_us"])
                              - int(history["source_timestamp_us"])) / 1000.0)
                    if current is not None and history is not None else 0.0)
        remote_score = raw[remote]
        nonanchors = len(remote) - len(anchor_indices)
        feature[condition] = np.asarray([
            prior_mean[0], prior_mean[1], np.log(max(prior_variance, 1e-12)),
            update[0], update[1], mean[0], mean[1], median[0], median[1],
            huber[0], huber[1], std[0], std[1], rms,
            np.log1p(len(anchor_indices)), np.log1p(nonanchors), np.log1p(len(fixed)),
            np.log1p(current_records),
            np.log1p(int(history["records"]) if history is not None else 0),
            np.log1p(current_age), np.log1p(history_age), np.log1p(interval),
            float(remote_score.mean()) if len(remote_score) else 0.0,
            float(remote_score.std()) if len(remote_score) else 0.0,
        ])
        if make_targets:
            geometry[condition] = geometry_target(
                cv, remote, gt_roi, update, residual_bound)
            task[condition], stats = task_target(
                cv, gt_roi, raw, source, protected, anchor_indices,
                update, prior_mean, residual_bound)
            task_stats[condition] = [
                stats["ego_uncovered_targets"], stats["nonanchor_candidates"],
                stats["proposals"], stats["baseline_tp70"], stats["target_tp70"],
                stats["rescued"], stats["damaged"], stats["target_matched_iou"],
            ]
    if not (np.isfinite(feature).all() and max_kf_error <= 5e-6):
        raise RuntimeError(("feature or KF reconstruction failure", max_kf_error))
    return {"feature": feature, "kf_delta": kf_delta,
            "geometry_target": geometry, "task_target": task,
            "task_stats": task_stats, "max_kf_error": np.asarray(max_kf_error),
            "dataset_index": data["dataset_index"].copy(),
            "sequence": data["sequence"].copy(), "wait_ms": data["wait_ms"].copy()}


def train_head(name: str, normalized: np.ndarray, residual: np.ndarray,
               batches: np.ndarray, scale: float, seed: int, output: Path) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(2); torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.ReLU(), nn.Linear(32, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    x = torch.from_numpy(normalized.astype(np.float32))
    y = torch.from_numpy(np.clip(residual / scale, -1.0, 1.0).astype(np.float32))
    trace, recent = [], []
    started = time.monotonic()
    for update, batch in enumerate(batches, 1):
        ids = torch.from_numpy(batch.astype(np.int64))
        prediction = torch.tanh(model(x[ids]))
        loss = F.smooth_l1_loss(prediction, y[ids], beta=.1)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError((name, update, "non-finite loss"))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        recent.append(float(loss.detach()))
        if update % 250 == 0:
            trace.append({"update": update, "mean_loss_last250": float(np.mean(recent[-250:]))})
    weights = output / f"{name}.pth"; torch.save(model.state_dict(), weights)
    return {"name": name, "input_dim": len(FEATURE_NAMES), "hidden_dim": 32,
            "output_dim": 2, "parameters": int(sum(p.numel() for p in model.parameters())),
            "seed": seed, "updates": int(len(batches)), "batch_size": int(batches.shape[1]),
            "optimizer": "Adam(lr=0.003)", "loss": "SmoothL1(beta=0.1)",
            "output": "fit_scale*tanh(logits)", "fit_scale_m": scale,
            "trace": trace, "wall_seconds": time.monotonic() - started,
            "weights": evidence(weights)}


def infer_head(name: str, normalized: np.ndarray, scale: float, output: Path) -> np.ndarray:
    import torch
    from torch import nn

    model = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.ReLU(), nn.Linear(32, 2))
    model.load_state_dict(torch.load(output / f"{name}.pth", map_location="cpu"), strict=True)
    model.eval(); values = []
    with torch.no_grad():
        tensor = torch.from_numpy(normalized.astype(np.float32))
        for begin in range(0, len(tensor), 65536):
            values.append((scale * torch.tanh(model(tensor[begin:begin + 65536]))).numpy())
    return np.concatenate(values).astype(np.float64)


def evaluate(data: dict[str, np.ndarray], corrections: dict[str, np.ndarray],
             save: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    collected = {method: {"candidate": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
                 for method in corrections}
    gt_count = defaultdict(int)
    for condition in range(len(data["dataset_index"])):
        a, b = data["condition_offsets"][condition:condition + 2].astype(int)
        ga, gb = data["gt_offsets"][condition:condition + 2].astype(int)
        cv, raw = data["cv_boxes"][a:b], data["raw_scores"][a:b]
        source, protected = data["source"][a:b], data["protected"][a:b]
        gt = data["gt_boxes"][ga:gb]; wait = int(data["wait_ms"][condition])
        gt_count[wait] += int(common_mask(gt).sum())
        remote = np.flatnonzero(source == 1)
        for method, delta in corrections.items():
            boxes = cv.copy(); boxes[remote, :2] += delta[condition].astype(np.float32)
            row = raw_prediction(boxes, gt, raw, source, protected)
            saved = collected[method]
            saved["candidate"].append(a + row["candidate"])
            for field in ("score", "tp50", "tp70"):
                saved[field].append(row[field])
            saved["offset"].append(saved["offset"][-1] + len(row["candidate"]))
    panels, archive = [], {}
    for method, values in collected.items():
        offset = np.asarray(values["offset"], np.int64)
        candidate = np.concatenate(values["candidate"])
        score = np.concatenate(values["score"])
        tp50 = np.concatenate(values["tp50"]); tp70 = np.concatenate(values["tp70"])
        archive[f"{method}__candidate"] = candidate
        archive[f"{method}__score"] = score
        archive[f"{method}__tp50"] = tp50
        archive[f"{method}__tp70"] = tp70
        archive[f"{method}__offset"] = offset
        for wait in WAITS:
            conditions = np.flatnonzero(data["wait_ms"] == wait)
            positions = np.concatenate([np.arange(offset[i], offset[i + 1], dtype=np.int64)
                                        for i in conditions])
            row = {"method": method, "wait_ms": wait, "gt": int(gt_count[wait]),
                   "outputs": int(len(positions))}
            for threshold, tp in ((50, tp50), (70, tp70)):
                metric = voc_ap(tp[positions].tolist(), (1 - tp[positions]).tolist(),
                                score[positions].tolist(), int(gt_count[wait]), True)
                row[f"ap{threshold}"] = float(metric["ap"] * 100.0)
                if threshold == 70:
                    row["tp70"] = int(metric["tp"]); row["fp70"] = int(metric["fp"])
            panels.append(row)
    if save is not None:
        np.savez_compressed(save, **archive)
    return panels, archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA must be disabled")
    contract = json.loads(args.contract.read_text())
    if contract["status"] != "frozen_released_for_execution":
        raise RuntimeError("contract is not released")
    if sha256(Path(__file__)) != contract["runner"]["sha256"]:
        raise RuntimeError("runner drift")
    for item in contract["inputs"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(item["path"])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    packets = packet_records(ROOT / contract["inputs"]["packet_index"]["path"])
    residual_bound = float(contract["method"]["target_residual_bound_m"])
    role_source, role_cache = {}, {}
    for role in ROLES:
        source_path = ROOT / contract["inputs"][f"{role}_candidates"]["path"]
        role_source[role] = load(source_path)
        role_cache[role] = role_material(source_path, packets, residual_bound,
                                         make_targets=role == "fit24")
        np.savez_compressed(args.output / f"{role}_frame_inputs.npz", **role_cache[role])
    fit = role_cache["fit24"]
    mean = fit["feature"].mean(axis=0)
    std = np.maximum(fit["feature"].std(axis=0), 1e-6)
    geometry_residual = fit["geometry_target"] - fit["kf_delta"]
    task_residual = fit["task_target"] - fit["kf_delta"]
    quantile = float(np.quantile(np.abs(np.r_[geometry_residual.ravel(),
                                           task_residual.ravel()]), .95))
    scale = float(min(residual_bound, max(.25, quantile)))
    constant = task_residual.mean(axis=0)
    seed = int(contract["training"]["seed"])
    rng = np.random.default_rng(seed)
    batches = rng.integers(0, len(fit["feature"]), size=(
        int(contract["training"]["updates"]), int(contract["training"]["batch_size"])),
        dtype=np.int32)
    np.savez_compressed(args.output / "fit_protocol.npz", feature_names=np.asarray(FEATURE_NAMES),
                        mean=mean, std=std, batches=batches, fit_scale=np.asarray(scale),
                        fit_task_constant_residual=constant,
                        geometry_residual=geometry_residual, task_residual=task_residual)
    normalized_fit = (fit["feature"] - mean) / std
    training = {
        "geometry_head": train_head("geometry_head", normalized_fit, geometry_residual,
                                    batches, scale, seed, args.output),
        "task_head": train_head("task_head", normalized_fit, task_residual,
                                batches, scale, seed, args.output),
    }
    residuals: dict[str, dict[str, np.ndarray]] = {}
    for role in ROLES:
        normalized = (role_cache[role]["feature"] - mean) / std
        residuals[role] = {
            "fit_task_constant": np.repeat(constant[None], len(normalized), axis=0),
            "geometry_head": infer_head("geometry_head", normalized, scale, args.output),
            "task_head": infer_head("task_head", normalized, scale, args.output),
        }
        np.savez_compressed(args.output / f"{role}_head_residuals.npz", **residuals[role])
    alpha_grid = [float(value) for value in contract["calibration"]["alpha_grid"]]
    selected, calibration_tables = {}, {}
    cal = role_cache["calibration8"]
    for arm in ARMS:
        candidates = []
        for alpha in alpha_grid:
            corrections = {arm: cal["kf_delta"] + alpha * residuals["calibration8"][arm]}
            panels, _ = evaluate(role_source["calibration8"], corrections)
            average = float(np.mean([row["ap70"] for row in panels]))
            candidates.append({"alpha": alpha, "mean_ap70": average, "panels": panels})
        chosen = max(candidates, key=lambda row: (row["mean_ap70"], -row["alpha"]))
        selected[arm] = float(chosen["alpha"]); calibration_tables[arm] = candidates
    calibration = {"schema": "task_supervised_shared_correction_calibration_v1",
                   "role": "calibration8", "criterion": "mean AP70 over frozen 0/400ms",
                   "alpha_grid": alpha_grid, "selected_alpha": selected,
                   "tables": calibration_tables, "no_weight_updates": True}
    atomic(args.output / "calibration.json", calibration)
    formal = {}
    for role in FORMAL_ROLES:
        base_delta = role_cache[role]["kf_delta"]
        corrections = {"kf": base_delta}
        for arm in ARMS:
            corrections[f"{arm}_alpha1"] = base_delta + residuals[role][arm]
            corrections[f"{arm}_calibrated"] = (
                base_delta + selected[arm] * residuals[role][arm])
        panels, _ = evaluate(role_source[role], corrections,
                             args.output / f"{role}_predictions.npz")
        formal[role] = panels
    task_stats = fit["task_stats"]
    target_summary = {
        "fit_conditions": int(len(task_stats)),
        "ego_uncovered_targets": int(task_stats[:, 0].sum()),
        "nonanchor_candidates": int(task_stats[:, 1].sum()),
        "proposals": int(task_stats[:, 2].sum()),
        "baseline_tp70_condition_sum": int(task_stats[:, 3].sum()),
        "target_tp70_condition_sum": int(task_stats[:, 4].sum()),
        "rescued_condition_sum": int(task_stats[:, 5].sum()),
        "damaged_condition_sum": int(task_stats[:, 6].sum()),
        "conditions_with_positive_task_delta": int(np.count_nonzero(task_stats[:, 4] > task_stats[:, 3])),
        "conditions_with_negative_task_delta": int(np.count_nonzero(task_stats[:, 4] < task_stats[:, 3])),
    }
    artifacts = ["fit_protocol.npz", "geometry_head.pth", "task_head.pth",
                 "calibration.json"]
    artifacts += [f"{role}_frame_inputs.npz" for role in ROLES]
    artifacts += [f"{role}_head_residuals.npz" for role in ROLES]
    artifacts += [f"{role}_predictions.npz" for role in FORMAL_ROLES]
    checks = {
        "all_frozen_inputs_exact": True,
        "reviewed_decomposition_is_input": True,
        "only_fit24_has_supervised_targets": all(
            (role == "fit24") == bool(np.isfinite(role_cache[role]["task_target"]).all())
            for role in ROLES),
        "same_seed_schedule_capacity": (
            training["geometry_head"]["parameters"] == training["task_head"]["parameters"]
            and training["geometry_head"]["updates"] == training["task_head"]["updates"]),
        "bounded_residual_output": all(
            float(np.abs(values[arm]).max(initial=0.0)) <= scale + 1e-6
            for values in residuals.values() for arm in ("geometry_head", "task_head")),
        "calibration_has_no_updates": True,
        "formal_roles_complete": set(formal) == set(FORMAL_ROLES),
        "zero_detector_backbone_forward": True,
        "zero_gpu": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {
        "schema": "strict60_task_supervised_shared_correction_head_producer_v1",
        "status": "complete_producer_review_pending", "contract": evidence(args.contract),
        "feature_names": FEATURE_NAMES, "method": contract["method"],
        "training": training, "fit_scale_m": scale,
        "fit_scale_raw_q95_m": quantile,
        "fit_task_constant_residual_xy_m": constant.tolist(),
        "target_summary": target_summary, "calibration": calibration,
        "formal_panels": formal, "checks": checks,
        "artifacts": {name: evidence(args.output / name) for name in artifacts},
        "actual_cost": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "cpu_threads": 2, "detector_forwards": 0, "backbone_forwards": 0,
            "head_updates": 2 * int(contract["training"]["updates"]), "gpu_seconds": 0,
        },
        "boundary": "Exposed strict60 development roles at actual-packet 0/400ms endpoints; raw fixed readout isolates geometry. Cached module cost is not full-system latency.",
    }
    atomic(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "fit_scale_m": scale,
                      "selected_alpha": selected, "target_summary": target_summary,
                      "actual_cost": summary["actual_cost"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
