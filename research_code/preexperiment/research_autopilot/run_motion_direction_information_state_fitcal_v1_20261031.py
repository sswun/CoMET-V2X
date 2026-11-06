#!/usr/bin/env python3
"""Fit/cal producer for the causal 2x2 motion-direction information state."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / "preexperiment/research_autopilot"
BASE = ROOT / "preexperiment/research_baselines"
sys.path[:0] = [str(AUTO), str(BASE)]

from audit_common_offset_peer_v1_20261026 import pairs
import motion_information_reference_main_20261031 as information
import run_crosscheckpoint_current_temporal_fixed_readout_controls_v1_20261029 as fixed
import run_crosscheckpoint_temporal_q005_task_refit_v1_20261029 as refit


TASK = "motion_direction_information_state_v1_20261031_main"
OWNER = "research_autopilot_motion_direction_information_state_v1"
METHODS = (
    "original_cv_q005_kf",
    "heading_original_q005_kf",
    "heading_directional_information",
    "heading_isotropic_equal_information",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def evidence(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def heading_projection(stale: np.ndarray, cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = cv.copy()
    yaw = stale[:, 6].astype(np.float64)
    unit = np.column_stack((np.cos(yaw), np.sin(yaw)))
    displacement = cv[:, :2].astype(np.float64) - stale[:, :2].astype(np.float64)
    along = np.einsum("ij,ij->i", displacement, unit)
    output[:, :2] = (stale[:, :2].astype(np.float64) + unit * along[:, None]).astype(np.float32)
    return output, unit


def frame_pairs(material: dict[str, np.ndarray], condition: int):
    a, b = material["condition_offsets"][condition:condition + 2].astype(int)
    source = material["source"][a:b]
    protected = material["protected"][a:b]
    remote_local = np.flatnonzero(source == 1)
    ego_local = np.flatnonzero((source == 0) & (protected == 1))
    if not len(remote_local) or not len(ego_local):
        return (np.empty((0, 2), np.float64), np.empty(0, np.float64),
                np.empty(0, np.float64), np.empty((0, 7), np.float32))
    remote, unit = heading_projection(
        material["stale_boxes"][a:b][remote_local], material["cv_boxes"][a:b][remote_local])
    ego = material["cv_boxes"][a:b][ego_local]
    remote_index, ego_index = pairs(remote, ego, 2.5, True)
    if not len(remote_index):
        return (np.empty((0, 2), np.float64), np.empty(0, np.float64),
                np.empty(0, np.float64), remote)
    residual = ego[ego_index, :2].astype(np.float64) - remote[remote_index, :2].astype(np.float64)
    heading = np.arctan2(unit[remote_index, 1], unit[remote_index, 0])
    age_seconds = material["cv_features"][a:b][remote_local][remote_index, 0].astype(np.float64) / 1000.0
    if np.any(age_seconds < 0) or np.any(age_seconds > 10.0):
        raise RuntimeError(("physical age outside frozen actual-packet range", condition,
                            float(age_seconds.min()), float(age_seconds.max())))
    return residual, heading, age_seconds, remote


def robust_mad(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError("empty or nonfinite robust scale population")
    return max(float(floor), float(np.median(np.abs(values - np.median(values))) / 0.6744897501960817))


def fit_scales(material: dict[str, np.ndarray]) -> dict[str, Any]:
    perpendicular: list[float] = []
    velocity: list[float] = []
    matched_frames = 0
    eligible_velocity_pairs = 0
    pair_count = 0
    for condition in range(len(material["wait_ms"])):
        residual, heading, age, _ = frame_pairs(material, condition)
        pair_count += len(residual)
        if len(residual) < 2:
            continue
        matched_frames += 1
        centered = residual - np.median(residual, axis=0)
        unit = np.column_stack((np.cos(heading), np.sin(heading)))
        normal = np.column_stack((-unit[:, 1], unit[:, 0]))
        perpendicular.extend(np.einsum("ij,ij->i", centered, normal).tolist())
        along = np.einsum("ij,ij->i", centered, unit)
        keep = age >= 0.05
        velocity.extend((along[keep] / age[keep]).tolist())
        eligible_velocity_pairs += int(keep.sum())
    sigma_perp = robust_mad(np.asarray(perpendicular), 0.05)
    sigma_velocity = robust_mad(np.asarray(velocity), 0.10)
    return {
        "sigma_perp_m": sigma_perp,
        "sigma_velocity_mps": sigma_velocity,
        "fit_pair_count": pair_count,
        "fit_multimatch_frame_count": matched_frames,
        "fit_perpendicular_residual_count": len(perpendicular),
        "fit_velocity_residual_count": eligible_velocity_pairs,
        "age_floor_seconds_for_velocity_fit": 0.05,
        "frame_common_mode": "componentwise median removed only while fitting scales",
        "robust_estimator": "MAD / 0.6744897501960817 with frozen physical floors",
    }


def rotation_jacobian(previous_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    rotation = current_pose[:3, :3].T @ previous_pose[:3, :3]
    return rotation[:2, :2]


def information_deltas(material: dict[str, np.ndarray], conditions: list[dict[str, Any]],
                       scales: dict[str, Any], isotropic: bool):
    count = len(material["wait_ms"])
    delta = np.zeros((count, 2), np.float64)
    covariance = np.zeros((count, 2, 2), np.float64)
    observation_information = np.zeros((count, 2, 2), np.float64)
    pair_counts = np.zeros(count, np.int16)
    age_mean = np.zeros(count, np.float64)
    states: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    seen: dict[tuple[str, int], set[tuple[int, str]]] = {}
    for condition, condition_row in enumerate(conditions):
        sequence = str(material["sequence"][condition])
        wait = int(material["wait_ms"][condition])
        key = (sequence, wait)
        pose = np.asarray(condition_row["current_ego_pose_lidar_to_world"], np.float64)
        residual, heading, age, _ = frame_pairs(material, condition)
        identity = (int(material["receiver_timestamp_us"][condition]),
                    str(material.get("packet_sha256", np.full(count, "", dtype="U1"))[condition]))
        if identity in seen.setdefault(key, set()):
            raise RuntimeError(("duplicate temporal evidence", key, identity))
        seen[key].add(identity)
        if key not in states:
            mean = np.zeros(2, np.float64)
            prior_covariance = np.eye(2, dtype=np.float64) + 0.05 ** 2 * np.eye(2)
        else:
            state = states[key]
            jacobian = rotation_jacobian(state["pose"], pose)
            mean = jacobian @ state["mean"]
            prior_covariance = jacobian @ state["covariance"] @ jacobian.T + 0.05 ** 2 * np.eye(2)
        mean, posterior_covariance, frame_information = information.update(
            mean, prior_covariance, residual, heading, age,
            float(scales["sigma_perp_m"]), float(scales["sigma_velocity_mps"]),
            isotropic_control=isotropic)
        states[key] = {"mean": mean.copy(), "covariance": posterior_covariance.copy(), "pose": pose.copy()}
        delta[condition] = mean
        covariance[condition] = posterior_covariance
        observation_information[condition] = frame_information
        pair_counts[condition] = len(residual)
        age_mean[condition] = float(age.mean()) if len(age) else 0.0
    return delta, covariance, observation_information, pair_counts, age_mean


def heading_scalar_delta(material: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = len(material["wait_ms"])
    delta = np.zeros((count, 2), np.float64)
    pair_counts = np.zeros(count, np.int16)
    states: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    for condition in range(count):
        key = (str(material["sequence"][condition]), int(material["wait_ms"][condition]))
        residual, _, _, _ = frame_pairs(material, condition)
        mean, variance = states.get(key, (np.zeros(2, np.float64), 1.0))
        prior = float(variance) + 0.05 ** 2
        if len(residual):
            gain = prior / (prior + 0.6 ** 2 / len(residual))
            mean = mean + gain * (residual.mean(axis=0) - mean)
            variance = (1.0 - gain) * prior
        else:
            variance = prior
        states[key] = (mean.copy(), float(variance))
        delta[condition] = mean
        pair_counts[condition] = len(residual)
    return delta, pair_counts


def build_heading(material: dict[str, np.ndarray], delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes = material["cv_boxes"].copy()
    features = material["cv_features"].copy()
    for condition, move in enumerate(delta):
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        source = material["source"][a:b]
        protected = material["protected"][a:b]
        remote = a + np.flatnonzero(source == 1)
        anchors = a + np.flatnonzero((source == 0) & (protected == 1))
        projected, _ = heading_projection(material["stale_boxes"][remote], material["cv_boxes"][remote])
        projected[:, :2] += move.astype(np.float32)
        boxes[remote] = projected
        features[remote] = fixed.base.actual.optional.update_features(
            material["cv_features"][remote], projected, material["cv_boxes"][anchors],
            material["raw_scores"][anchors])
    return boxes, features


def conditions_for(material: dict[str, np.ndarray], all_conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [all_conditions[int(index)] for index in material["dataset_index"]]
    for ordinal, row in enumerate(rows):
        identity = (str(row["sequence"]) == str(material["sequence"][ordinal])
                    and int(row["wait_ms"]) == int(material["wait_ms"][ordinal])
                    and int(row["receiver_timestamp_us"]) == int(material["receiver_timestamp_us"][ordinal]))
        if not identity:
            raise RuntimeError(("condition identity drift", ordinal))
    return rows


def signs(values: list[float]) -> dict[str, Any]:
    counts = Counter("positive" if value > 0 else "negative" if value < 0 else "zero" for value in values)
    return {"cells": len(values), "positive": int(counts["positive"]),
            "negative": int(counts["negative"]), "zero": int(counts["zero"]),
            "mean_pp": float(np.mean(values)), "minimum_pp": float(np.min(values)),
            "maximum_pp": float(np.max(values))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cpu_started = time.process_time()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = load_json(args.contract)
    if contract.get("task_id") != TASK or contract.get("owner") != OWNER \
            or contract.get("status") != "frozen_released_for_fitcal_execution":
        raise RuntimeError("contract identity/status")
    if sha(Path(__file__).resolve()) != contract["producer"]["sha256"]:
        raise RuntimeError("producer drift")
    for item in contract["inputs"].values():
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise RuntimeError(("input drift", item["path"]))
    args.output.mkdir(parents=True)
    fit = load_npz(ROOT / contract["inputs"]["fit_material"]["path"])
    cal = load_npz(ROOT / contract["inputs"]["calibration_material"]["path"])
    cal_q005 = load_npz(ROOT / contract["inputs"]["calibration_q005_frame"]["path"])
    all_conditions = load_jsonl(ROOT / contract["inputs"]["train_conditions"]["path"])
    fit_conditions = conditions_for(fit, all_conditions)
    cal_conditions = conditions_for(cal, all_conditions)
    if len(fit_conditions) != len(fit["wait_ms"]) or len(cal_conditions) != len(cal["wait_ms"]):
        raise RuntimeError("condition coverage")

    scales = fit_scales(fit)
    directional, directional_covariance, directional_information, pair_counts, age_mean = \
        information_deltas(cal, cal_conditions, scales, False)
    isotropic, isotropic_covariance, isotropic_information, isotropic_pairs, isotropic_age = \
        information_deltas(cal, cal_conditions, scales, True)
    heading_scalar, heading_pairs = heading_scalar_delta(cal)
    if not np.array_equal(pair_counts, isotropic_pairs) or not np.array_equal(pair_counts, heading_pairs) \
            or not np.array_equal(age_mean, isotropic_age):
        raise RuntimeError("same-pair control identity drift")
    if not np.allclose(np.trace(directional_information, axis1=1, axis2=2),
                       np.trace(isotropic_information, axis1=1, axis2=2), rtol=0, atol=1e-10):
        raise RuntimeError("equal-information trace control drift")
    q005 = cal_q005["q005_delta"].astype(np.float64)
    if len(q005) != len(cal["wait_ms"]):
        raise RuntimeError("q005 calibration identity length")

    boxes: dict[str, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    boxes[METHODS[0]], features[METHODS[0]] = refit.task_geometry(cal, q005)
    boxes[METHODS[1]], features[METHODS[1]] = build_heading(cal, heading_scalar)
    boxes[METHODS[2]], features[METHODS[2]] = build_heading(cal, directional)
    boxes[METHODS[3]], features[METHODS[3]] = build_heading(cal, isotropic)
    head = (ROOT / contract["inputs"]["matched_q005_head_direct_u"]["path"]).parent
    scores = {name: refit.score_with_head(cal, features[name], head) for name in METHODS}
    panels, predictions = fixed.base.evaluate(cal, boxes, scores)
    table = {(int(row["wait_ms"]), row["method"]): row for row in panels}
    comparisons = []
    for wait in sorted(map(int, np.unique(cal["wait_ms"]))):
        baseline = table[(wait, METHODS[1])]
        iso = table[(wait, METHODS[3])]
        direction = table[(wait, METHODS[2])]
        original = table[(wait, METHODS[0])]
        comparisons.append({
            "wait_ms": wait,
            "directional_minus_heading_kf_ap50_pp": direction["ap50"] - baseline["ap50"],
            "directional_minus_heading_kf_ap70_pp": direction["ap70"] - baseline["ap70"],
            "directional_minus_isotropic_ap50_pp": direction["ap50"] - iso["ap50"],
            "directional_minus_isotropic_ap70_pp": direction["ap70"] - iso["ap70"],
            "directional_minus_original_cv_ap70_pp": direction["ap70"] - original["ap70"],
            "isotropic_minus_heading_kf_ap70_pp": iso["ap70"] - baseline["ap70"],
        })
    prediction_path = args.output / "calibration8_predictions.npz"
    state_path = args.output / "calibration8_state.npz"
    save_npz(prediction_path, **predictions)
    save_npz(
        state_path,
        q005_delta=q005,
        heading_scalar_delta=heading_scalar,
        directional_delta=directional,
        isotropic_delta=isotropic,
        directional_covariance=directional_covariance,
        isotropic_covariance=isotropic_covariance,
        directional_information=directional_information,
        isotropic_information=isotropic_information,
        pair_count=pair_counts,
        mean_age_seconds=age_mean,
        condition_offsets=cal["condition_offsets"],
        dataset_index=cal["dataset_index"],
        sequence=cal["sequence"],
        wait_ms=cal["wait_ms"],
        receiver_timestamp_us=cal["receiver_timestamp_us"],
    )
    dir_eigen = np.linalg.eigvalsh(directional_covariance)
    iso_eigen = np.linalg.eigvalsh(isotropic_covariance)
    observed = pair_counts > 0
    summary = {
        "schema": "motion_direction_information_state_fitcal_producer_v1",
        "status": "fitcal_producer_complete_review_pending",
        "created_at": contract["frozen_at"],
        "task_id": TASK,
        "owner": OWNER,
        "contract": evidence(args.contract),
        "fit_scales": scales,
        "calibration": {
            "panels": panels,
            "comparisons": comparisons,
            "aggregate": {
                "directional_minus_heading_kf_ap70": signs(
                    [row["directional_minus_heading_kf_ap70_pp"] for row in comparisons]),
                "directional_minus_isotropic_ap70": signs(
                    [row["directional_minus_isotropic_ap70_pp"] for row in comparisons]),
                "directional_minus_original_cv_ap70": signs(
                    [row["directional_minus_original_cv_ap70_pp"] for row in comparisons]),
            },
            "conditions": len(cal["wait_ms"]),
            "matched_conditions": int(observed.sum()),
            "matched_pairs": int(pair_counts.sum()),
        },
        "state_diagnostics": {
            "directional_posterior_eigenvalue_quantiles": np.quantile(dir_eigen, [0, .1, .5, .9, 1]).tolist(),
            "isotropic_posterior_eigenvalue_quantiles": np.quantile(iso_eigen, [0, .1, .5, .9, 1]).tolist(),
            "directional_condition_number_quantiles": np.quantile(
                dir_eigen[:, 1] / np.maximum(dir_eigen[:, 0], 1e-15), [0, .1, .5, .9, 1]).tolist(),
            "directional_information_trace_equals_isotropic_max_abs": float(np.max(np.abs(
                np.trace(directional_information, axis1=1, axis2=2)
                - np.trace(isotropic_information, axis1=1, axis2=2)), initial=0.0)),
        },
        "outputs": {"predictions": evidence(prediction_path), "state": evidence(state_path)},
        "checks": {
            "fit_and_cal_condition_identity_exact": True,
            "same_pair_lists_all_heading_arms": True,
            "isotropic_total_information_trace_exact": True,
            "physical_age_seconds_nonnegative_and_below_unit_error_guard_10s": True,
            "one_method_per_arm_no_wait_selection": True,
            "q005_input_geometry_and_matched_readout_fixed": True,
            "zero_detector_or_backbone_forward": True,
            "zero_training_update": True,
            "zero_gpu": True,
        },
        "actual_cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_process_seconds": time.process_time() - cpu_started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "new_output_bytes": prediction_path.stat().st_size + state_path.stat().st_size,
            "detector_or_backbone_forwards": 0,
            "training_updates": 0,
            "gpu_seconds": 0,
        },
        "boundary": (
            "Fit24 estimates only two robust observation scales after within-frame common-mode removal; "
            "calibration8 evaluates four frozen system arms with one method per arm. The directional and "
            "isotropic controls use identical pairs and per-frame total Fisher trace. This is not formal "
            "hold14/val21 evidence, a per-cell selector, detector execution, or a universal claim."
        ),
    }
    if not all(summary["checks"].values()):
        raise RuntimeError(summary["checks"])
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "fit_scales": scales,
                      "aggregate": summary["calibration"]["aggregate"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
