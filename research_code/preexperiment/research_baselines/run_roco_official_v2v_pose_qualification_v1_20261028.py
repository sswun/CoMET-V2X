#!/usr/bin/env python3
"""Qualify the unmodified official RoCo box/pose-graph entry on V2 inputs.

This is a cached-box interface/coordinate pilot.  It uses the official
hard-coded candidate radius and disables uncertainty because the frozen V2
inputs do not contain a lawful RoCo uncertainty field.  It does not claim a
full RoCo reproduction or inspect AP.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "preexperiment/research_baselines"
COALIGN = ROOT / "third_party/external/CoAlign_64d1987affe4fb881c439cc498c30ab01bec67e8"
G2O_OVERLAY = ROOT / ".conda/coalign-g2o-python-20261010"
sys.path[:0] = [str(G2O_OVERLAY), str(COALIGN), str(BASE)]

# Freeze the OpenCOOD provider before the FreeAlign helper imported by pose can
# prepend its own minimal tree.  RoCo then receives the reviewed CoAlign graph.
import opencood  # noqa: F401
from opencood.utils.box_utils import boxes_to_corners_3d

import run_v2v4real_pose_stress_cached_geometry_v1_20261028 as pose
from diagnose_v2v_legal_history_geometry_v1_20261023 import local_artifact, sensor_manifests
from v2xseq_real_motion_packets import transform_boxes7d


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_roco(path: Path):
    spec = importlib.util.spec_from_file_location("official_roco_box_align_20261028", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official RoCo source")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def planar(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = np.asarray(matrix, np.float64)[:2, :2]
    result[:2, 3] = np.asarray(matrix, np.float64)[:2, 3]
    return result


def matrix_to_pose6(matrix: np.ndarray) -> np.ndarray:
    matrix = planar(matrix)
    return np.asarray([matrix[0, 3], matrix[1, 3], 0.0, 0.0,
                       math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])), 0.0],
                      dtype=np.float64)


def pose3_to_matrix(value: np.ndarray) -> np.ndarray:
    yaw = math.radians(float(value[2]))
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = [[math.cos(yaw), -math.sin(yaw)],
                      [math.sin(yaw), math.cos(yaw)]]
    result[:2, 3] = np.asarray(value[:2], np.float64)
    return result


def pose_error(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    xy = float(np.linalg.norm(estimate[:2, 3] - truth[:2, 3]))
    yaw_e = math.atan2(float(estimate[1, 0]), float(estimate[0, 0]))
    yaw_t = math.atan2(float(truth[1, 0]), float(truth[0, 0]))
    return xy, math.degrees(abs(float(pose.angle(yaw_e - yaw_t))))


def selected_ordinals(length: int, count: int) -> list[int]:
    if length < count:
        raise RuntimeError(("pilot population too small", length, count))
    values = np.rint(np.linspace(0, length - 1, count)).astype(np.int64).tolist()
    if len(set(values)) != count:
        raise RuntimeError(("non-unique pilot ordinals", values))
    return [int(value) for value in values]


def official_call(module: Any, ego_boxes: np.ndarray, source_boxes: np.ndarray,
                  nominal: np.ndarray) -> np.ndarray:
    corners = [boxes_to_corners_3d(ego_boxes, "hwl"),
               boxes_to_corners_3d(source_boxes, "hwl")]
    poses = np.stack((np.zeros(6, dtype=np.float64), matrix_to_pose6(nominal)))
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        refined = module.box_alignment_relative_sample_np(
            [value.copy() for value in corners], poses.copy(), uncertainty_list=None,
            use_uncertainty=False)
    refined = np.asarray(refined, np.float64)
    if refined.shape != (2, 3) or not np.isfinite(refined).all():
        raise ValueError(("invalid official RoCo return", refined.shape))
    return refined


def synthetic_probe(module: Any) -> dict[str, Any]:
    source = np.asarray([
        [-8.0, -3.0, 0.0, 1.6, 1.9, 4.1, -0.3],
        [-3.0, 7.0, 0.0, 1.5, 1.8, 3.8, 0.1],
        [2.0, -6.0, 0.0, 1.7, 2.0, 4.4, 0.5],
        [6.0, 4.0, 0.0, 1.6, 1.9, 4.0, -0.7],
        [11.0, -1.0, 0.0, 1.8, 2.2, 5.2, 1.0],
        [15.0, 9.0, 0.0, 1.5, 1.8, 3.6, -1.2],
    ], dtype=np.float64)
    yaw = math.radians(1.0)
    truth = np.eye(4, dtype=np.float64)
    truth[:2, :2] = [[math.cos(yaw), -math.sin(yaw)],
                      [math.sin(yaw), math.cos(yaw)]]
    truth[:2, 3] = [0.5, -0.25]
    receiver = transform_boxes7d(source, truth, np.eye(4, dtype=np.float64)).astype(np.float64)
    nominal = np.eye(4, dtype=np.float64)
    source_before = source.copy(); receiver_before = receiver.copy(); nominal_before = nominal.copy()
    refined = official_call(module, receiver, source, nominal)
    estimate = pose3_to_matrix(refined[1]); xy, yaw_deg = pose_error(estimate, truth)
    unchanged = bool(np.array_equal(source, source_before)
                     and np.array_equal(receiver, receiver_before)
                     and np.array_equal(nominal, nominal_before))
    ego_fixed = float(np.max(np.abs(refined[0]), initial=0.0)) <= 1e-8
    return {"refined_pose": refined.tolist(), "caller_inputs_unchanged": unchanged,
            "ego_fixed": ego_fixed, "translation_error_m": xy, "yaw_error_deg": yaw_deg,
            "passed": unchanged and ego_fixed and xy <= 1e-4 and yaw_deg <= 1e-4}


def run_case(module: Any, dataset: pose.PerturbedDataset, index: int) -> dict[str, Any]:
    pool = dataset.pool(index); kind = pool["hypothesis_type"].astype(np.int8)
    ego_ids = np.flatnonzero(kind == 0); stale_ids = np.flatnonzero(kind == 1)
    source_ids = pool["source_detection_indices"][stale_ids].astype(np.int32)
    raw = dataset.raw(local_artifact(ROOT, dataset.history[index]["current_raw_path"]))
    ego_boxes = pool["boxes"][ego_ids].astype(np.float64, copy=True)
    source_boxes = raw["pred_boxes"][source_ids].astype(np.float64, copy=True)
    meta = dataset.meta[index]; history = dataset.history[index]
    manifest = dataset.manifests[int(history["source_task_index"])]
    true_source = np.asarray(manifest["source_pose_lidar_to_world"], np.float64)
    noisy_source = pose.perturb_pose(true_source, dataset.arm, manifest["timestamp"])
    receiver_pose = np.asarray(meta["current_ego_pose_lidar_to_world"], np.float64)
    truth = planar(np.linalg.solve(receiver_pose, true_source))
    nominal = planar(np.linalg.solve(receiver_pose, noisy_source))
    reconstructed = transform_boxes7d(source_boxes, noisy_source, receiver_pose)
    reconstruction_error = float(np.max(np.abs(reconstructed - pool["boxes"][stale_ids]), initial=0.0))
    ego_before = ego_boxes.copy(); source_before = source_boxes.copy(); nominal_before = nominal.copy()
    status = "estimated"; error = None; estimate = nominal; ego_fixed = True
    started = time.perf_counter()
    try:
        refined = official_call(module, ego_boxes, source_boxes, nominal)
        ego_fixed = float(np.max(np.abs(refined[0]), initial=0.0)) <= 1e-8
        estimate = pose3_to_matrix(refined[1])
    except Exception as exception:
        status = f"exception_nominal_fallback:{type(exception).__name__}"
        error = f"{type(exception).__name__}: {exception}"; estimate = nominal
    elapsed = time.perf_counter() - started
    unchanged = bool(np.array_equal(ego_boxes, ego_before)
                     and np.array_equal(source_boxes, source_before)
                     and np.array_equal(nominal, nominal_before))
    nominal_xy, nominal_yaw = pose_error(nominal, truth)
    estimate_xy, estimate_yaw = pose_error(estimate, truth)
    return {
        "arm": dataset.arm, "dataset_index": int(index), "sequence": str(meta["sequence"]),
        "wait_ms": int(meta["wait_ms"]), "effective_age_ms": float(meta["effective_age_ms"]),
        "ego_boxes": int(len(ego_boxes)), "source_boxes": int(len(source_boxes)),
        "status": status, "error": error, "caller_inputs_unchanged": unchanged,
        "ego_pose_fixed": ego_fixed, "nominal_translation_error_m": nominal_xy,
        "nominal_yaw_error_deg": nominal_yaw, "estimated_translation_error_m": estimate_xy,
        "estimated_yaw_error_deg": estimate_yaw,
        "stale_reconstruction_max_abs": reconstruction_error, "official_call_seconds": elapsed,
    }


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    module = load_roco(ROOT / contract["inputs"]["roco_source"]["path"])
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    probe = synthetic_probe(module); manifests = sensor_manifests(ROOT)
    pose_contract = load_json(ROOT / contract["inputs"]["pose_contract"]["path"])
    rows: list[dict[str, Any]] = []; selection: dict[str, Any] = {}
    for role in contract["pilot"]["roles"]:
        split = "train" if role == "mechanism_holdout14" else "val"
        clean = pose.PerturbedDataset(ROOT / pose_contract["datasets"][split],
                                     ROOT / pose_contract["history"][split], "clean", manifests)
        data, _ = pose.build_role(clean, role)
        ordinals = selected_ordinals(len(data["dataset_index"]),
                                    int(contract["pilot"]["conditions_per_role"]))
        indices = [int(data["dataset_index"][ordinal]) for ordinal in ordinals]
        selection[role] = {"population_conditions": int(len(data["dataset_index"])),
                           "ordinals": ordinals, "dataset_indices": indices}
        for arm in contract["pilot"]["arms"]:
            dataset = pose.PerturbedDataset(ROOT / pose_contract["datasets"][split],
                                            ROOT / pose_contract["history"][split], arm, manifests)
            for index in indices:
                row = run_case(module, dataset, index); row["role"] = role; rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    clean_rows = [row for row in rows if row["arm"] == "clean"]
    checks = {
        "synthetic_direction_probe": bool(probe["passed"]),
        "all_declared_cases_completed": len(rows) == (len(contract["pilot"]["roles"])
            * len(contract["pilot"]["arms"]) * int(contract["pilot"]["conditions_per_role"])),
        "caller_inputs_unchanged": all(row["caller_inputs_unchanged"] for row in rows),
        "reference_agent_fixed": all(row["ego_pose_fixed"] for row in rows),
        "stale_coordinate_reconstruction": max(
            (row["stale_reconstruction_max_abs"] for row in rows), default=0.0) <= 5e-5,
        "clean_nominal_pose_exact": max(
            (row["nominal_translation_error_m"] for row in clean_rows), default=0.0) <= 1e-5
            and max((row["nominal_yaw_error_deg"] for row in clean_rows), default=0.0) <= 1e-4,
        "cpu_only": not bool(torch.cuda.is_available()),
    }
    summary = {
        "schema": "roco_official_v2v_pose_qualification_summary_v1",
        "status": "complete_producer_review_pending" if all(checks.values()) else "failed_qualification",
        "created_at": now(), "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": evidence(ROOT / contract["contract_path"]),
        "official_scope": "Unmodified official RoCo box_alignment_relative_sample_np with hard-coded radius 2m and uncertainty disabled on lawful cached V2 boxes/poses.",
        "synthetic_probe": probe, "selection": selection, "checks": checks,
        "status_counts": dict(sorted(counts.items())), "rows": rows,
        "actual_cost": {"wall_seconds": time.monotonic() - started,
                        "cpu_process_seconds": time.process_time(),
                        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                        "official_calls": len(rows) + 1, "detector_or_backbone_forwards": 0,
                        "optimizer_or_head_updates": 0, "gpu_seconds": 0},
        "boundary": "Interface/coordinate qualification only; no AP, training, learned uncertainty, full RoCo/CBM system, feature perturbation, or complete-system timing.",
    }
    atomic_json(output / "summary.json", summary); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA must remain disabled")
    contract = load_json(args.contract)
    if contract.get("status") != "frozen_released_for_execution":
        raise RuntimeError("contract is not released")
    if sha256(Path(__file__)) != contract["runner"]["sha256"]:
        raise RuntimeError("runner identity drift")
    for name, item in contract["inputs"].items():
        if isinstance(item, dict) and "sha256" in item:
            path = ROOT / item["path"]
            if sha256(path) != item["sha256"]:
                raise RuntimeError(("input identity drift", name, item["path"]))
    summary = compute(contract, args.output)
    print(json.dumps({"status": summary["status"], "checks": summary["checks"],
                      "actual_cost": summary["actual_cost"]}, sort_keys=True), flush=True)
    if summary["status"] != "complete_producer_review_pending":
        raise RuntimeError("RoCo qualification failed")


if __name__ == "__main__":
    main()
