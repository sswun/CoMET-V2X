#!/usr/bin/env python3
"""Full V2 cached pose-stress control for official CBM matching plus SVD.

The detector outputs, source identities, direct-39D readout and all baseline
predictions are frozen.  Only CBM geometry is newly computed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from types import SimpleNamespace
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "preexperiment/research_baselines"
sys.path.insert(0, str(BASE))

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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


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


def load_cbm(path: Path):
    spec = importlib.util.spec_from_file_location("official_cbm_full_20261028", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official CBM")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args = SimpleNamespace(sigma1=10.0 * math.pi / 180.0, sigma2=3.0, sigma3=1.0,
                           absolute_dis_lim=20.0)
    matcher = module.CBM(args)
    if str(matcher.device) != "cpu":
        raise RuntimeError(("CBM must remain CPU-only", str(matcher.device)))
    return matcher


def planar_svd(source_xy: np.ndarray, receiver_xy: np.ndarray) -> np.ndarray:
    source_xy = np.asarray(source_xy, np.float64); receiver_xy = np.asarray(receiver_xy, np.float64)
    if source_xy.shape != receiver_xy.shape or len(source_xy) < 2:
        raise ValueError("invalid matched centers")
    source0 = source_xy - source_xy.mean(axis=0)
    receiver0 = receiver_xy - receiver_xy.mean(axis=0)
    if float(np.linalg.norm(source0)) <= 1e-6:
        raise ValueError("degenerate source centers")
    u, _, vt = np.linalg.svd(source0.T @ receiver0)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0; rotation = vt.T @ u.T
    translation = receiver_xy.mean(axis=0) - rotation @ source_xy.mean(axis=0)
    result = np.eye(4, dtype=np.float64); result[:2, :2] = rotation; result[:2, 3] = translation
    return result


def pose_error(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    xy = float(np.linalg.norm(estimate[:2, 3] - truth[:2, 3]))
    yaw_e = math.atan2(float(estimate[1, 0]), float(estimate[0, 0]))
    yaw_t = math.atan2(float(truth[1, 0]), float(truth[0, 0]))
    return xy, math.degrees(abs(float(pose.angle(yaw_e - yaw_t))))


def stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(len(array)), "mean": float(array.mean()),
            "median": float(np.median(array)), "p95": float(np.percentile(array, 95)),
            "max": float(array.max())}


def build_cbm_geometry(dataset: pose.PerturbedDataset, data: dict[str, np.ndarray], matcher: Any
                       ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    boxes: list[np.ndarray] = []; features: list[np.ndarray] = []
    status = Counter(); match_counts: list[float] = []; call_seconds: list[float] = []
    pose_xy: list[float] = []; pose_yaw: list[float] = []; reconstruction: list[float] = []
    for index_value in data["dataset_index"]:
        index = int(index_value); pool = dataset.pool(index); meta = dataset.meta[index]
        ego, stale_ids, _ = pose.formal.identities(pool)
        ego_boxes = pool["boxes"][ego].astype(np.float64, copy=True)
        source_ids = pool["source_detection_indices"][stale_ids].astype(np.int32)
        raw = dataset.raw(local_artifact(ROOT, dataset.history[index]["current_raw_path"]))
        source_boxes = raw["pred_boxes"][source_ids].astype(np.float64, copy=True)
        history = dataset.history[index]
        manifest = dataset.manifests[int(history["source_task_index"])]
        true_source = np.asarray(manifest["source_pose_lidar_to_world"], np.float64)
        noisy_source = pose.perturb_pose(true_source, dataset.arm, manifest["timestamp"])
        receiver_pose = np.asarray(meta["current_ego_pose_lidar_to_world"], np.float64)
        truth = np.linalg.solve(receiver_pose, true_source)
        nominal = np.linalg.solve(receiver_pose, noisy_source)
        stale = pool["boxes"][stale_ids].astype(np.float64, copy=True)
        reconstructed = transform_boxes7d(source_boxes, noisy_source, receiver_pose)
        reconstruction.append(float(np.max(np.abs(reconstructed - stale), initial=0.0)))
        estimate = nominal; case_status = "estimated"; matches = np.empty((0, 2), np.int64)
        started = time.perf_counter()
        try:
            value = matcher(ego_boxes.copy(), source_boxes.copy(), nominal.copy())
            matches = value.detach().cpu().numpy().astype(np.int64)
            if len(matches) < 2:
                case_status = "insufficient_matches_nominal_fallback"
            else:
                estimate = planar_svd(source_boxes[matches[:, 1], :2], ego_boxes[matches[:, 0], :2])
        except Exception as exception:  # retain official/adapter failure without tuning
            case_status = f"exception_nominal_fallback:{type(exception).__name__}"
            estimate = nominal
        call_seconds.append(time.perf_counter() - started); status[case_status] += 1
        match_counts.append(float(len(matches)))
        xy, yaw = pose_error(estimate, truth); pose_xy.append(xy); pose_yaw.append(yaw)
        remote = transform_boxes7d(source_boxes, estimate, np.eye(4, dtype=np.float64)).astype(np.float32)
        protected_local = np.flatnonzero(pool["is_protected"][ego])
        protected_boxes = pool["boxes"][ego][protected_local]
        protected_scores = pool["raw_scores"][ego][protected_local]
        first30 = pose.feature_matrix(
            remote, source_boxes, pool["raw_scores"][stale_ids], float(meta["effective_age_ms"]),
            estimate, protected_boxes, protected_scores)
        remote_feature = np.column_stack((first30, pool["features"][stale_ids, 30:])).astype(np.float32)
        boxes.append(np.concatenate((pool["boxes"][ego], remote)).astype(np.float32))
        features.append(np.concatenate((pool["features"][ego], remote_feature)).astype(np.float32))
    diagnostics = {
        "status_counts": dict(sorted(status.items())),
        "success_rate": status["estimated"] / max(len(data["dataset_index"]), 1),
        "match_count": stats(match_counts),
        "pose_translation_error_m": stats(pose_xy),
        "pose_yaw_error_deg": stats(pose_yaw),
        "stale_reconstruction_max_abs": max(reconstruction, default=0.0),
        "official_call_seconds": float(sum(call_seconds)),
    }
    return np.concatenate(boxes), np.concatenate(features), diagnostics


def evaluate(dataset: pose.PerturbedDataset, role: str, arm: str, data: dict[str, np.ndarray],
             boxes: np.ndarray, features: np.ndarray, readout: Path, maps: dict[str, Any],
             output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    direct = pose.formal.direct_scores({"cbm": features}, readout)["cbm"]
    utility = pose.formal.source_map(direct, data["source"], maps["maps"]["direct_to_utility"])
    methods = ("cbm_raw", "cbm_frozen39d")
    aggregate = defaultdict(lambda: {"score": [], "tp50": [], "tp70": [], "gt": 0, "outputs": 0})
    saved = {name: {"candidate": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
             for name in methods}
    for condition, index_value in enumerate(data["dataset_index"]):
        index = int(index_value); gt = dataset.pool(index)["gt_boxes"]
        a, b = data["condition_offsets"][condition:condition + 2].astype(int)
        raw = data["raw_scores"][a:b]; source = data["source"][a:b]
        protected = data["protected"][a:b]; wait = int(data["wait_ms"][condition])
        for method, scoring in (("cbm_raw", raw), ("cbm_frozen39d", utility[a:b])):
            ids, score = pose.formal.ordered(raw, source, protected, scoring, None)
            matrix = pose.iou32(boxes[a:b][ids], gt)
            match50 = pose.greedy(matrix, np.arange(len(ids)), .5)
            match70 = pose.greedy(matrix, np.arange(len(ids)), .7)
            row = aggregate[(wait, method)]; row["score"].append(score)
            row["tp50"].append((match50 >= 0).astype(np.int8)); row["tp70"].append((match70 >= 0).astype(np.int8))
            row["gt"] += len(gt); row["outputs"] += len(ids)
            record = saved[method]; record["candidate"].append((a + ids).astype(np.int64))
            record["score"].append(score.astype(np.float64)); record["tp50"].append((match50 >= 0).astype(np.int8))
            record["tp70"].append((match70 >= 0).astype(np.int8)); record["offset"].append(record["offset"][-1] + len(ids))
    panels = []
    for (wait, method), row in sorted(aggregate.items()):
        panel = {"arm": arm, "role": role, "wait_ms": wait, "method": method,
                 "gt": int(row["gt"]), "outputs": int(row["outputs"])}
        scores = np.concatenate(row["score"])
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"])
            metric = pose.voc_ap(tp.tolist(), (1 - tp).tolist(), scores.tolist(), row["gt"], True)
            panel[f"ap{threshold}"] = 100.0 * metric["ap"]
            panel[f"tp{threshold}"] = int(metric["tp"]); panel[f"fp{threshold}"] = int(metric["fp"])
        panels.append(panel)
    archive: dict[str, np.ndarray] = {
        "method_names": np.asarray(methods, dtype="U32"),
        "condition_offsets": data["condition_offsets"], "dataset_index": data["dataset_index"],
        "sequence": data["sequence"], "wait_ms": data["wait_ms"],
        "receiver_timestamp_us": data["receiver_timestamp_us"], "source": data["source"],
        "cbm__boxes": boxes, "cbm__direct39d": direct,
    }
    for method, record in saved.items():
        for field in ("candidate", "score", "tp50", "tp70"):
            archive[f"{method}__{field}"] = np.concatenate(record[field])
        archive[f"{method}__offset"] = np.asarray(record["offset"], np.int64)
    path = output / f"{arm}__{role}__cbm_predictions.npz"
    np.savez_compressed(path, **archive)
    return panels, evidence(path)


def compare_baselines(cbm_panels: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    base = {(row["arm"], row["role"], int(row["wait_ms"]), row["method"]): row
            for row in baseline["panels"]}
    result = []
    for row in cbm_panels:
        scoring = "raw" if row["method"].endswith("_raw") else "frozen39d"
        item = dict(row)
        for geometry in ("stale", "freealign", "causal_kf", "current_task"):
            reference = base[(row["arm"], row["role"], int(row["wait_ms"]), f"{geometry}_{scoring}")]
            item[f"minus_{geometry}_ap50_pp"] = row["ap50"] - reference["ap50"]
            item[f"minus_{geometry}_ap70_pp"] = row["ap70"] - reference["ap70"]
        result.append(item)
    return result


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    matcher = load_cbm(ROOT / contract["inputs"]["cbm_source"]["path"])
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    pose_contract = load_json(ROOT / contract["inputs"]["pose_contract"]["path"])
    baseline = load_json(ROOT / contract["inputs"]["baseline_pose_summary"]["path"])
    baseline_manifest = load_json(ROOT / contract["inputs"]["baseline_prediction_manifest"]["path"])
    if baseline_manifest.get("summary_sha256") != contract["inputs"]["baseline_pose_summary"]["sha256"]:
        raise RuntimeError("baseline summary/manifest identity mismatch")
    for item in baseline_manifest["predictions"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(("baseline prediction identity drift", item["path"]))
    manifests = sensor_manifests(ROOT); maps = load_json(ROOT / pose_contract["inputs"]["source_maps"]["path"])
    readout = ROOT / pose_contract["inputs"]["direct_readout_root"]
    panels: list[dict[str, Any]] = []; arm_summaries: dict[str, Any] = {}; predictions: dict[str, Any] = {}
    baseline_identity_checks: dict[str, Any] = {}
    for arm in contract["arms"]:
        arm_summary = {"roles": {}}
        for role in contract["roles"]:
            split = "train" if role == "mechanism_holdout14" else "val"
            dataset = pose.PerturbedDataset(ROOT / pose_contract["datasets"][split],
                                            ROOT / pose_contract["history"][split], arm, manifests)
            data, _ = pose.build_role(dataset, role)
            baseline_file = ROOT / baseline["predictions"][f"{arm}|{role}"]["path"]
            baseline_npz = load_npz(baseline_file)
            identity_ok = (
                np.array_equal(data["dataset_index"], baseline_npz["dataset_index"])
                and np.array_equal(data["condition_offsets"], baseline_npz["condition_offsets"])
                and np.array_equal(data["source"], baseline_npz["source"])
                and np.array_equal(data["wait_ms"], baseline_npz["wait_ms"])
            )
            baseline_identity_checks[f"{arm}|{role}"] = bool(identity_ok)
            if not identity_ok:
                raise RuntimeError(("baseline population identity mismatch", arm, role))
            cbm_boxes, cbm_features, diagnostic = build_cbm_geometry(dataset, data, matcher)
            role_panels, prediction = evaluate(dataset, role, arm, data, cbm_boxes, cbm_features,
                                               readout, maps, output)
            panels.extend(role_panels); predictions[f"{arm}|{role}"] = prediction
            arm_summary["roles"][role] = {"conditions": int(len(data["dataset_index"])),
                                                  "candidates": int(len(data["source"])),
                                                  "diagnostic": diagnostic, "prediction": prediction}
        arm_summaries[arm] = arm_summary
    comparisons = compare_baselines(panels, baseline)
    summary = {
        "schema": "cbm_official_v2v_pose_full_summary_v1",
        "status": "complete_producer_review_pending",
        "created_at": now(), "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": evidence(ROOT / contract["contract_path"]),
        "official_scope": "Official CBM matching with unchanged defaults plus predeclared planar SVD; not full CBM.",
        "arms": contract["arms"], "roles": contract["roles"],
        "methods": ["cbm_raw", "cbm_frozen39d"],
        "panels": panels, "comparisons": comparisons, "arm_summaries": arm_summaries,
        "predictions": predictions,
        "baseline_population_identity": {"passed": all(baseline_identity_checks.values()),
                                         "checks": baseline_identity_checks},
        "input_and_weight_identity": contract["inputs"],
        "actual_cost": {
            "wall_seconds": time.monotonic() - started, "cpu_process_seconds": time.process_time(),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "official_cbm_calls": sum(role["conditions"] for arm in arm_summaries.values()
                                      for role in arm["roles"].values()),
            "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0, "gpu_seconds": 0,
        },
        "boundary": "Full exposed-role cached geometry/readout control; not full CBM/RoCo, hidden test, feature-system perturbation, or complete-system latency.",
    }
    atomic_json(output / "summary.json", summary)
    return summary


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
        if isinstance(item, dict) and "sha256" in item and sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(("input identity drift", name, item["path"]))
    result = compute(contract, args.output)
    print(json.dumps({"status": result["status"], "panels": len(result["panels"]),
                      "actual_cost": result["actual_cost"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
