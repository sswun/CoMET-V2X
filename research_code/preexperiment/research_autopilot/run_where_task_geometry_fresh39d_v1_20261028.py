#!/usr/bin/env python3
"""Current 24D task geometry on the reviewed Where KF+matched-39D cache."""
from __future__ import annotations

import argparse
import datetime as dt
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
BASE = ROOT / "preexperiment/research_baselines"
sys.path[:0] = [str(AUTO), str(BASE)]

import run_v2xvit_seed2024_task_geometry_fresh39d_v1_20261028 as core
import run_strict60_task_supervised_shared_correction_head_v1_20261028 as task
import run_shared_offset_optional_formal_v1_20261026 as formal
from where_receiver_timestamp_adapter_v1_20261026 import TimestampedPackedDataset

ROLES = core.ROLES
FORMAL_ROLES = core.FORMAL_ROLES
WAITS = core.WAITS


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def combined_material(contract: dict[str, Any], role: str) -> dict[str, np.ndarray]:
    cv = core.load_npz(ROOT / contract["materials"][role]["cv"])
    kf = core.load_npz(ROOT / contract["materials"][role]["kf"])
    common = ("raw_scores", "source", "protected", "anchor", "condition_offsets",
              "dataset_index", "wait_ms", "receiver_timestamp_us")
    for key in common:
        if not core.exact(cv[key], kf[key]):
            raise RuntimeError((role, "cv/kf population drift", key))
    if not np.array_equal(cv["sequence"].astype(str), kf["sequence"].astype(str)):
        raise RuntimeError((role, "cv/kf sequence drift"))
    return {
        **{key: kf[key] for key in common},
        "sequence": kf["sequence"].astype("U16"),
        "cv_boxes": cv["boxes"], "kf_boxes": kf["boxes"],
        "cv_features": cv["features"], "kf_features": kf["features"],
    }


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    timestamp_archive = ROOT / contract["datasets"]["receiver_timestamps"]
    train = TimestampedPackedDataset(ROOT / contract["datasets"]["train"], timestamp_archive, "train")
    val = TimestampedPackedDataset(ROOT / contract["datasets"]["val"], timestamp_archive, "val")
    role_data: dict[str, dict[str, np.ndarray]] = {}
    role_frame: dict[str, dict[str, np.ndarray]] = {}
    material: dict[str, dict[str, np.ndarray]] = {}
    identity: dict[str, Any] = {}
    for role in ROLES:
        dataset = train if role != "val21" else val
        data, frame = core.build_role(dataset, role, role == "fit24")
        frozen = combined_material(contract, role)
        exact_keys = ("raw_scores", "source", "protected", "anchor", "condition_offsets",
                      "dataset_index", "wait_ms", "receiver_timestamp_us")
        for key in exact_keys:
            identity[f"{role}_{key}_exact"] = core.exact(data[key], frozen[key])
        identity[f"{role}_sequence_equal"] = bool(np.array_equal(
            data["sequence"].astype(str), frozen["sequence"].astype(str)))
        identity[f"{role}_cv_boxes_max_error"] = float(np.max(
            np.abs(data["cv_boxes"] - frozen["cv_boxes"]), initial=0.0))
        identity[f"{role}_kf_boxes_max_error"] = float(np.max(
            np.abs(data["kf_boxes"] - frozen["kf_boxes"]), initial=0.0))
        role_data[role], role_frame[role], material[role] = data, frame, frozen
        np.savez_compressed(output / f"{role}_frame_inputs.npz", **frame)
    boolean_identity = [value for key, value in identity.items() if key.endswith(("_exact", "_equal"))]
    numeric_identity = [value for key, value in identity.items() if key.endswith("_max_error")]
    if not all(boolean_identity) or max(numeric_identity, default=0.0) > 5e-6:
        raise RuntimeError({"input_identity": identity})

    fit = role_frame["fit24"]
    mean = fit["feature"].mean(axis=0)
    std = np.maximum(fit["feature"].std(axis=0), 1e-6)
    residual = fit["task_target"] - fit["kf_delta"]
    quantile = float(np.quantile(np.abs(residual.ravel()), .95))
    scale = float(min(2.0, max(.25, quantile)))
    rng = np.random.default_rng(int(contract["training"]["seed"]))
    batches = rng.integers(0, len(fit["feature"]), size=(
        int(contract["training"]["updates"]), int(contract["training"]["batch_size"])),
        dtype=np.int32)
    np.savez_compressed(output / "fit_protocol.npz", feature_names=np.asarray(task.FEATURE_NAMES),
                        mean=mean, std=std, batches=batches, fit_scale=np.asarray(scale),
                        task_residual=residual)
    training = task.train_head("task_head", (fit["feature"] - mean) / std,
                               residual, batches, scale, int(contract["training"]["seed"]), output)
    residuals: dict[str, np.ndarray] = {}
    for role in ROLES:
        residuals[role] = task.infer_head(
            "task_head", (role_frame[role]["feature"] - mean) / std, scale, output)
        np.savez_compressed(output / f"{role}_head_residuals.npz", task_head=residuals[role])

    calibration_rows = []
    for alpha in contract["calibration"]["alpha_grid"]:
        panels = core.raw_panels(role_data["calibration8"], {
            "task_head": role_frame["calibration8"]["kf_delta"] + float(alpha) * residuals["calibration8"]
        })
        calibration_rows.append({"alpha": float(alpha),
                                 "mean_ap70": float(np.mean([x["ap70"] for x in panels])),
                                 "panels": panels})
    selected = max(calibration_rows, key=lambda row: (row["mean_ap70"], -row["alpha"]))
    calibration = {"role": "calibration8", "criterion": "mean raw-readout AP70 over 0/400ms",
                   "alpha_grid": contract["calibration"]["alpha_grid"],
                   "selected_alpha": selected["alpha"], "tables": calibration_rows}
    core.atomic(output / "calibration.json", calibration)

    readout = ROOT / contract["readout"]["root"]
    maps = core.load_json(readout / "source_maps.json")
    panels_by_role: dict[str, Any] = {}
    comparisons = []
    formal_outputs: dict[str, Any] = {}
    for role in FORMAL_ROLES:
        dataset = train if role == "mechanism_holdout14" else val
        panels, predictions, diagnostic = core.evaluate_formal(
            dataset, role_data[role], role_frame[role], material[role], residuals[role],
            float(selected["alpha"]), readout, maps, output, role)
        panels_by_role[role] = panels
        prediction_path = output / f"{role}_predictions.npz"
        np.savez_compressed(prediction_path, **predictions)
        formal_outputs[role] = {
            "predictions": core.evidence(prediction_path),
            "task_material": core.evidence(output / f"{role}_task_material.npz"),
            "diagnostic": diagnostic,
        }
        table = {(int(row["wait_ms"]), str(row["method"])): row for row in panels}
        for wait in WAITS:
            baseline = table[(wait, "kf_direct_u")]
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
    summary = {
        "schema": "where_task_geometry_fresh39d_producer_v1",
        "status": "complete_producer_review_pending", "created_at": now(),
        "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": core.evidence(ROOT / contract["contract_path"]),
        "detector": contract["detector"], "feature_names": list(task.FEATURE_NAMES),
        "training": training, "fit_scale_m": scale, "fit_scale_raw_q95_m": quantile,
        "calibration": calibration, "input_identity": identity,
        "panels": panels_by_role, "comparisons": comparisons,
        "formal_outputs": formal_outputs,
        "artifacts": {"fit_protocol": core.evidence(output / "fit_protocol.npz"),
                      "task_head_weights": core.evidence(output / "task_head.pth")},
        "actual_cost": {"wall_seconds": time.monotonic() - started,
                        "cpu_process_seconds": time.process_time(),
                        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                        "task_head_updates": int(contract["training"]["updates"]),
                        "detector_or_backbone_forwards": 0, "gpu_seconds": 0},
        "boundary": "Where2comm frozen cached formal pipeline on exposed SPD roles; zero detector forward. This is current 24D task-geometry cross-backbone evidence, not actual-network cost, hidden test, V2V4Real transfer, or final core claim.",
    }
    core.atomic(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA must be disabled")
    contract = core.load_json(args.contract)
    if contract.get("status") != "frozen_released_for_execution":
        raise RuntimeError("contract not released")
    if core.sha256(Path(__file__)) != contract["runner"]["sha256"]:
        raise RuntimeError("runner identity drift")
    for item in contract["inputs"].values():
        if core.sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(("input drift", item["path"]))
    summary = compute(contract, args.output)
    print(json.dumps({"status": summary["status"], "calibration": summary["calibration"]["selected_alpha"],
                      "comparisons": summary["comparisons"], "actual_cost": summary["actual_cost"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

