#!/usr/bin/env python3
"""Current 24D causal-KF task geometry on the reviewed V2V4Real cache."""
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

import run_v2xvit_seed2024_task_geometry_fresh39d_v1_20261028 as recipe
import run_where_task_geometry_fresh39d_v1_20261028 as where_recipe
import run_strict60_task_supervised_shared_correction_head_v1_20261028 as task
from object_history_selector_core_v1_20261021 import PackedDataset

ROLES = recipe.ROLES
FORMAL_ROLES = recipe.FORMAL_ROLES
WAITS = (0, 400)


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def source_metadata(dataset: PackedDataset) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    packet_counts: dict[str, int] = {}
    for row in dataset.meta:
        path = row.get("source_raw_path")
        source = row.get("source")
        if not path:
            if source is not None:
                raise RuntimeError("source metadata without path")
            continue
        path = str((Path(path) if Path(path).is_absolute() else ROOT / path).resolve())
        # ``diagnostic.stale`` is the condition-specific retained population,
        # not the sender packet identity: one sender file can legitimately
        # yield different retained counts in different receiver conditions.
        # The frozen sender NPZ is the authoritative wire-record source.
        if path not in packet_counts:
            with np.load(path, allow_pickle=False) as archive:
                packet_counts[path] = int(len(archive["pred_boxes"]))
        value = (int(source["source_timestamp_us"]), packet_counts[path])
        if path in result and result[path] != value:
            raise RuntimeError((path, result[path], value))
        result[path] = value
    previous = {str((Path(row.get("previous_source_raw_path") or row.get("previous_raw_path"))
                     if Path(row.get("previous_source_raw_path") or row.get("previous_raw_path")).is_absolute()
                     else ROOT / (row.get("previous_source_raw_path") or row.get("previous_raw_path"))).resolve())
                for row in dataset.meta
                if row.get("previous_source_raw_path") or row.get("previous_raw_path")}
    missing = previous - set(result)
    if missing:
        raise RuntimeError(("unresolved previous sender packets", sorted(missing)[:5]))
    return result


def build_roles(train: PackedDataset, val: PackedDataset):
    # V2's reviewed cache calls this field ``previous_raw_path``.  Normalize
    # the already frozen path in memory to the recipe's explicit legal-history
    # name; persistent metadata is not modified.
    for dataset in (train, val):
        for index, row in enumerate(dataset.meta):
            row = dict(row)
            if row.get("previous_raw_path"):
                path = Path(row["previous_raw_path"])
                row["previous_source_raw_path"] = str(
                    (path if path.is_absolute() else ROOT / path).resolve())
            if row.get("source_raw_path"):
                path = Path(row["source_raw_path"])
                row["source_raw_path"] = str(
                    (path if path.is_absolute() else ROOT / path).resolve())
            if row.get("source") is not None:
                source = dict(row["source"])
                # Full-population qualification verifies effective_age_ms is
                # exactly (receiver_timestamp_us-source_timestamp_us)/1000.
                source["actual_age_ms"] = float(row["effective_age_ms"])
                row["source"] = source
            dataset.meta[index] = row
    old = recipe.source_metadata
    recipe.source_metadata = source_metadata
    try:
        role_data, role_frame = {}, {}
        for role in ROLES:
            dataset = train if role != "val21" else val
            role_data[role], role_frame[role] = recipe.build_role(dataset, role, role == "fit24")
        return role_data, role_frame
    finally:
        recipe.source_metadata = old


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    train = PackedDataset(ROOT / contract["datasets"]["train"])
    val = PackedDataset(ROOT / contract["datasets"]["val"])
    role_data, role_frame = build_roles(train, val)
    material, identity_checks = {}, {}
    for role in ROLES:
        frozen = where_recipe.combined_material(contract, role)
        data = role_data[role]
        for key in ("raw_scores", "source", "protected", "anchor", "condition_offsets",
                    "dataset_index", "wait_ms", "receiver_timestamp_us"):
            identity_checks[f"{role}_{key}_exact"] = recipe.exact(data[key], frozen[key])
        identity_checks[f"{role}_sequence_equal"] = bool(np.array_equal(
            data["sequence"].astype(str), frozen["sequence"].astype(str)))
        identity_checks[f"{role}_cv_boxes_max_error"] = float(np.max(
            np.abs(data["cv_boxes"] - frozen["cv_boxes"]), initial=0.0))
        identity_checks[f"{role}_kf_boxes_max_error"] = float(np.max(
            np.abs(data["kf_boxes"] - frozen["kf_boxes"]), initial=0.0))
        material[role] = frozen
        np.savez_compressed(output / f"{role}_frame_inputs.npz", **role_frame[role])
    booleans = [v for k, v in identity_checks.items() if k.endswith(("_exact", "_equal"))]
    errors = [v for k, v in identity_checks.items() if k.endswith("_max_error")]
    if not all(booleans) or max(errors, default=0.0) > 5e-6:
        raise RuntimeError({"input_identity": identity_checks})

    fit = role_frame["fit24"]
    mean = fit["feature"].mean(axis=0); std = np.maximum(fit["feature"].std(axis=0), 1e-6)
    fit_residual = fit["task_target"] - fit["kf_delta"]
    quantile = float(np.quantile(np.abs(fit_residual.ravel()), .95))
    scale = float(min(2.0, max(.25, quantile)))
    rng = np.random.default_rng(int(contract["training"]["seed"]))
    batches = rng.integers(0, len(fit["feature"]), size=(
        int(contract["training"]["updates"]), int(contract["training"]["batch_size"])), dtype=np.int32)
    np.savez_compressed(output / "fit_protocol.npz", feature_names=np.asarray(task.FEATURE_NAMES),
                        mean=mean, std=std, batches=batches, fit_scale=np.asarray(scale),
                        task_residual=fit_residual)
    training = task.train_head("task_head", (fit["feature"] - mean) / std, fit_residual,
                               batches, scale, int(contract["training"]["seed"]), output)
    residuals = {}
    for role in ROLES:
        residuals[role] = task.infer_head(
            "task_head", (role_frame[role]["feature"] - mean) / std, scale, output)
        np.savez_compressed(output / f"{role}_head_residuals.npz", task_head=residuals[role])

    calibration_rows = []
    for alpha in contract["calibration"]["alpha_grid"]:
        panels = recipe.raw_panels(role_data["calibration8"], {
            "task_head": role_frame["calibration8"]["kf_delta"] + float(alpha) * residuals["calibration8"]
        }, WAITS)
        calibration_rows.append({"alpha": float(alpha),
                                 "mean_ap70": float(np.mean([x["ap70"] for x in panels])),
                                 "panels": panels})
    selected = max(calibration_rows, key=lambda x: (x["mean_ap70"], -x["alpha"]))
    calibration = {"role": "calibration8", "criterion": "mean raw-readout AP70 over 0/400ms",
                   "alpha_grid": contract["calibration"]["alpha_grid"],
                   "selected_alpha": selected["alpha"], "tables": calibration_rows}
    recipe.atomic(output / "calibration.json", calibration)

    reference_summary = recipe.load_json(ROOT / contract["selected_stale_reference"])
    stale = {(row["role"], int(row["wait_ms"])): row
             for row in reference_summary["selected_deployment_panels"]
             if row["domain"] == "v2v4real" and row["geometry"] == "stale"}
    if set(stale) != {(role, wait) for role in FORMAL_ROLES for wait in WAITS}:
        raise RuntimeError(("selected stale reference incomplete", sorted(stale)))
    readout = ROOT / contract["readout"]["root"]
    maps = recipe.load_json(readout / "source_maps.json")
    panels_by_role, comparisons, formal_outputs = {}, [], {}
    for role in FORMAL_ROLES:
        dataset = train if role == "mechanism_holdout14" else val
        panels, predictions, diagnostic = recipe.evaluate_formal(
            dataset, role_data[role], role_frame[role], material[role], residuals[role],
            float(selected["alpha"]), readout, maps, output, role)
        panels_by_role[role] = panels
        prediction_path = output / f"{role}_predictions.npz"
        np.savez_compressed(prediction_path, **predictions)
        formal_outputs[role] = {"predictions": recipe.evidence(prediction_path),
                                "task_material": recipe.evidence(output / f"{role}_task_material.npz"),
                                "diagnostic": diagnostic}
        table = {(int(x["wait_ms"]), str(x["method"])): x for x in panels}
        for wait in WAITS:
            baseline = table[(wait, "kf_direct_u")]
            fixed = table[(wait, "task_geometry_fixed_kf_readout")]
            fresh = table[(wait, "task_geometry_fresh39d_direct_u")]
            strong = stale[(role, wait)]
            comparisons.append({
                "role": role, "wait_ms": wait,
                "system_delta_ap50_pp": fresh["ap50"] - baseline["ap50"],
                "system_delta_ap70_pp": fresh["ap70"] - baseline["ap70"],
                "geometry_fixed_readout_delta_ap70_pp": fixed["ap70"] - baseline["ap70"],
                "conditional_readout_delta_ap70_pp": fresh["ap70"] - fixed["ap70"],
                "system_delta_tp70": fresh["tp70"] - baseline["tp70"],
                "selected_stale_ap70": strong["ap70"],
                "system_minus_selected_stale_ap70_pp": fresh["ap70"] - strong["ap70"],
                "system_minus_selected_stale_tp70": fresh["tp70"] - strong["tp70"],
            })
    summary = {
        "schema": "v2v4real_task_geometry_fresh39d_producer_v1",
        "status": "complete_producer_review_pending", "created_at": now(),
        "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": recipe.evidence(ROOT / contract["contract_path"]),
        "detector": contract["detector"], "feature_names": list(task.FEATURE_NAMES),
        "training": training, "fit_scale_m": scale, "fit_scale_raw_q95_m": quantile,
        "calibration": calibration, "input_identity": identity_checks,
        "selected_stale_reference": recipe.evidence(ROOT / contract["selected_stale_reference"]),
        "panels": panels_by_role, "comparisons": comparisons,
        "formal_outputs": formal_outputs,
        "artifacts": {"fit_protocol": recipe.evidence(output / "fit_protocol.npz"),
                      "task_head_weights": recipe.evidence(output / "task_head.pth")},
        "actual_cost": {"wall_seconds": time.monotonic() - started,
                        "cpu_process_seconds": time.process_time(),
                        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                        "task_head_updates": int(contract["training"]["updates"]),
                        "detector_or_backbone_forwards": 0, "gpu_seconds": 0},
        "boundary": "V2V4Real cached formal cross-domain diagnostic on existing roles. Current task geometry starts from causal KF and is also compared to the calibration-selected stronger stale system. This is not the main vehicle-road final claim, hidden test, actual-network cost, or a repaired external algorithm.",
    }
    recipe.atomic(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "": raise RuntimeError("CUDA disabled required")
    contract = recipe.load_json(args.contract)
    if contract.get("status") != "frozen_released_for_execution": raise RuntimeError("contract not released")
    if recipe.sha256(Path(__file__)) != contract["runner"]["sha256"]: raise RuntimeError("runner drift")
    for item in contract["inputs"].values():
        if recipe.sha256(ROOT / item["path"]) != item["sha256"]: raise RuntimeError(("input drift", item["path"]))
    summary = compute(contract, args.output)
    print(json.dumps({"status": summary["status"], "calibration": summary["calibration"]["selected_alpha"],
                      "comparisons": summary["comparisons"], "actual_cost": summary["actual_cost"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__": main()

