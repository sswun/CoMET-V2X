#!/usr/bin/env python3
"""Refit current task targets/heads from the actual q005 temporal base."""
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

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / "preexperiment/research_autopilot"
sys.path.insert(0, str(AUTO))
import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
import run_crosscheckpoint_current_temporal_fixed_readout_controls_v1_20261029 as fixed
import run_v2xvit_seed2024_task_geometry_fresh39d_v1_20261028 as task_core
import run_strict60_task_supervised_shared_correction_head_v1_20261028 as task


TASK = "crosscheckpoint_temporal_q005_task_refit_v1_20261029"
DOMAINS = ("v2xvit_seed2025", "where2comm")
ROLES = matched.ROLES
FORMAL = matched.FORMAL


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ev(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha(path),
            "bytes": path.stat().st_size}


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def q005_frame(old: dict[str, np.ndarray], material: dict[str, np.ndarray],
               make_targets: bool) -> dict[str, np.ndarray]:
    feature = old["feature"].astype(np.float64).copy()
    anchors, measurements = fixed.anchor_inputs(old)
    state: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    delta = np.zeros((len(feature), 2), np.float64)
    target = np.full((len(feature), 2), np.nan, np.float64)
    stats_array = np.zeros((len(feature), 8), np.float64)
    for condition, (sequence_value, wait_value) in enumerate(
            zip(material["sequence"], material["wait_ms"])):
        key = (str(sequence_value), int(wait_value)); count = int(anchors[condition])
        mean, variance = state.get(key, (np.zeros(2, np.float64), 1.0))
        mean = mean.copy(); prior_mean = mean.copy(); prior = float(variance) + .05 ** 2
        feature[condition, :2] = prior_mean; feature[condition, 2] = np.log(prior)
        if count:
            gain = prior / (prior + .6 ** 2 / count)
            mean += gain * (measurements[condition] - mean)
            variance = (1.0 - gain) * prior
        else:
            variance = prior
        feature[condition, 3:5] = mean; delta[condition] = mean
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
    output = {key: value.copy() for key, value in old.items()}
    output.update(feature=feature, kf_delta=delta, task_target=target,
                  task_stats=stats_array)
    return output


def task_geometry(material: dict[str, np.ndarray], delta: np.ndarray):
    boxes = material["cv_boxes"].copy(); features = material["cv_features"].copy()
    for condition, move in enumerate(delta):
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        source = material["source"][a:b]; protected = material["protected"][a:b]
        remote = np.flatnonzero(source == 1); fixed_ids = np.flatnonzero(protected == 1)
        boxes[a:b][remote, :2] += move.astype(np.float32)
        features[a:b][remote] = fixed.base.actual.optional.update_features(
            material["cv_features"][a:b][remote], boxes[a:b][remote],
            material["cv_boxes"][a:b][fixed_ids], material["raw_scores"][a:b][fixed_ids])
    return boxes, features


def score_with_head(material: dict[str, np.ndarray], features: np.ndarray, head: Path):
    value = dict(material); value["features"] = features
    direct = matched.readout.score_direct_u(value, head)
    maps = load(head / "source_maps.json")
    return matched.readout.apply_source_map(
        direct, material["source"], maps["maps"]["direct_to_utility"])


def domain_compute(name: str, spec: dict[str, Any], contract: dict[str, Any],
                   output: Path) -> dict[str, Any]:
    materials, deltas, _ = matched.prepare_domain(spec)
    frames = {role: load_npz(ROOT / spec["roles"][role]["frame_input"]) for role in ROLES}
    qframes = {role: q005_frame(frames[role], materials[role], role == "fit24") for role in ROLES}
    q005_errors = {}
    for role in ROLES:
        q005_errors[role] = float(np.max(np.abs(
            qframes[role]["kf_delta"] - deltas[role]["q005"]), initial=0.0))
    if max(q005_errors.values()) > 1e-12:
        raise RuntimeError((name, "q005 state mismatch", q005_errors))

    domain = output / name; domain.mkdir(parents=True, exist_ok=False)
    fit = qframes["fit24"]; protocol = load_npz(ROOT / contract["domains"][name]["old_fit_protocol"])
    batches = protocol["batches"].astype(np.int32)
    if len(batches) != 2000 or batches.max() >= len(fit["feature"]):
        raise RuntimeError((name, "old schedule population drift"))
    mean = fit["feature"].mean(axis=0); std = np.maximum(fit["feature"].std(axis=0), 1e-6)
    residual_target = fit["task_target"] - fit["kf_delta"]
    quantile = float(np.quantile(np.abs(residual_target.ravel()), .95))
    scale = float(min(2.0, max(.25, quantile)))
    np.savez_compressed(domain / "fit_protocol.npz",
                        feature_names=np.asarray(task.FEATURE_NAMES), mean=mean, std=std,
                        batches=batches, fit_scale=np.asarray(scale),
                        task_residual=residual_target,
                        q005_delta=fit["kf_delta"], task_target=fit["task_target"])
    training = task.train_head("task_head", (fit["feature"] - mean) / std,
                               residual_target, batches, scale, 20261028, domain)
    residuals = {}
    for role in ROLES:
        residuals[role] = task.infer_head(
            "task_head", (qframes[role]["feature"] - mean) / std, scale, domain)
        np.savez_compressed(domain / f"{role}_frame_and_residual.npz",
                            feature=qframes[role]["feature"], q005_delta=qframes[role]["kf_delta"],
                            task_residual=residuals[role], task_target=qframes[role]["task_target"],
                            task_stats=qframes[role]["task_stats"])

    alpha_rows = []
    for alpha in contract["alpha_grid"]:
        panels = task_core.raw_panels(
            materials["calibration8"],
            {"q005_task_refit": qframes["calibration8"]["kf_delta"]
             + float(alpha) * residuals["calibration8"]})
        alpha_rows.append({"alpha": float(alpha),
                           "mean_ap70": float(np.mean([row["ap70"] for row in panels])),
                           "panels": panels})
    selected = max(alpha_rows, key=lambda row: (row["mean_ap70"], -row["alpha"]))
    calibration = {"criterion": "mean raw AP70 over calibration8 0/400ms",
                   "alpha_grid": contract["alpha_grid"],
                   "selected_alpha": selected["alpha"], "tables": alpha_rows}
    atomic(domain / "calibration.json", calibration)

    q005_head = ROOT / contract["domains"][name]["matched_q005_head"]
    panels_by_role = {}; outputs = {}; comparisons = []
    for role in FORMAL:
        q005_boxes, q005_features, _ = fixed.base.compose(materials[role], deltas[role]["q005"])
        task_delta = qframes[role]["kf_delta"] + float(selected["alpha"]) * residuals[role]
        task_boxes, task_features = task_geometry(materials[role], task_delta)
        q005_score = score_with_head(materials[role], q005_features, q005_head)
        task_score = score_with_head(materials[role], task_features, q005_head)
        old_task = load_npz(ROOT / spec["roles"][role]["task_material"])
        boxes_by_method = {
            "q005_matched39d_control": q005_boxes,
            "q005_task_refit_fixed_matched39d": task_boxes,
            "q005_task_refit_fresh_matched39d": task_boxes,
            "old_q015_task_fresh39d_diagnostic": old_task["task_boxes"],
        }
        scores_by_method = {
            "q005_matched39d_control": q005_score,
            "q005_task_refit_fixed_matched39d": q005_score,
            "q005_task_refit_fresh_matched39d": task_score,
            "old_q015_task_fresh39d_diagnostic": old_task["task_direct_u"],
        }
        panels, prediction = fixed.base.evaluate(materials[role], boxes_by_method, scores_by_method)
        panels_by_role[role] = panels
        prediction_path = domain / f"{role}_predictions.npz"
        material_path = domain / f"{role}_task_material.npz"
        np.savez_compressed(prediction_path, **prediction)
        np.savez_compressed(material_path, task_delta=task_delta, task_boxes=task_boxes,
                            task_features=task_features, q005_matched_score=q005_score,
                            task_fresh_matched_score=task_score,
                            condition_offsets=materials[role]["condition_offsets"],
                            dataset_index=materials[role]["dataset_index"],
                            sequence=materials[role]["sequence"], wait_ms=materials[role]["wait_ms"],
                            receiver_timestamp_us=materials[role]["receiver_timestamp_us"])
        outputs[role] = {"predictions": ev(prediction_path), "material": ev(material_path)}
        table = {(int(row["wait_ms"]), row["method"]): row for row in panels}
        for wait in sorted(map(int, np.unique(materials[role]["wait_ms"]))):
            base = table[(wait, "q005_matched39d_control")]
            fresh = table[(wait, "q005_task_refit_fresh_matched39d")]
            fixed_row = table[(wait, "q005_task_refit_fixed_matched39d")]
            old = table[(wait, "old_q015_task_fresh39d_diagnostic")]
            comparisons.append({
                "role": role, "wait_ms": wait,
                "fresh_refit_minus_q005_matched_ap70_pp": fresh["ap70"] - base["ap70"],
                "fixed_refit_minus_q005_matched_ap70_pp": fixed_row["ap70"] - base["ap70"],
                "conditional_readout_ap70_pp": fresh["ap70"] - fixed_row["ap70"],
                "fresh_refit_minus_old_q015_task_ap70_pp": fresh["ap70"] - old["ap70"],
                "fresh_refit_minus_q005_matched_tp70": fresh["tp70"] - base["tp70"],
                "fresh_refit_minus_q005_matched_fp70": fresh["fp70"] - base["fp70"],
            })
    signs = Counter("positive" if row["fresh_refit_minus_q005_matched_ap70_pp"] > 0
                    else "negative" if row["fresh_refit_minus_q005_matched_ap70_pp"] < 0 else "zero"
                    for row in comparisons)
    checks = {
        "q005_state_matches_fixed_temporal_runner": max(q005_errors.values()) <= 1e-12,
        "old_batches_reused_exact": np.array_equal(batches, protocol["batches"]),
        "same_seed_updates_and_head_structure": training["seed"] == 20261028
            and training["updates"] == 2000 and training["parameters"] == 866,
        "fit24_only_task_target": all((role == "fit24") == bool(np.isfinite(qframes[role]["task_target"]).all())
                                      for role in ROLES),
        "single_alpha_selected_on_calibration8": selected["alpha"] in contract["alpha_grid"],
        "all_formal_roles_waits_retained": len(comparisons) == 16,
        "zero_detector_backbone_forward": True, "zero_gpu": True,
    }
    if not all(checks.values()): raise RuntimeError((name, checks))
    return {"identity": spec["identity"], "training": training,
            "fit_scale_m": scale, "fit_scale_raw_q95_m": quantile,
            "calibration": calibration, "q005_state_max_error": q005_errors,
            "formal_panels": panels_by_role, "comparisons": comparisons,
            "sign_counts_vs_q005_matched": dict(signs), "outputs": outputs,
            "checks": checks,
            "artifacts": {"head": ev(domain / "task_head.pth"),
                          "fit_protocol": ev(domain / "fit_protocol.npz"),
                          "calibration": ev(domain / "calibration.json")}}


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    fixed_contract = load(ROOT / contract["fixed_contract"]["path"])
    domains = {name: domain_compute(name, fixed_contract["domains"][name], contract, output)
               for name in DOMAINS}
    checks = {"exact_domains": set(domains) == set(DOMAINS),
              "all_domain_checks": all(all(value["checks"].values()) for value in domains.values()),
              "zero_detector_backbone_forward": True, "zero_gpu": True}
    if not all(checks.values()): raise RuntimeError(checks)
    summary = {
        "schema": "crosscheckpoint_temporal_q005_task_refit_producer_v1",
        "status": "complete_producer_review_pending", "created_at": contract["frozen_at"],
        "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": ev(ROOT / contract["contract_path"]), "domains": domains,
        "checks": checks,
        "actual_cost": {"wall_seconds": time.monotonic() - started,
                        "cpu_process_seconds": time.process_time(),
                        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                        "task_head_updates": 4000, "detector_or_backbone_forwards": 0,
                        "gpu_seconds": 0},
        "boundary": (
            "Q005-base current task teacher/head refit for seed2025 and Where only. Fit24 supplies targets, "
            "old batches/schedule are reused, calibration8 selects one alpha, and all exposed formal cells "
            "are retained. Cached module evidence only: no detector forward, hidden confirmation, sequence "
            "selection, complete-system cost, or final core claim."
        ),
    }
    atomic(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"],
                      "selected_alpha": {name: value["calibration"]["selected_alpha"]
                                         for name, value in domains.items()},
                      "signs": {name: value["sign_counts_vs_q005_matched"] for name, value in domains.items()},
                      "actual_cost": summary["actual_cost"]}, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "": raise RuntimeError("CUDA must be hidden")
    contract = load(args.contract)
    if contract.get("task_id") != TASK or contract.get("status") != "frozen_released":
        raise RuntimeError("contract identity/status")
    if sha(Path(__file__).resolve()) != contract["runner"]["sha256"]: raise RuntimeError("runner drift")
    for item in contract["inputs"].values():
        if sha(ROOT / item["path"]) != item["sha256"]: raise RuntimeError(("input drift", item["path"]))
    compute(contract, args.output)


if __name__ == "__main__": main()
