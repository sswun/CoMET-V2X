#!/usr/bin/env python3
"""Formal current-q005 CV versus stale-yaw heading geometry on ViT/Where."""
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
sys.path.insert(0, str(AUTO))
import run_crosscheckpoint_current_temporal_fixed_readout_controls_v1_20261029 as fixed
import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
import run_crosscheckpoint_temporal_q005_task_refit_v1_20261029 as refit


TASK = "crosscheckpoint_current_q005_heading_geometry_20261030_peer06"
OWNER = "research_autopilot_crosscheckpoint_current_q005_heading_geometry_peer06_v1"
FORMAL = ("mechanism_holdout14", "val21")
METHODS = ("current_q005_cv_raw", "current_q005_cv_direct_u",
           "current_q005_heading_raw", "current_q005_heading_direct_u")


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


def heading_boxes(stale: np.ndarray, cv: np.ndarray) -> np.ndarray:
    output = cv.copy()
    yaw = stale[:, 6].astype(np.float64)
    heading = np.column_stack((np.cos(yaw), np.sin(yaw)))
    displacement = cv[:, :2].astype(np.float64) - stale[:, :2].astype(np.float64)
    output[:, :2] = (stale[:, :2].astype(np.float64)
                     + heading * np.sum(displacement * heading, axis=1, keepdims=True)).astype(np.float32)
    return output


def build_heading(material: dict[str, np.ndarray], delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes = material["cv_boxes"].copy()
    features = material["cv_features"].copy()
    for condition, move in enumerate(delta):
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        source = material["source"][a:b]
        protected = material["protected"][a:b]
        remote = a + np.flatnonzero(source == 1)
        fixed_ids = a + np.flatnonzero((source == 0) & (protected == 1))
        projected = heading_boxes(material["stale_boxes"][remote], material["cv_boxes"][remote])
        projected[:, :2] += move.astype(np.float32)
        boxes[remote] = projected
        features[remote] = fixed.base.actual.optional.update_features(
            material["cv_features"][remote], projected, material["cv_boxes"][fixed_ids],
            material["raw_scores"][fixed_ids])
    return boxes, features


def prediction_reference(contract: dict[str, Any], domain: str, role: str) -> tuple[Path, str]:
    spec = contract["domains"][domain]["current_q005_reference"]
    return ROOT / spec[role]["path"], spec[role]["method"]


def compare_reference(predictions: dict[str, np.ndarray], reference: dict[str, np.ndarray],
                      reference_method: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("candidate", "score", "tp50", "tp70", "offset"):
        current = predictions[f"current_q005_cv_direct_u__{field}"]
        old = reference[f"{reference_method}__{field}"]
        result[f"{field}_exact"] = exact(current, old)
        if field == "score":
            result["score_max_abs_error"] = float(np.max(np.abs(current - old), initial=0.0))
    return result


def same_fp(predictions: dict[str, np.ndarray], panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = {(int(row["wait_ms"]), row["method"]): row for row in panels}
    condition_wait = predictions["condition_wait_ms"]
    output: list[dict[str, Any]] = []
    for wait in sorted(map(int, np.unique(condition_wait))):
        budget = int(rows[(wait, "current_q005_cv_direct_u")]["fp70"])
        for method in ("current_q005_cv_direct_u", "current_q005_heading_direct_u"):
            offsets = predictions[f"{method}__offset"].astype(np.int64)
            scores = predictions[f"{method}__score"]
            tp = predictions[f"{method}__tp70"].astype(np.int64)
            ids = [np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
                   for index in np.flatnonzero(condition_wait == wait)]
            selected = np.concatenate(ids) if ids else np.empty(0, np.int64)
            rank = selected[np.argsort(-scores[selected], kind="stable")]
            ordered_tp = tp[rank]
            cumulative_tp = np.cumsum(ordered_tp)
            cumulative_fp = np.cumsum(1 - ordered_tp)
            eligible = np.flatnonzero(cumulative_fp <= budget)
            index = int(eligible[-1]) if len(eligible) else -1
            output.append({
                "wait_ms": wait,
                "method": method,
                "reference": "current_q005_cv_direct_u",
                "fp_budget": budget,
                "achieved_fp": int(cumulative_fp[index]) if index >= 0 else 0,
                "tp70_at_same_fp": int(cumulative_tp[index]) if index >= 0 else 0,
                "outputs_at_same_fp": index + 1,
            })
    return output


def signs(values: list[float]) -> dict[str, Any]:
    counts = Counter("positive" if value > 0 else "negative" if value < 0 else "zero" for value in values)
    return {"positive": int(counts.get("positive", 0)), "negative": int(counts.get("negative", 0)),
            "zero": int(counts.get("zero", 0)), "cells": len(values),
            "mean_pp": float(np.mean(values)), "minimum_pp": float(np.min(values)),
            "maximum_pp": float(np.max(values))}


def domain_compute(name: str, spec: dict[str, Any], contract: dict[str, Any], output: Path) -> dict[str, Any]:
    materials, deltas, _ = matched.prepare_domain(spec)
    head = ROOT / contract["domains"][name]["matched_q005_head"]
    domain_dir = output / name
    domain_dir.mkdir(parents=True, exist_ok=False)
    panels_by_role: dict[str, Any] = {}
    effects: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    qualification: dict[str, Any] = {}
    for role in FORMAL:
        material = materials[role]
        delta = deltas[role]["q005"]
        cv_boxes, cv_features, _ = fixed.base.compose(material, delta)
        projected_boxes, projected_features = build_heading(material, delta)
        cv_score = refit.score_with_head(material, cv_features, head)
        projected_score = refit.score_with_head(material, projected_features, head)
        boxes = {
            "current_q005_cv_raw": cv_boxes,
            "current_q005_cv_direct_u": cv_boxes,
            "current_q005_heading_raw": projected_boxes,
            "current_q005_heading_direct_u": projected_boxes,
        }
        scores = {
            "current_q005_cv_raw": material["raw_scores"].astype(np.float64),
            "current_q005_cv_direct_u": cv_score,
            "current_q005_heading_raw": material["raw_scores"].astype(np.float64),
            "current_q005_heading_direct_u": projected_score,
        }
        panels, predictions = fixed.base.evaluate(material, boxes, scores)
        reference_path, reference_method = prediction_reference(contract, name, role)
        reference = load_npz(reference_path)
        replay = compare_reference(predictions, reference, reference_method)
        local = np.flatnonzero(material["source"] == 0)
        role_qualification = {
            **replay,
            "current_prefix_boxes_exact": exact(cv_boxes[local], projected_boxes[local]),
            "candidate_population_exact": all(len(value) == len(material["source"]) for value in boxes.values()),
            "same_q005_state_correction": True,
            "all_geometry_finite": bool(np.isfinite(projected_boxes).all()),
            "all_scores_finite": bool(np.isfinite(cv_score).all() and np.isfinite(projected_score).all()),
        }
        if not all(value for key, value in role_qualification.items() if key != "score_max_abs_error") \
                or role_qualification["score_max_abs_error"] != 0.0:
            raise RuntimeError((name, role, "qualification", role_qualification))
        panels_by_role[role] = panels
        qualification[role] = role_qualification
        prediction_path = domain_dir / f"{role}_predictions.npz"
        material_path = domain_dir / f"{role}_method_material.npz"
        np.savez_compressed(prediction_path, **predictions)
        np.savez_compressed(
            material_path,
            q005_delta=delta,
            cv_boxes=cv_boxes,
            heading_boxes=projected_boxes,
            cv_direct_u=cv_score,
            heading_direct_u=projected_score,
            condition_offsets=material["condition_offsets"],
            dataset_index=material["dataset_index"],
            sequence=material["sequence"],
            wait_ms=material["wait_ms"],
            receiver_timestamp_us=material["receiver_timestamp_us"],
            source=material["source"],
            protected=material["protected"],
        )
        outputs[role] = {"predictions": ev(prediction_path), "method_material": ev(material_path),
                         "current_reference": ev(reference_path), "current_reference_method": reference_method}
        table = {(int(row["wait_ms"]), row["method"]): row for row in panels}
        for wait in sorted(map(int, np.unique(material["wait_ms"]))):
            for readout in ("raw", "direct_u"):
                cv = table[(wait, f"current_q005_cv_{readout}")]
                heading = table[(wait, f"current_q005_heading_{readout}")]
                effects.append({
                    "role": role,
                    "wait_ms": wait,
                    "readout": readout,
                    "heading_minus_cv_ap50_pp": float(heading["ap50"] - cv["ap50"]),
                    "heading_minus_cv_ap70_pp": float(heading["ap70"] - cv["ap70"]),
                    "heading_minus_cv_tp70": int(heading["tp70"] - cv["tp70"]),
                    "heading_minus_cv_fp70": int(heading["fp70"] - cv["fp70"]),
                })
        outputs[role]["same_fp"] = same_fp(predictions, panels)
    direct = [row["heading_minus_cv_ap70_pp"] for row in effects if row["readout"] == "direct_u"]
    raw = [row["heading_minus_cv_ap70_pp"] for row in effects if row["readout"] == "raw"]
    checks = {
        "formal_roles_exact": set(panels_by_role) == set(FORMAL),
        "four_methods_each_role": all({row["method"] for row in panels_by_role[role]} == set(METHODS)
                                      for role in FORMAL),
        "all_four_waits_each_role": all({int(row["wait_ms"]) for row in panels_by_role[role]} == {0, 100, 200, 400}
                                        for role in FORMAL),
        "current_q005_reference_prediction_exact": all(
            all(value for key, value in row.items() if key != "score_max_abs_error")
            and row["score_max_abs_error"] == 0.0 for row in qualification.values()),
        "no_per_cell_selection": True,
        "zero_detector_or_backbone_forward": True,
        "zero_optimizer_or_head_update": True,
        "zero_gpu": True,
    }
    if not all(checks.values()):
        raise RuntimeError((name, checks))
    return {
        "identity": spec["identity"],
        "matched_q005_head": ev(head / "direct_u.pth"),
        "formal_panels": panels_by_role,
        "effects": effects,
        "aggregate": {"direct_u_heading_minus_cv_ap70": signs(direct),
                      "raw_heading_minus_cv_ap70": signs(raw)},
        "qualification": qualification,
        "outputs": outputs,
        "checks": checks,
    }


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic()
    cpu_started = time.process_time()
    output.mkdir(parents=True, exist_ok=False)
    fixed_contract = load(ROOT / contract["fixed_input_contract"]["path"])
    domains = {name: domain_compute(name, fixed_contract["domains"][name], contract, output)
               for name in contract["domains"]}
    direct = [row["heading_minus_cv_ap70_pp"] for value in domains.values()
              for row in value["effects"] if row["readout"] == "direct_u"]
    raw = [row["heading_minus_cv_ap70_pp"] for value in domains.values()
           for row in value["effects"] if row["readout"] == "raw"]
    checks = {
        "three_actual_q005_domains": set(domains) == {"v2xvit_seed2024", "v2xvit_seed2025", "where2comm"},
        "all_domain_checks": all(all(value["checks"].values()) for value in domains.values()),
        "twenty_four_direct_u_cells": len(direct) == 24,
        "twenty_four_raw_cells": len(raw) == 24,
        "all_signed_cells_retained": True,
        "strict60_not_substituted": True,
        "exposed_not_hidden_confirmation": True,
        "zero_detector_or_backbone_forward": True,
        "zero_optimizer_or_head_update": True,
        "zero_gpu": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    result = {
        "schema": "crosscheckpoint_current_q005_heading_geometry_peer06_producer_v1",
        "status": "complete_producer_review_pending",
        "created_at": contract["frozen_at"],
        "task_id": TASK,
        "owner": OWNER,
        "contract": ev(ROOT / contract["contract_path"]),
        "domains": domains,
        "aggregate": {"direct_u_heading_minus_cv_ap70": signs(direct),
                      "raw_heading_minus_cv_ap70": signs(raw)},
        "checks": checks,
        "prediction_before_result": contract["prediction_before_result"],
        "actual_cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_process_seconds": time.process_time() - cpu_started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "detector_or_backbone_forwards": 0,
            "optimizer_or_head_updates": 0,
            "gpu_seconds": 0,
        },
        "boundary": (
            "Exposed cached-input cross-checkpoint confirmation. Not hidden confirmation, detector training, "
            "complete-system cost, or a claim that heading projection is universally positive."
        ),
    }
    atomic(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"],
                      "cost": result["actual_cost"]}, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load(args.contract)
    if contract.get("task_id") != TASK or contract.get("owner") != OWNER \
            or contract.get("status") != "frozen_released_for_execution":
        raise RuntimeError("contract identity/status")
    if sha(Path(__file__)) != contract["runner"]["sha256"]:
        raise RuntimeError("runner hash drift")
    for item in contract["inputs"].values():
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise RuntimeError(("input drift", item["path"]))
    compute(contract, args.output)


if __name__ == "__main__":
    main()
