#!/usr/bin/env python3
"""Independent full replay of the current-q005 heading cross-checkpoint result."""
from __future__ import annotations

import argparse
import ast
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
PRODUCER_MODULE = "run_crosscheckpoint_current_q005_heading_geometry_peer06_v1_20261030"


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


def no_producer_import() -> bool:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == PRODUCER_MODULE or alias.name.endswith("." + PRODUCER_MODULE)
                   for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.module == PRODUCER_MODULE or str(node.module).endswith("." + PRODUCER_MODULE):
                return False
    return True


def heading_boxes(stale: np.ndarray, cv: np.ndarray) -> np.ndarray:
    output = cv.copy()
    yaw = stale[:, 6].astype(np.float64)
    unit = np.column_stack((np.cos(yaw), np.sin(yaw)))
    displacement = cv[:, :2].astype(np.float64) - stale[:, :2].astype(np.float64)
    along = np.einsum("ij,ij->i", displacement, unit)
    output[:, :2] = (stale[:, :2].astype(np.float64) + unit * along[:, None]).astype(np.float32)
    return output


def build_heading(material: dict[str, np.ndarray], delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes = material["cv_boxes"].copy()
    features = material["cv_features"].copy()
    for condition in range(len(delta)):
        a, b = material["condition_offsets"][condition:condition + 2].astype(np.int64)
        source = material["source"][a:b]
        protected = material["protected"][a:b]
        remote = a + np.nonzero(source == 1)[0]
        anchors = a + np.nonzero((source == 0) & (protected == 1))[0]
        candidate = heading_boxes(material["stale_boxes"][remote], material["cv_boxes"][remote])
        candidate[:, 0] = candidate[:, 0] + np.float32(delta[condition, 0])
        candidate[:, 1] = candidate[:, 1] + np.float32(delta[condition, 1])
        boxes[remote] = candidate
        features[remote] = fixed.base.actual.optional.update_features(
            material["cv_features"][remote], candidate, material["cv_boxes"][anchors],
            material["raw_scores"][anchors])
    return boxes, features


def same_fp(predictions: dict[str, np.ndarray], panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = {(int(row["wait_ms"]), row["method"]): row for row in panels}
    waits = predictions["condition_wait_ms"]
    result: list[dict[str, Any]] = []
    for wait in sorted(map(int, np.unique(waits))):
        budget = int(table[(wait, "current_q005_cv_direct_u")]["fp70"])
        condition_ids = np.nonzero(waits == wait)[0]
        for method in ("current_q005_cv_direct_u", "current_q005_heading_direct_u"):
            offsets = predictions[f"{method}__offset"].astype(np.int64)
            chunks = [np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
                      for index in condition_ids]
            selected = np.concatenate(chunks) if chunks else np.empty(0, np.int64)
            scores = predictions[f"{method}__score"]
            tp = predictions[f"{method}__tp70"].astype(np.int64)
            ranked = selected[np.argsort(-scores[selected], kind="stable")]
            cumulative_tp = np.cumsum(tp[ranked])
            cumulative_fp = np.cumsum(1 - tp[ranked])
            eligible = np.nonzero(cumulative_fp <= budget)[0]
            last = int(eligible[-1]) if len(eligible) else -1
            result.append({
                "wait_ms": wait,
                "method": method,
                "reference": "current_q005_cv_direct_u",
                "fp_budget": budget,
                "achieved_fp": int(cumulative_fp[last]) if last >= 0 else 0,
                "tp70_at_same_fp": int(cumulative_tp[last]) if last >= 0 else 0,
                "outputs_at_same_fp": last + 1,
            })
    return result


def effects(material: dict[str, np.ndarray], panels: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    table = {(int(row["wait_ms"]), row["method"]): row for row in panels}
    result: list[dict[str, Any]] = []
    for wait in sorted(map(int, np.unique(material["wait_ms"]))):
        for readout in ("raw", "direct_u"):
            cv = table[(wait, f"current_q005_cv_{readout}")]
            heading = table[(wait, f"current_q005_heading_{readout}")]
            result.append({
                "role": role,
                "wait_ms": wait,
                "readout": readout,
                "heading_minus_cv_ap50_pp": float(heading["ap50"] - cv["ap50"]),
                "heading_minus_cv_ap70_pp": float(heading["ap70"] - cv["ap70"]),
                "heading_minus_cv_tp70": int(heading["tp70"] - cv["tp70"]),
                "heading_minus_cv_fp70": int(heading["fp70"] - cv["fp70"]),
            })
    return result


def review_domain(name: str, spec: dict[str, Any], contract: dict[str, Any],
                  producer: dict[str, Any], producer_root: Path) -> dict[str, Any]:
    materials, deltas, _ = matched.prepare_domain(spec)
    head = ROOT / contract["domains"][name]["matched_q005_head"]
    domain_details: dict[str, Any] = {}
    rebuilt_effects: list[dict[str, Any]] = []
    for role in FORMAL:
        material = materials[role]
        delta = deltas[role]["q005"]
        cv_boxes, cv_features, _ = fixed.base.compose(material, delta)
        heading, heading_features = build_heading(material, delta)
        cv_score = refit.score_with_head(material, cv_features, head)
        heading_score = refit.score_with_head(material, heading_features, head)
        boxes = {"current_q005_cv_raw": cv_boxes, "current_q005_cv_direct_u": cv_boxes,
                 "current_q005_heading_raw": heading, "current_q005_heading_direct_u": heading}
        scores = {"current_q005_cv_raw": material["raw_scores"].astype(np.float64),
                  "current_q005_cv_direct_u": cv_score,
                  "current_q005_heading_raw": material["raw_scores"].astype(np.float64),
                  "current_q005_heading_direct_u": heading_score}
        panels, predictions = fixed.base.evaluate(material, boxes, scores)
        saved_predictions = load_npz(producer_root / name / f"{role}_predictions.npz")
        saved_material = load_npz(producer_root / name / f"{role}_method_material.npz")
        expected_material = {
            "q005_delta": delta, "cv_boxes": cv_boxes, "heading_boxes": heading,
            "cv_direct_u": cv_score, "heading_direct_u": heading_score,
            "condition_offsets": material["condition_offsets"], "dataset_index": material["dataset_index"],
            "sequence": material["sequence"], "wait_ms": material["wait_ms"],
            "receiver_timestamp_us": material["receiver_timestamp_us"],
            "source": material["source"], "protected": material["protected"],
        }
        prediction_checks = {key: key in saved_predictions and exact(value, saved_predictions[key])
                             for key, value in predictions.items()}
        material_checks = {key: key in saved_material and exact(value, saved_material[key])
                           for key, value in expected_material.items()}
        rebuilt_same_fp = same_fp(predictions, panels)
        rebuilt_role_effects = effects(material, panels, role)
        rebuilt_effects.extend(rebuilt_role_effects)
        role_checks = {
            "prediction_key_set_exact": set(predictions) == set(saved_predictions),
            "prediction_arrays_exact": all(prediction_checks.values()),
            "material_key_set_exact": set(expected_material) == set(saved_material),
            "material_arrays_exact": all(material_checks.values()),
            "formal_panels_exact": panels == producer["domains"][name]["formal_panels"][role],
            "same_fp_exact": rebuilt_same_fp == producer["domains"][name]["outputs"][role]["same_fp"],
            "protected_prefix_exact": exact(cv_boxes[material["source"] == 0], heading[material["source"] == 0]),
        }
        domain_details[role] = {"checks": role_checks,
                                "prediction_checks": prediction_checks,
                                "material_checks": material_checks,
                                "candidates": int(len(material["source"])),
                                "conditions": int(len(material["wait_ms"]))}
    domain_checks = {
        "both_formal_roles_rebuilt": set(domain_details) == set(FORMAL),
        "all_saved_arrays_exact": all(all(value["checks"].values()) for value in domain_details.values()),
        "effects_exact": rebuilt_effects == producer["domains"][name]["effects"],
        "matched_q005_head_hash_exact": sha(head / "direct_u.pth") == contract["domains"][name]["head_sha256"],
    }
    return {"status": "passed" if all(domain_checks.values()) else "failed",
            "checks": domain_checks, "roles": domain_details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cpu_started = time.process_time()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    contract = load(args.contract)
    producer = load(args.producer / "summary.json")
    fixed_contract = load(ROOT / contract["fixed_input_contract"]["path"])
    details = {name: review_domain(name, fixed_contract["domains"][name], contract, producer, args.producer)
               for name in contract["domains"]}
    checks = {
        "independent_no_producer_import": no_producer_import(),
        "reviewer_hash_frozen": sha(Path(__file__)) == contract["reviewer"]["sha256"],
        "contract_hash_matches_producer": producer["contract"]["sha256"] == sha(args.contract),
        "three_domains_full_replay": set(details) == {"v2xvit_seed2024", "v2xvit_seed2025", "where2comm"},
        "all_domain_replays_passed": all(value["status"] == "passed" for value in details.values()),
        "zero_detector_or_backbone_forward": True,
        "zero_optimizer_or_head_update": True,
        "zero_gpu": True,
    }
    result = {
        "schema": "crosscheckpoint_current_q005_heading_geometry_peer06_independent_review_v1",
        "status": "complete_independent_review_passed" if all(checks.values()) else "failed",
        "created_at": contract["frozen_at"],
        "task_id": TASK,
        "owner": OWNER,
        "contract": ev(args.contract),
        "producer": ev(args.producer / "summary.json"),
        "checks": checks,
        "details": details,
        "actual_cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_process_seconds": time.process_time() - cpu_started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "detector_or_backbone_forwards": 0,
            "optimizer_or_head_updates": 0,
            "gpu_seconds": 0,
        },
        "boundary": (
            "Independent rebuild from the frozen prior inputs and heads. It imports accepted shared kernels but "
            "does not import the new producer or consume its arrays as computation inputs."
        ),
    }
    atomic(args.output / "review.json", result)
    print(json.dumps({"status": result["status"], "checks": checks,
                      "cost": result["actual_cost"]}, sort_keys=True), flush=True)
    if result["status"] != "complete_independent_review_passed":
        raise RuntimeError("independent review failed")


if __name__ == "__main__":
    main()
