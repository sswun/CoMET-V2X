#!/usr/bin/env python3
"""ViT2024/2025 and Where current-input calibrated temporal fixed-readout controls."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(AUTO))
import run_strict60_current_temporal_fixed_readout_controls_v1_20261029 as base  # noqa: E402


TASK = "crosscheckpoint_current_temporal_fixed_readout_controls_v1_20261029"
ROLES = ("fit24", "calibration8", "mechanism_holdout14", "val21")
FORMAL = ROLES[2:]
ARMS = base.ARMS
READOUTS = base.READOUTS
Q_BY_ARM = base.Q_BY_ARM


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def evidence(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    return base.exact(left, right)


def anchor_inputs(frame: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    feature = frame["feature"].astype(np.float64)
    anchors = np.rint(np.expm1(feature[:, 14])).astype(np.int64)
    if np.max(np.abs(np.log1p(anchors) - feature[:, 14]), initial=0.0) > 1e-12:
        raise RuntimeError("anchor count decode drift")
    measurements = feature[:, 5:7].copy()
    return anchors, measurements


def run_state(frame: dict[str, np.ndarray], sequence: np.ndarray, waits: np.ndarray,
              mode: str, constants: dict[int, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    feature = frame["feature"].astype(np.float64)
    anchors, measurements = anchor_inputs(frame)
    states: dict[tuple[str, int], Any] = {}
    result = []
    errors = {"q015_prior_mean": 0.0, "q015_prior_log_variance": 0.0, "q015_state": 0.0}
    for ordinal, (sequence_value, wait_value) in enumerate(zip(sequence, waits)):
        key = (str(sequence_value), int(wait_value))
        count = int(anchors[ordinal])
        measurement = measurements[ordinal] if count else None
        if mode == "cv":
            value = np.zeros(2, np.float64)
        elif mode == "fit_constant":
            value = constants[key[1]]
        elif mode == "cumulative_anchor_mean":
            total, population = states.get(key, (np.zeros(2, np.float64), 0))
            if measurement is not None:
                total = total + measurement * count
                population += count
            value = total / population if population else np.zeros(2, np.float64)
            states[key] = (total, population)
        else:
            process = float(Q_BY_ARM[mode])
            mean, variance = states.get(key, (np.zeros(2, np.float64), 1.0))
            prior = variance + process
            if mode == "q015":
                errors["q015_prior_mean"] = max(
                    errors["q015_prior_mean"], float(np.max(np.abs(mean - feature[ordinal, :2]))))
                errors["q015_prior_log_variance"] = max(
                    errors["q015_prior_log_variance"], abs(float(np.log(prior) - feature[ordinal, 2])))
            if measurement is not None:
                gain = prior / (prior + .6 ** 2 / count)
                mean = mean + gain * (measurement - mean)
                variance = (1.0 - gain) * prior
            else:
                variance = prior
            value = mean
            states[key] = (mean.copy(), variance)
            if mode == "q015":
                errors["q015_state"] = max(
                    errors["q015_state"], float(np.max(np.abs(mean - feature[ordinal, 3:5]))))
        result.append(np.asarray(value, np.float64))
    return np.asarray(result, np.float64), errors


def fit_constants(frame: dict[str, np.ndarray], waits: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    anchors, measurements = anchor_inputs(frame)
    constants: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for wait in sorted(map(int, np.unique(waits))):
        use = (waits == wait) & (anchors > 0)
        count = int(anchors[use].sum())
        constants[wait] = ((measurements[use] * anchors[use, None]).sum(axis=0) / count
                           if count else np.zeros(2, np.float64))
        counts[wait] = count
    return constants, counts


def identity(population: dict[str, np.ndarray], frame: dict[str, np.ndarray]) -> None:
    for key in ("dataset_index", "sequence", "wait_ms", "receiver_timestamp_us"):
        if not exact(population[key], frame[key]):
            raise RuntimeError((key, "population/frame identity drift"))
    if not exact(frame["kf_delta"], frame["feature"][:, 3:5]):
        raise RuntimeError("frame kf state drift")


def reconstruct_vit_material(population: dict[str, np.ndarray], reference: dict[str, np.ndarray],
                             formal_role: bool) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray,
                                                         dict[str, float]]:
    common = ("raw_scores", "source", "protected", "anchor", "condition_offsets",
              "dataset_index", "sequence", "wait_ms", "receiver_timestamp_us")
    for key in common:
        if key in reference and not exact(population[key], reference[key]):
            raise RuntimeError((key, "ViT population/reference drift"))
    if formal_role:
        cv_boxes, cv_features = reference["cv_boxes"], reference["cv_features"]
        kf_boxes, kf_features = reference["kf_boxes"], reference["kf_features"]
        if not exact(population["cv_boxes"], cv_boxes):
            raise RuntimeError("ViT formal CV boxes drift")
    else:
        cv_boxes = population["cv_boxes"]
        kf_boxes, kf_features = reference["boxes"], reference["features"]
        cv_features = kf_features.copy()
        for condition in range(len(population["wait_ms"])):
            a, b = population["condition_offsets"][condition:condition + 2].astype(int)
            source, protected = population["source"][a:b], population["protected"][a:b]
            remote = a + np.flatnonzero(source == 1)
            fixed = a + np.flatnonzero((source == 0) & (protected == 1))
            if len(remote):
                cv_features[remote] = base.actual.optional.update_features(
                    kf_features[remote], cv_boxes[remote], cv_boxes[fixed],
                    population["raw_scores"][fixed])
    material = dict(population)
    material["cv_boxes"] = cv_boxes.copy()
    material["cv_features"] = cv_features.copy()
    checks = {"cv_boxes_population_max_error": float(np.max(
        np.abs(population["cv_boxes"] - cv_boxes), initial=0.0))}
    return material, kf_boxes, kf_features, checks


def build_methods(material: dict[str, np.ndarray], deltas: dict[str, np.ndarray], readout: Path,
                  source_maps: dict[str, Any], task: dict[str, np.ndarray] | None):
    mapping = source_maps["maps"]["direct_to_utility"]
    fixed = base.actual.optional.direct_scores({"base": material["cv_features"]}, readout)["base"]
    fixed = base.actual.optional.source_map(fixed, material["source"], mapping)
    boxes_by_method: dict[str, np.ndarray] = {}
    scores_by_method: dict[str, np.ndarray] = {}
    built: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        boxes, features, _ = base.compose(material, deltas[arm])
        fresh = base.actual.optional.direct_scores({"fresh": features}, readout)["fresh"]
        fresh = base.actual.optional.source_map(fresh, material["source"], mapping)
        boxes_by_method[f"{arm}_raw"] = boxes
        scores_by_method[f"{arm}_raw"] = material["raw_scores"].astype(np.float64)
        boxes_by_method[f"{arm}_fixed39d"] = boxes
        scores_by_method[f"{arm}_fixed39d"] = fixed
        boxes_by_method[f"{arm}_fresh39d"] = boxes
        scores_by_method[f"{arm}_fresh39d"] = fresh
        built[arm] = {"boxes": boxes, "features": features, "fresh39d": fresh}
    if task is not None:
        for key in ("condition_offsets", "dataset_index", "sequence", "wait_ms", "receiver_timestamp_us"):
            if not exact(material[key], task[key]):
                raise RuntimeError((key, "task material identity drift"))
        boxes_by_method["current_task_fresh39d"] = task["task_boxes"]
        scores_by_method["current_task_fresh39d"] = task["task_direct_u"]
    return boxes_by_method, scores_by_method, built, fixed


def domain_compute(name: str, spec: dict[str, Any], output: Path) -> dict[str, Any]:
    readout = ROOT / spec["readout_root"]
    source_maps = load(readout / "source_maps.json")
    populations: dict[str, dict[str, np.ndarray]] = {}
    frames: dict[str, dict[str, np.ndarray]] = {}
    materials: dict[str, dict[str, np.ndarray]] = {}
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    identity_checks: dict[str, Any] = {}
    for role in ROLES:
        role_spec = spec["roles"][role]
        populations[role] = load_npz(ROOT / role_spec["population"])
        frames[role] = load_npz(ROOT / role_spec["frame_input"])
        identity(populations[role], frames[role])
        if spec["material_kind"] == "vit_reconstructed":
            reference = load_npz(ROOT / role_spec["reference"])
            materials[role], kf_boxes, kf_features, checks = reconstruct_vit_material(
                populations[role], reference, role in FORMAL)
        else:
            materials[role] = dict(populations[role])
            kf_boxes, kf_features = materials[role]["kf_boxes"], materials[role]["kf_features"]
            checks = {"cv_boxes_population_max_error": 0.0}
        references[role] = (kf_boxes, kf_features)
        identity_checks[role] = checks

    waits = sorted(map(int, np.unique(populations["fit24"]["wait_ms"])))
    if any(sorted(map(int, np.unique(populations[role]["wait_ms"]))) != waits for role in ROLES):
        raise RuntimeError((name, "wait grid drift"))
    constants, constant_counts = fit_constants(frames["fit24"], populations["fit24"]["wait_ms"])
    deltas: dict[str, dict[str, np.ndarray]] = {}
    q015_errors = {"prior_mean": 0.0, "prior_log_variance": 0.0, "state": 0.0,
                   "boxes": 0.0, "features": 0.0, "fresh39d": 0.0}
    for role in ROLES:
        deltas[role] = {}
        for arm in ARMS:
            value, errors = run_state(frames[role], populations[role]["sequence"],
                                      populations[role]["wait_ms"], arm, constants)
            deltas[role][arm] = value
            if arm == "q015":
                q015_errors["prior_mean"] = max(q015_errors["prior_mean"], errors["q015_prior_mean"])
                q015_errors["prior_log_variance"] = max(
                    q015_errors["prior_log_variance"], errors["q015_prior_log_variance"])
                q015_errors["state"] = max(q015_errors["state"], errors["q015_state"])
        q_boxes, q_features, _ = base.compose(materials[role], deltas[role]["q015"])
        q015_errors["boxes"] = max(q015_errors["boxes"], float(np.max(
            np.abs(q_boxes - references[role][0]), initial=0.0)))
        q015_errors["features"] = max(q015_errors["features"], float(np.max(
            np.abs(q_features - references[role][1]), initial=0.0)))
        identity_checks[role].update({
            "q015_boxes_max_error": float(np.max(np.abs(q_boxes - references[role][0]), initial=0.0)),
            "q015_features_max_error": float(np.max(np.abs(q_features - references[role][1]), initial=0.0)),
        })

    cal_boxes, cal_scores, cal_built, _ = build_methods(
        materials["calibration8"], deltas["calibration8"], readout, source_maps, None)
    calibration_panels, _ = base.evaluate(materials["calibration8"], cal_boxes, cal_scores)
    table = {(int(row["wait_ms"]), row["method"]): row for row in calibration_panels}
    objectives = {arm: float(np.mean([table[(wait, f"{arm}_fresh39d")]["ap70"] for wait in waits]))
                  for arm in Q_BY_ARM}
    q_arms = tuple(Q_BY_ARM)
    selected = max(q_arms, key=lambda arm: (objectives[arm], -q_arms.index(arm)))

    formal_panels: dict[str, list[dict[str, Any]]] = {}
    outputs: dict[str, Any] = {}
    for role in FORMAL:
        role_spec = spec["roles"][role]
        task = load_npz(ROOT / role_spec["task_material"])
        boxes, scores, built, fixed = build_methods(
            materials[role], deltas[role], readout, source_maps, task)
        if "direct_material" in role_spec:
            direct = load_npz(ROOT / role_spec["direct_material"])
            for key in ("condition_offsets", "dataset_index", "sequence", "wait_ms"):
                if not exact(materials[role][key], direct[key]):
                    raise RuntimeError((name, role, key, "direct material identity drift"))
            q015_errors["fresh39d"] = max(q015_errors["fresh39d"], float(np.max(
                np.abs(built["q015"]["fresh39d"] - direct["kf_fresh39d"]), initial=0.0)))
        panels, predictions = base.evaluate(materials[role], boxes, scores)
        formal_panels[role] = panels
        domain_dir = output / name
        domain_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = domain_dir / f"{role}_predictions.npz"
        np.savez_compressed(prediction_path, **predictions)
        state_path = domain_dir / f"{role}_temporal_state.npz"
        state_archive: dict[str, np.ndarray] = {
            "condition_offsets": materials[role]["condition_offsets"],
            "gt_offsets": materials[role]["gt_offsets"],
            "dataset_index": materials[role]["dataset_index"],
            "sequence": materials[role]["sequence"],
            "wait_ms": materials[role]["wait_ms"],
            "selected_q_boxes": built[selected]["boxes"],
            "selected_q_fresh39d": built[selected]["fresh39d"],
            "q015_boxes": built["q015"]["boxes"],
            "q015_fresh39d": built["q015"]["fresh39d"],
            "fixed_cv39d": fixed,
        }
        for arm in ARMS:
            state_archive[f"{arm}__delta"] = deltas[role][arm]
        np.savez_compressed(state_path, **state_archive)
        outputs[role] = {"predictions": evidence(prediction_path), "temporal_state": evidence(state_path),
                         "task_material": evidence(ROOT / role_spec["task_material"]),
                         "population": evidence(ROOT / role_spec["population"]),
                         "frame_input": evidence(ROOT / role_spec["frame_input"])}

    all_formal = [row for role in FORMAL for row in formal_panels[role]]
    checks = {
        "population_frame_identity_exact": True,
        "current_checkpoint_and_readout_frozen": True,
        "q015_prior_mean_exact": q015_errors["prior_mean"] == 0.0,
        "q015_prior_variance_exact": q015_errors["prior_log_variance"] == 0.0,
        "q015_state_exact": q015_errors["state"] == 0.0,
        "q015_boxes_float32_exact": q015_errors["boxes"] == 0.0,
        "q015_features_float32_exact": q015_errors["features"] == 0.0,
        "q015_fresh39d_exact": q015_errors["fresh39d"] == 0.0,
        "fit24_only_constant": True,
        "calibration8_only_single_q_selection": True,
        "all_actual_waits_retained": waits == [0, 100, 200, 400],
        "all_formal_methods_retained": len(all_formal) == len(FORMAL) * len(waits) * (
            len(ARMS) * len(READOUTS) + 1),
        "raw_fixed_fresh_separate": True,
        "current_task_actual_input_control": True,
        "zero_detector_backbone_forward": True,
        "zero_optimizer_or_head_update": True,
        "zero_gpu": True,
    }
    if not all(checks.values()):
        raise RuntimeError({"domain": name, "checks": checks, "q015_errors": q015_errors})
    return {
        "identity": spec["identity"], "waits_ms": waits,
        "fit_constant_by_wait": {str(wait): constants[wait].tolist() for wait in waits},
        "fit_constant_anchor_counts": {str(wait): constant_counts[wait] for wait in waits},
        "q015_replay_max_error": q015_errors, "input_identity": identity_checks,
        "q_selection": {"metric": "mean fresh39d AP70 across calibration8 actual waits",
                        "objectives": objectives, "selected": selected,
                        "selected_q_variance_m2": Q_BY_ARM[selected]},
        "calibration_panels": calibration_panels, "formal_panels": formal_panels,
        "effects": base.formal_effects(formal_panels, selected), "outputs": outputs,
        "checks": checks,
    }


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic()
    cpu_started = time.process_time()
    output.mkdir(parents=True, exist_ok=False)
    domains = {name: domain_compute(name, spec, output)
               for name, spec in contract["domains"].items()}
    checks = {
        "all_domain_checks_passed": all(all(value["checks"].values()) for value in domains.values()),
        "three_current_checkpoint_domains": set(domains) == {"v2xvit_seed2024", "v2xvit_seed2025", "where2comm"},
        "selection_is_per_checkpoint_not_eval_role": True,
        "old_task_branch_not_substituted_for_current_temporal_method": True,
        "zero_detector_backbone_forward": True,
        "zero_optimizer_or_head_update": True,
        "zero_gpu": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {
        "schema": "crosscheckpoint_current_temporal_fixed_readout_controls_producer_v1",
        "status": "complete_producer_review_pending", "created_at": now(),
        "task_id": TASK, "owner": contract["owner"], "contract": evidence(ROOT / contract["contract_path"]),
        "domains": domains, "checks": checks,
        "prediction_before_result": contract["prediction_before_result"],
        "actual_cost": {"wall_seconds": time.monotonic() - started,
                        "cpu_process_seconds": time.process_time() - cpu_started,
                        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                        "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0,
                        "gpu_seconds": 0},
        "boundary": (
            "Current actual-input ViT2024, ViT2025 and Where exposed development roles. Each checkpoint uses "
            "its own frozen detector, candidate population and current 39D readout; calibration8 selects one Q "
            "without hold/val selection. Raw/fixed/fresh cells and current-task controls are retained. This is "
            "cached module replay, not equal-opportunity matched39D training, detector forward, complete-system "
            "cost, hidden confirmation, or a final cross-domain claim."
        ),
    }
    atomic(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA must be hidden")
    contract = load(args.contract)
    if contract.get("task_id") != TASK or contract.get("status") != "frozen_released_for_execution":
        raise RuntimeError("contract identity/status")
    if sha(Path(__file__)) != contract["runner"]["sha256"]:
        raise RuntimeError("runner identity drift")
    for item in contract["inputs"].values():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(("input identity drift", item["path"]))
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = compute(contract, args.output)
    print(json.dumps({"status": summary["status"],
                      "selected": {name: value["q_selection"]["selected"]
                                   for name, value in summary["domains"].items()},
                      "actual_cost": summary["actual_cost"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
