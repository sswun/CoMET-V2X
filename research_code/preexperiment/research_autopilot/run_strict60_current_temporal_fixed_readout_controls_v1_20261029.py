#!/usr/bin/env python3
"""Strict60 current-packet temporal controls under the frozen current readout.

The experiment reconstructs legal anchor measurements from the saved Q=.15^2
state ledger, builds the preregistered CV/constant/cumulative/Q arms, selects one
Q on calibration8, and retains every formal raw/fixed/fresh AP cell.  It does
not run a detector, update a readout, or use an official pair offset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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
BASE = ROOT / "preexperiment/research_baselines"
sys.path[:0] = [str(AUTO), str(BASE)]

import run_strict60_actual_packet_kf_value_v1_20261027 as actual
from e1_receiver_value_core_v1_20261020 import common_mask, greedy, iou32
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap


TASK = "strict60_current_temporal_fixed_readout_controls_v1_20261029"
ROLES = ("fit24", "calibration8", "mechanism_holdout14", "val21")
FORMAL = ROLES[2:]
ARMS = ("cv", "fit_constant", "cumulative_anchor_mean", "q0", "q005", "q015", "q030")
Q_BY_ARM = {"q0": 0.0, "q005": .05 ** 2, "q015": .15 ** 2, "q030": .30 ** 2}
READOUTS = ("raw", "fixed39d", "fresh39d")


def now(): return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str: return str(Path(path).resolve().relative_to(ROOT))


def ev(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path): return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def exact(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def atomic(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def ledger_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def recover_measurements(rows: list[dict[str, Any]]):
    """Invert the saved scalar KF exactly to recover each legal anchor mean."""
    states: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    measurements: list[np.ndarray | None] = []
    maximum = 0.0
    for row in rows:
        key = (str(row["sequence"]), int(row["wait_ms"]))
        old, variance = states.get(key, (np.zeros(2, np.float64), 1.0))
        prior = variance + .15 ** 2
        target = np.asarray([row["kf_dx"], row["kf_dy"]], np.float64)
        anchors = int(row["anchors_used"])
        if anchors:
            gain = prior / (prior + .6 ** 2 / anchors)
            measurement = old + (target - old) / gain
            updated = old + gain * (measurement - old)
            variance = (1.0 - gain) * prior
        else:
            measurement = None
            updated = old
            variance = prior
        maximum = max(maximum, float(np.max(np.abs(updated - target))))
        states[key] = (updated.copy(), variance)
        measurements.append(None if measurement is None else measurement.copy())
    return measurements, maximum


def fit_constants(rows: list[dict[str, Any]], measurements: list[np.ndarray | None], waits: list[int]):
    total = {wait: np.zeros(2, np.float64) for wait in waits}
    count = {wait: 0 for wait in waits}
    for row, measurement in zip(rows, measurements):
        if measurement is None:
            continue
        wait = int(row["wait_ms"]); anchors = int(row["anchors_used"])
        total[wait] += measurement * anchors; count[wait] += anchors
    constant = {wait: total[wait] / count[wait] if count[wait] else np.zeros(2) for wait in waits}
    return constant, count


def shifts(rows: list[dict[str, Any]], measurements: list[np.ndarray | None],
           constant: dict[int, np.ndarray], mode: str) -> np.ndarray:
    states: dict[tuple[str, int], Any] = {}
    result = []
    for row, measurement in zip(rows, measurements):
        key = (str(row["sequence"]), int(row["wait_ms"]))
        anchors = int(row["anchors_used"])
        if mode == "cv":
            value = np.zeros(2, np.float64)
        elif mode == "fit_constant":
            value = constant[key[1]]
        elif mode == "cumulative_anchor_mean":
            total, count = states.get(key, (np.zeros(2, np.float64), 0))
            if measurement is not None:
                total = total + measurement * anchors; count += anchors
            value = total / count if count else np.zeros(2, np.float64)
            states[key] = (total, count)
        else:
            process = float(Q_BY_ARM[mode])
            mean, variance = states.get(key, (np.zeros(2, np.float64), 1.0))
            prior = variance + process
            if measurement is not None:
                gain = prior / (prior + .6 ** 2 / anchors)
                mean = mean + gain * (measurement - mean)
                variance = (1.0 - gain) * prior
            else:
                variance = prior
            value = mean
            states[key] = (mean.copy(), variance)
        result.append(np.asarray(value, np.float64))
    return np.asarray(result, np.float64)


def compose(material: dict[str, np.ndarray], delta: np.ndarray):
    boxes = material["cv_boxes"].copy()
    features = material["cv_features"].copy()
    changed_remote = 0
    for condition, move in enumerate(delta):
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        source = material["source"][a:b]; protected = material["protected"][a:b]
        remote = a + np.flatnonzero(source == 1)
        fixed = a + np.flatnonzero((source == 0) & (protected == 1))
        if not np.any(move):
            continue
        boxes[remote, :2] += move.astype(np.float32)
        features[remote] = actual.optional.update_features(
            material["cv_features"][remote], boxes[remote], boxes[fixed], material["raw_scores"][fixed])
        changed_remote += len(remote)
    return boxes, features, changed_remote


def method_names(include_task: bool) -> tuple[str, ...]:
    names = tuple(f"{arm}_{readout}" for arm in ARMS for readout in READOUTS)
    return names + (("current_task_fresh39d",) if include_task else ())


def build_role(material: dict[str, np.ndarray], deltas: dict[str, np.ndarray],
               readout: Path, source_maps: dict[str, Any], task_material: dict[str, np.ndarray] | None):
    boxes_by_method: dict[str, np.ndarray] = {}
    scores_by_method: dict[str, np.ndarray] = {}
    built: dict[str, dict[str, Any]] = {}
    mapping = source_maps["maps"]["direct_to_utility"]
    frozen = actual.optional.direct_scores({"base": material["cv_features"]}, readout)["base"]
    frozen = actual.optional.source_map(frozen, material["source"], mapping)
    raw = material["raw_scores"].astype(np.float64)
    for arm in ARMS:
        boxes, features, changed = compose(material, deltas[arm])
        fresh = actual.optional.direct_scores({"fresh": features}, readout)["fresh"]
        fresh = actual.optional.source_map(fresh, material["source"], mapping)
        boxes_by_method[f"{arm}_raw"] = boxes; scores_by_method[f"{arm}_raw"] = raw
        boxes_by_method[f"{arm}_fixed39d"] = boxes; scores_by_method[f"{arm}_fixed39d"] = frozen
        boxes_by_method[f"{arm}_fresh39d"] = boxes; scores_by_method[f"{arm}_fresh39d"] = fresh
        built[arm] = {"boxes": boxes, "features": features, "fresh39d": fresh,
                      "changed_remote_candidates": int(changed)}
    if task_material is not None:
        for key in ("condition_offsets", "gt_offsets", "dataset_index", "sequence", "wait_ms",
                    "receiver_timestamp_us", "source", "protected", "raw_scores"):
            if not exact(material[key], task_material[key]):
                raise RuntimeError(("current task material identity drift", key))
        boxes_by_method["current_task_fresh39d"] = task_material["task_boxes"]
        scores_by_method["current_task_fresh39d"] = task_material["task_direct_u"]
    return boxes_by_method, scores_by_method, built, frozen


def evaluate(material: dict[str, np.ndarray], boxes_by_method: dict[str, np.ndarray],
             scores_by_method: dict[str, np.ndarray]):
    names = tuple(boxes_by_method)
    stats = defaultdict(lambda: {"score": [], "tp50": [], "tp70": [], "gt": 0,
                                 "outputs": 0, "remote": 0, "dropped_prediction_roi": 0,
                                 "dropped_gt_roi": 0})
    saved = {name: {"candidate": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
             for name in names}
    for condition, wait_value in enumerate(material["wait_ms"]):
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        ga, gb = material["gt_offsets"][condition:condition + 2].astype(int)
        raw = material["raw_scores"][a:b]; source = material["source"][a:b]
        protected = material["protected"][a:b]; gt_all = material["gt_boxes"][ga:gb]
        gt_keep = common_mask(gt_all); gt = gt_all[gt_keep]
        for name in names:
            ids, rank = actual.optional.ordered(raw, source, protected, scores_by_method[name][a:b], None)
            boxes = boxes_by_method[name][a:b]
            keep = common_mask(boxes[ids]); dropped = int((~keep).sum())
            ids, rank = ids[keep], rank[keep]
            matrix = iou32(boxes[ids], gt)
            tp50 = (greedy(matrix, np.arange(len(ids)), .5) >= 0).astype(np.int8)
            tp70 = (greedy(matrix, np.arange(len(ids)), .7) >= 0).astype(np.int8)
            row = stats[(int(wait_value), name)]
            row["score"].append(rank); row["tp50"].append(tp50); row["tp70"].append(tp70)
            row["gt"] += len(gt); row["outputs"] += len(ids)
            row["remote"] += int((source[ids] == 1).sum())
            row["dropped_prediction_roi"] += dropped; row["dropped_gt_roi"] += int((~gt_keep).sum())
            record = saved[name]; record["candidate"].append((a + ids).astype(np.int64))
            record["score"].append(rank.astype(np.float64)); record["tp50"].append(tp50)
            record["tp70"].append(tp70); record["offset"].append(record["offset"][-1] + len(ids))
    panels = []
    for (wait, name), row in sorted(stats.items()):
        score = np.concatenate(row["score"])
        panel = {"wait_ms": wait, "method": name,
                 **{field: int(row[field]) for field in ("gt", "outputs", "remote",
                                                         "dropped_prediction_roi", "dropped_gt_roi")}}
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"])
            metric = voc_ap(tp.tolist(), (1 - tp).tolist(), score.tolist(), row["gt"], True)
            panel[f"ap{threshold}"] = 100.0 * float(metric["ap"])
            panel[f"tp{threshold}"] = int(metric["tp"]); panel[f"fp{threshold}"] = int(metric["fp"])
        panels.append(panel)
    archive: dict[str, np.ndarray] = {
        "method_names": np.asarray(names, dtype="U64"),
        "condition_wait_ms": material["wait_ms"].astype(np.int16),
    }
    for name, record in saved.items():
        for field in ("candidate", "score", "tp50", "tp70"):
            archive[f"{name}__{field}"] = np.concatenate(record[field])
        archive[f"{name}__offset"] = np.asarray(record["offset"], np.int64)
    return panels, archive


def formal_effects(panels_by_role: dict[str, list[dict[str, Any]]], selected: str):
    result = []
    for role, panels in panels_by_role.items():
        table = {(int(row["wait_ms"]), row["method"]): row for row in panels}
        for wait in sorted({int(row["wait_ms"]) for row in panels}):
            row: dict[str, Any] = {"role": role, "wait_ms": wait, "selected_q_arm": selected}
            for readout in READOUTS:
                selected_panel = table[(wait, f"{selected}_{readout}")]
                q015 = table[(wait, f"q015_{readout}")]
                row[f"selected_q_over_q015_{readout}_delta_ap70_pp"] = selected_panel["ap70"] - q015["ap70"]
                for arm in ("fit_constant", "cumulative_anchor_mean"):
                    row[f"{arm}_over_q015_{readout}_delta_ap70_pp"] = table[(wait, f"{arm}_{readout}")]["ap70"] - q015["ap70"]
            task = table[(wait, "current_task_fresh39d")]
            row["current_task_over_q015_fresh39d_delta_ap70_pp"] = task["ap70"] - table[(wait, "q015_fresh39d")]["ap70"]
            row["current_task_over_selected_q_fresh39d_delta_ap70_pp"] = task["ap70"] - table[(wait, f"{selected}_fresh39d")]["ap70"]
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic(); cpu_started = time.process_time()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA must be hidden")
    contract = load(args.contract)
    if contract.get("task_id") != TASK or contract.get("status") != "frozen_released_for_execution":
        raise RuntimeError("contract identity/status")
    if sha(Path(__file__)) != contract["runner"]["sha256"]:
        raise RuntimeError("runner drift")
    for item in contract["inputs"].values():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(("input drift", item["path"]))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    readout = ROOT / contract["readout"]["root"]
    source_maps = load(readout / "source_maps.json")
    materials: dict[str, dict[str, np.ndarray]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    measurements: dict[str, list[np.ndarray | None]] = {}
    state_errors: dict[str, float] = {}
    waits_by_role: dict[str, list[int]] = {}
    for role in ROLES:
        materials[role] = load_npz(ROOT / contract["materials"][role]["material"])
        rows[role] = ledger_rows(ROOT / contract["materials"][role]["ledger"])
        material = materials[role]
        if len(rows[role]) != len(material["wait_ms"]):
            raise RuntimeError((role, "ledger length"))
        for ordinal, (row, index, sequence, wait, timestamp, anchors) in enumerate(zip(
                rows[role], material["dataset_index"], material["sequence"], material["wait_ms"],
                material["receiver_timestamp_us"], material["anchor_count"])):
            identity = (int(row["condition_ordinal"]) == ordinal and int(row["dataset_index"]) == int(index)
                        and str(row["sequence"]) == str(sequence) and int(row["wait_ms"]) == int(wait)
                        and int(row["receiver_timestamp_us"]) == int(timestamp)
                        and int(row["anchors_used"]) == int(anchors))
            if not identity:
                raise RuntimeError((role, ordinal, "ledger/material identity drift"))
        measurements[role], state_errors[role] = recover_measurements(rows[role])
        waits_by_role[role] = sorted(map(int, np.unique(material["wait_ms"])))
    actual_waits = waits_by_role["fit24"]
    if any(waits_by_role[role] != actual_waits for role in ROLES):
        raise RuntimeError(("wait identity", waits_by_role))
    constants, constant_counts = fit_constants(rows["fit24"], measurements["fit24"], actual_waits)
    deltas: dict[str, dict[str, np.ndarray]] = {}
    q015_max = {"state": 0.0, "boxes": 0.0, "features": 0.0}
    delta_inventory: dict[str, Any] = {}
    built_cache: dict[str, Any] = {}
    for role in ROLES:
        deltas[role] = {arm: shifts(rows[role], measurements[role], constants, arm) for arm in ARMS}
        target = np.asarray([[row["kf_dx"], row["kf_dy"]] for row in rows[role]], np.float64)
        q015_max["state"] = max(q015_max["state"], float(np.max(np.abs(deltas[role]["q015"] - target))))
        _, _, built, frozen = build_role(materials[role], deltas[role], readout, source_maps, None)
        q015_max["boxes"] = max(q015_max["boxes"], float(np.max(
            np.abs(built["q015"]["boxes"] - materials[role]["kf_boxes"]), initial=0.0)))
        q015_max["features"] = max(q015_max["features"], float(np.max(
            np.abs(built["q015"]["features"] - materials[role]["kf_features"]), initial=0.0)))
        delta_inventory[role] = {arm: {"mean_norm_m": float(np.linalg.norm(value, axis=1).mean()),
                                               "max_norm_m": float(np.linalg.norm(value, axis=1).max(initial=0.0))}
                                     for arm, value in deltas[role].items()}
        built_cache[role] = (built, frozen)
    cal_boxes, cal_scores, _, _ = build_role(materials["calibration8"], deltas["calibration8"],
                                              readout, source_maps, None)
    calibration_panels, _ = evaluate(materials["calibration8"], cal_boxes, cal_scores)
    q_arms = tuple(Q_BY_ARM)
    cal_table = {(int(row["wait_ms"]), row["method"]): row for row in calibration_panels}
    q_objective = {arm: float(np.mean([cal_table[(wait, f"{arm}_fresh39d")]["ap70"]
                                             for wait in actual_waits])) for arm in q_arms}
    selected = max(q_arms, key=lambda arm: (q_objective[arm], -q_arms.index(arm)))
    formal_panels: dict[str, list[dict[str, Any]]] = {}; outputs = {}
    for role in FORMAL:
        task_material = load_npz(ROOT / contract["current_task_materials"][role]["path"])
        boxes, scores, built, frozen = build_role(materials[role], deltas[role], readout, source_maps, task_material)
        panels, predictions = evaluate(materials[role], boxes, scores)
        formal_panels[role] = panels
        prediction_path = args.output / f"{role}_predictions.npz"
        np.savez_compressed(prediction_path, **predictions)
        material_path = args.output / f"{role}_temporal_material.npz"
        archive = {"condition_offsets": materials[role]["condition_offsets"],
                   "gt_offsets": materials[role]["gt_offsets"], "dataset_index": materials[role]["dataset_index"],
                   "sequence": materials[role]["sequence"], "wait_ms": materials[role]["wait_ms"],
                   "frozen_cv39d": frozen}
        for arm in ARMS:
            archive[f"{arm}__delta"] = deltas[role][arm]
            archive[f"{arm}__boxes"] = built[arm]["boxes"]
            archive[f"{arm}__features"] = built[arm]["features"]
            archive[f"{arm}__fresh39d"] = built[arm]["fresh39d"]
        np.savez_compressed(material_path, **archive)
        outputs[role] = {"predictions": ev(prediction_path), "material": ev(material_path),
                         "source_material": ev(ROOT / contract["materials"][role]["material"]),
                         "ledger": ev(ROOT / contract["materials"][role]["ledger"]),
                         "current_task_material": ev(ROOT / contract["current_task_materials"][role]["path"])}
    all_formal = [row for role in FORMAL for row in formal_panels[role]]
    expected_per_wait = len(ARMS) * len(READOUTS) + 1
    checks = {
        "ledger_material_identity_exact": True,
        "legal_measurement_recovery_exact": max(state_errors.values()) < 1e-12,
        "q015_state_exact": q015_max["state"] < 1e-12,
        "q015_boxes_float32_exact": q015_max["boxes"] == 0.0,
        "q015_features_float32_exact": q015_max["features"] == 0.0,
        "fit24_only_constant": True,
        "calibration8_only_single_q_selection": True,
        "actual_waits_not_assumed": actual_waits == [0, 400],
        "all_formal_methods_retained": len(all_formal) == len(FORMAL) * len(actual_waits) * expected_per_wait,
        "raw_fixed_fresh_separate": True,
        "current_task_formal_only": True,
        "unmatched_and_fp_retained": True,
        "zero_detector_backbone_forward": True,
        "zero_optimizer_or_head_update": True,
        "zero_gpu": True,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "q015": q015_max, "state": state_errors})
    summary = {
        "schema": "strict60_current_temporal_fixed_readout_controls_producer_v1",
        "status": "complete_producer_review_pending", "created_at": now(),
        "task_id": TASK, "owner": contract["owner"], "contract": ev(args.contract),
        "method_identity": {
            "detector_checkpoint": contract["identity"]["detector_checkpoint"],
            "candidate_population": "current strict60 actual-packet saved full population",
            "geometry_base": "cv_boxes/cv_features from current material",
            "readout": contract["readout"], "arms": list(ARMS), "readouts": list(READOUTS),
            "actual_waits_ms": actual_waits,
        },
        "fit_constant_by_wait": {str(wait): constants[wait].tolist() for wait in actual_waits},
        "fit_constant_anchor_counts": {str(wait): int(constant_counts[wait]) for wait in actual_waits},
        "state_replay_max_error": state_errors, "q015_replay_max_error": q015_max,
        "delta_inventory": delta_inventory,
        "calibration_panels": calibration_panels,
        "q_selection": {"metric": "mean fresh39d AP70 across calibration8 actual waits",
                        "objectives": q_objective, "selected": selected,
                        "selected_q_variance_m2": Q_BY_ARM[selected]},
        "formal_panels": formal_panels,
        "effects": formal_effects(formal_panels, selected),
        "outputs": outputs, "checks": checks,
        "prediction_before_result": contract["prediction_before_result"],
        "actual_cost": {"wall_seconds": time.monotonic() - started,
                        "cpu_process_seconds": time.process_time() - cpu_started,
                        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                        "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0,
                        "gpu_seconds": 0},
        "boundary": (
            "Exact cached current strict60 actual-packet population. Legal anchor measurements are algebraically "
            "recovered from the saved Q=.15^2 state; fit24 defines constants and calibration8 selects one Q. "
            "Formal roles are exposed and retain all signed cells. Frozen current readout isolates geometry; "
            "fresh39D recomputes only the legal 39D box features. This is module replay, not matched readout "
            "training, detector/backbone forward, complete-system cost, hidden confirmation, or final core claim."
        ),
    }
    atomic(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "selected": selected,
                      "formal_panels": len(all_formal), "cost": summary["actual_cost"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
