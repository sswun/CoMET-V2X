#!/usr/bin/env python3
"""Independent replay of the peer07 SPD GT-provenance AP supplement."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "preexperiment/research_autopilot"),
                str(ROOT / "preexperiment/research_baselines")]
import run_motion_direction_information_state_fitcal_v1_20261031 as core
from e1_receiver_value_core_v1_20261020 import common_mask, iou32
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap

TASK = "peer07_timing_2x2_source_split_ap_v2_20261031"
OWNER = "research_autopilot_peer07_timing_2x2_source_split_ap_v2"
ROLES = ("mechanism_holdout14", "val21")
WAITS = (0, 100, 200, 400)
METHODS = (
    "timing_off_direction_off",
    "timing_on_direction_off",
    "timing_off_direction_on",
    "timing_on_direction_on",
)
GROUPS = ("all", "veh_coop", "inf")
CODE = {"veh": 0, "coop": 1, "inf": 2}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def ev(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and (
        bool(np.array_equal(left, right, equal_nan=True)) if left.dtype.kind in "fc"
        else bool(np.array_equal(left, right)))


def atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def provenance(material: dict[str, np.ndarray], conditions: list[dict[str, Any]],
               role: str, labels: Path) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.full(len(material["gt_boxes"]), -1, np.int8)
    cache = {}; unmatched = 0; ambiguous = 0; maximum = 0.0; counts = Counter()
    for ordinal, index in enumerate(material["dataset_index"]):
        row = conditions[int(index)]
        identity = (str(row["role"]) == role
                    and str(row["sequence"]) == str(material["sequence"][ordinal])
                    and int(row["wait_ms"]) == int(material["wait_ms"][ordinal])
                    and int(row["receiver_timestamp_us"]) == int(material["receiver_timestamp_us"][ordinal]))
        if not identity:
            raise RuntimeError((role, ordinal, "condition identity"))
        frame = str(row["current_vehicle_frame"])
        if frame not in cache:
            objects = read(labels / f"{frame}.json")
            centers = np.asarray([[obj["3d_location"]["x"], obj["3d_location"]["y"]]
                                  for obj in objects], np.float64)
            sides = np.asarray([CODE[str(obj["from_side"])] for obj in objects], np.int8)
            cache[frame] = centers, sides, cKDTree(centers)
        centers, sides, tree = cache[frame]
        ga, gb = material["gt_offsets"][ordinal:ordinal + 2].astype(int)
        gt = material["gt_boxes"][ga:gb, :2].astype(np.float64)
        if not len(gt):
            continue
        distance, nearest = tree.query(gt, k=1); valid = distance < 1e-3
        maximum = max(maximum, float(distance.max(initial=0.0)))
        unmatched += int((~valid).sum())
        ambiguous += sum(len(tree.query_ball_point(point, 1e-3)) != 1 for point in gt[valid])
        result[ga:gb][valid] = sides[nearest[valid]]
        counts.update(result[ga:gb][valid].tolist())
    return result, {"unmatched": unmatched, "ambiguous": ambiguous,
                    "maximum_nearest_center_distance_m": maximum,
                    "counts": {("veh", "coop", "inf")[int(key)]: value
                               for key, value in sorted(counts.items())}}


def group_masks(side: np.ndarray, name: str):
    if name == "all": return np.ones(len(side), bool), np.zeros(len(side), bool)
    if name == "veh_coop": return side != CODE["inf"], side == CODE["inf"]
    if name == "inf": return side == CODE["inf"], side != CODE["inf"]
    raise ValueError(name)


def match(prediction: np.ndarray, target: np.ndarray, ignored_gt: np.ndarray, threshold: float):
    matrix = iou32(prediction, target); remaining = list(range(len(target)))
    tp = np.zeros(len(prediction), np.int8)
    for row in range(len(prediction)):
        if remaining:
            values = matrix[row, remaining]; best = int(np.argmax(values))
            if float(values[best]) >= threshold:
                tp[row] = 1; remaining.pop(best)
    ignored = np.zeros(len(prediction), np.int8)
    candidates = np.flatnonzero(tp == 0)
    if len(candidates) and len(ignored_gt):
        ignored[candidates] = (iou32(prediction[candidates], ignored_gt)
                               .max(axis=1, initial=0.0) >= threshold).astype(np.int8)
    fp = ((tp == 0) & (ignored == 0)).astype(np.int8)
    return tp, fp, ignored


def score_metric(tp: np.ndarray, fp: np.ndarray, scores: np.ndarray, gt: int):
    if not gt: return 0.0, int(tp.sum()), int(fp.sum())
    item = voc_ap(tp.tolist(), fp.tolist(), scores.tolist(), gt, True)
    return 100.0 * float(item["ap"]), int(item["tp"]), int(item["fp"])


def replay_role(material: dict[str, np.ndarray], state: dict[str, np.ndarray],
                predictions: dict[str, np.ndarray], side: np.ndarray, role: str):
    geometry = {name: material["cv_boxes"].copy() for name in METHODS}
    # Independently reconstruct projection, then apply stored condition/candidate correction.
    for ordinal in range(len(material["wait_ms"])):
        a, b = material["condition_offsets"][ordinal:ordinal + 2].astype(int)
        remote = a + np.flatnonzero(material["source"][a:b] == 1)
        projected, _ = core.heading_projection(material["stale_boxes"][remote], material["cv_boxes"][remote])
        for method, key in ((METHODS[0], "baseline_scalar"), (METHODS[2], "baseline_direction")):
            moved = projected.copy()
            moved[:, :2] += state[key][ordinal].astype(np.float32)
            geometry[method][remote] = moved
        for method, key in ((METHODS[1], "timing_scalar_correction"), (METHODS[3], "timing_direction_correction")):
            moved = projected.copy()
            moved[:, :2] += state[key][remote].astype(np.float32)
            geometry[method][remote] = moved
    totals = defaultdict(lambda: {"score": [], "tp50": [], "fp50": [], "ig50": [],
                                  "tp70": [], "fp70": [], "ig70": [], "gt": 0,
                                  "ignore_gt": 0, "outputs": 0})
    sequence = defaultdict(lambda: {"score": [], "tp50": [], "fp50": [], "ig50": [],
                                    "tp70": [], "fp70": [], "ig70": [], "gt": 0,
                                    "ignore_gt": 0, "outputs": 0})
    rebuilt = {}
    for method in METHODS:
        offsets = predictions[f"{method}__offset"].astype(int)
        candidate = predictions[f"{method}__candidate"].astype(int)
        scores = predictions[f"{method}__score"].astype(np.float64)
        for name in GROUPS:
            for threshold in (50, 70):
                for field in ("tp", "fp", "ignore"):
                    rebuilt[f"{method}__{name}__{field}{threshold}"] = np.zeros(len(candidate), np.int8)
        for ordinal, wait in enumerate(material["wait_ms"]):
            ga, gb = material["gt_offsets"][ordinal:ordinal + 2].astype(int)
            pa, pb = offsets[ordinal:ordinal + 2]
            ids = candidate[pa:pb]; predicted = geometry[method][ids]
            if not common_mask(predicted).all(): raise RuntimeError((role, method, ordinal, "ROI"))
            gt = material["gt_boxes"][ga:gb]; in_roi = common_mask(gt)
            for name in GROUPS:
                target_side, ignore_side = group_masks(side[ga:gb], name)
                target = gt[in_roi & target_side]; ignored_gt = gt[in_roi & ignore_side]
                rows = (totals[(int(wait), method, name)],
                        sequence[(str(material["sequence"][ordinal]), int(wait), method, name)])
                for row in rows:
                    row["score"].append(scores[pa:pb]); row["gt"] += len(target)
                    row["ignore_gt"] += len(ignored_gt); row["outputs"] += len(ids)
                for threshold in (50, 70):
                    tp, fp, ignored = match(predicted, target, ignored_gt, threshold / 100.0)
                    rebuilt[f"{method}__{name}__tp{threshold}"][pa:pb] = tp
                    rebuilt[f"{method}__{name}__fp{threshold}"][pa:pb] = fp
                    rebuilt[f"{method}__{name}__ignore{threshold}"][pa:pb] = ignored
                    for row in rows:
                        row[f"tp{threshold}"].append(tp); row[f"fp{threshold}"].append(fp)
                        row[f"ig{threshold}"].append(ignored)
    panels = []
    for (wait, method, name), row in sorted(totals.items()):
        item = {"role": role, "wait_ms": wait, "method": method, "source_group": name,
                "gt": row["gt"], "ignore_gt": row["ignore_gt"], "outputs_before_ignore": row["outputs"]}
        scores = np.concatenate(row["score"])
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"]); fp = np.concatenate(row[f"fp{threshold}"])
            ignored = np.concatenate(row[f"ig{threshold}"]).astype(bool)
            ap, ntp, nfp = score_metric(tp[~ignored], fp[~ignored], scores[~ignored], row["gt"])
            item[f"ap{threshold}"] = ap; item[f"tp{threshold}"] = ntp; item[f"fp{threshold}"] = nfp
            item[f"ignored_predictions{threshold}"] = int(ignored.sum())
        panels.append(item)
    sequence_rows = []
    for (seq, wait, method, name), row in sorted(sequence.items()):
        item = {"role": role, "sequence": seq, "wait_ms": wait, "method": method,
                "source_group": name, "gt": row["gt"], "ignore_gt": row["ignore_gt"],
                "outputs_before_ignore": row["outputs"]}
        scores = np.concatenate(row["score"])
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"]); fp = np.concatenate(row[f"fp{threshold}"])
            ignored = np.concatenate(row[f"ig{threshold}"]).astype(bool)
            ap, ntp, nfp = score_metric(tp[~ignored], fp[~ignored], scores[~ignored], row["gt"])
            item[f"ap{threshold}"] = ap; item[f"tp{threshold}"] = ntp; item[f"fp{threshold}"] = nfp
            item[f"ignored_predictions{threshold}"] = int(ignored.sum())
        sequence_rows.append(item)
    return panels, sequence_rows, rebuilt


def compare_rows(left: list[dict[str, Any]], right: list[dict[str, Any]], keys: tuple[str, ...]):
    def index(rows): return {tuple(row[key] for key in keys): row for row in rows}
    a, b = index(left), index(right)
    if set(a) != set(b): return False, float("inf")
    maximum = 0.0
    for key in a:
        if set(a[key]) != set(b[key]): return False, float("inf")
        for field in a[key]:
            x, y = a[key][field], b[key][field]
            if isinstance(x, float): maximum = max(maximum, abs(x - float(y)))
            elif x != y: return False, float("inf")
    return maximum <= 1e-10, maximum


def rebuild_bootstrap(rows):
    table = {(r["role"], r["sequence"], r["wait_ms"], r["source_group"], r["method"]): r for r in rows}
    generator = np.random.default_rng(20261031)
    result = []
    for role in ROLES:
        sequences = sorted({r["sequence"] for r in rows if r["role"] == role})
        for wait in WAITS:
            for group in ("veh_coop", "inf"):
                eligible = [seq for seq in sequences if table[(role, seq, wait, group, METHODS[0])]["gt"] > 0]
                for left, right in ((METHODS[1], METHODS[0]), (METHODS[3], METHODS[2])):
                    values = np.array([table[(role, seq, wait, group, left)]["ap70"] -
                                       table[(role, seq, wait, group, right)]["ap70"] for seq in eligible], dtype=np.float64)
                    indices = generator.integers(0, len(values), size=(10000, len(values)))
                    samples = np.mean(values[indices], axis=1)
                    result.append({
                        "role": role, "wait_ms": wait, "source_group": group, "left": left, "right": right,
                        "eligible_sequences": len(values), "sequence_mean_delta_ap70_pp": float(values.mean()),
                        "bootstrap_replicates": 10000, "bootstrap_seed": 20261031,
                        "ci95_low_pp": float(np.quantile(samples, .025)), "ci95_high_pp": float(np.quantile(samples, .975)),
                        "probability_positive": float(np.mean(samples > 0))})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); began = time.monotonic(); cpu_began = time.process_time()
    if args.output.exists(): raise FileExistsError(args.output)
    contract, producer = read(args.contract), read(args.producer)
    if contract.get("task_id") != TASK or contract.get("owner") != OWNER \
            or producer.get("status") != "formal_current_recipe_source_split_complete_review_pending":
        raise RuntimeError("identity/status")
    if ev(Path(__file__).resolve()) != contract["reviewer"] \
            or rel(args.producer) != contract["execution"]["producer_summary_path"]:
        raise RuntimeError("reviewer/producer identity")
    for spec in contract["inputs"].values():
        if ev(ROOT / spec["path"]) != spec: raise RuntimeError(("input drift", spec["path"]))
    conditions = {"mechanism_holdout14": read_lines(ROOT / contract["inputs"]["train_conditions"]["path"]),
                  "val21": read_lines(ROOT / contract["inputs"]["val_conditions"]["path"])}
    label_root = ROOT / contract["label_roots"]["main_domain"]["path"]
    rebuilt_panels = []; rebuilt_sequences = []; provenance_checks = {}; outcome_checks = {}
    for role in ROLES:
        material = npz(ROOT / contract["inputs"][f"{role}_material"]["path"])
        state = npz(ROOT / contract["inputs"][f"{role}_state"]["path"])
        prediction = npz(ROOT / contract["inputs"][f"{role}_predictions"]["path"])
        if len(prediction["method_names"]) != len(METHODS) or set(map(str, prediction["method_names"])) != set(METHODS):
            raise RuntimeError((role, "actual method identity"))
        if not all(equal(material[key], state[key]) for key in
                   ("dataset_index", "sequence", "wait_ms", "receiver_timestamp_us")):
            raise RuntimeError((role, "actual state/material identity"))
        side, audit = provenance(material, conditions[role], role, label_root)
        stored_side = npz(ROOT / producer["outputs"][role]["gt_provenance"]["path"])
        provenance_checks[role] = {"exact": equal(side, stored_side["gt_from_side"]), **audit}
        panels, sequences, outcomes = replay_role(material, state, prediction, side, role)
        for method in METHODS:
            for threshold in (50, 70):
                if not equal(outcomes[f"{method}__all__tp{threshold}"], prediction[f"{method}__tp{threshold}"]):
                    raise RuntimeError((role, method, threshold, "independent full-GT TP replay"))
        stored_outcomes = npz(ROOT / producer["outputs"][role]["source_split_outcomes"]["path"])
        outcome_checks[role] = {"keys_exact": set(outcomes) == set(stored_outcomes),
                                "arrays_exact": set(outcomes) == set(stored_outcomes)
                                and all(equal(outcomes[key], stored_outcomes[key]) for key in outcomes)}
        rebuilt_panels.extend(panels); rebuilt_sequences.extend(sequences)
    bootstrap_rows = rebuild_bootstrap(rebuilt_sequences)
    bootstrap_exact, bootstrap_error = compare_rows(
        bootstrap_rows, producer["paired_sequence_bootstrap"],
        ("role", "wait_ms", "source_group", "left", "right"))
    accepted = read(ROOT / contract["inputs"]["timing_formal_summary"]["path"])
    expected = {(row["role"], row["wait_ms"], row["method"]): row for row in accepted["panels"]}
    all_ap_replay = all(abs(row[f"ap{t}"] - expected[(row["role"], row["wait_ms"], row["method"])][f"ap{t}"]) < 1e-10
                        for row in rebuilt_panels if row["source_group"] == "all" for t in (50, 70))
    panels_exact, panel_error = compare_rows(
        rebuilt_panels, producer["formal_panels"], ("role", "wait_ms", "method", "source_group"))
    sequences_exact, sequence_error = compare_rows(
        rebuilt_sequences, producer["per_sequence_panels"],
        ("role", "sequence", "wait_ms", "method", "source_group"))
    checks = {
        "bootstrap_independently_recomputed": bootstrap_exact,
        "accepted_timing_full_gt_ap_independently_replayed": all_ap_replay,
        "all_provenance_arrays_rebuilt_exact": all(value["exact"] for value in provenance_checks.values()),
        "all_gt_unique_and_matched": all(not value["unmatched"] and not value["ambiguous"] for value in provenance_checks.values()),
        "all_source_split_outcome_arrays_rebuilt_exact": all(value["keys_exact"] and value["arrays_exact"] for value in outcome_checks.values()),
        "formal_panels_recomputed_exact": panels_exact,
        "per_sequence_panels_recomputed_exact": sequences_exact,
        "producer_checks_all_true": all(producer.get("checks", {}).values()),
        "official_full_gt_tp_replay_exact": all(all(value.values()) for value in producer["official_full_gt_replay_checks"].values()),
        "protocol_target_first_then_other_source_ignore": True,
        "zero_detector_or_backbone_forward": True, "zero_optimizer_or_head_update": True,
        "zero_gpu": True,
    }
    review = {
        "schema": "peer07_spd_gt_source_split_ap_independent_review_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": contract["frozen_at"], "task_id": TASK, "owner": OWNER,
        "contract": ev(args.contract), "producer": ev(args.producer),
        "checks": checks, "provenance_replay": provenance_checks,
        "outcome_replay": outcome_checks,
        "maximum_panel_abs_error": panel_error, "maximum_sequence_panel_abs_error": sequence_error,
        "actual_cost": {"wall_seconds": time.monotonic() - began,
                        "cpu_process_seconds": time.process_time() - cpu_began,
                        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                        "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0,
                        "gpu_seconds": 0},
    }
    atomic(args.output, review)
    if review["status"] != "passed": raise RuntimeError(checks)
    print(json.dumps({"status": review["status"], "checks": checks}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
