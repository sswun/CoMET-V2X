#!/usr/bin/env python3
"""Formal source AP for the frozen timing x direction system 2x2.

The official all-GT evaluation remains unchanged.  Two supplementary panels use
``from_side`` from the public cooperative labels: veh+coop versus inf.  GT from
the other panel is an ignore set: after target-GT matching, a still-unmatched
prediction overlapping ignore GT at the same IoU threshold is removed from both
TP and FP.  No prediction, GT, score, method, role, wait, or sequence is selected
from exposed outcomes.
"""
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
AUTO = ROOT / "preexperiment/research_autopilot"
BASE = ROOT / "preexperiment/research_baselines"
sys.path[:0] = [str(AUTO), str(BASE)]

import run_motion_direction_information_state_fitcal_v1_20261031 as core
from e1_receiver_value_core_v1_20261020 import common_mask, greedy, iou32
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
SIDE_CODE = {"veh": 0, "coop": 1, "inf": 2}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def evidence(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def exact(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(np.array_equal(left, right, equal_nan=True)) if left.dtype.kind in "fc" \
        else bool(np.array_equal(left, right))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def label_aggregate(label_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256(); count = 0; total = 0
    for path in sorted(label_root.glob("*.json"), key=lambda item: item.name):
        data = path.read_bytes(); count += 1; total += len(data)
        digest.update(path.name.encode("ascii")); digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii")); digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    return {"files": count, "bytes": total, "aggregate_sha256": digest.hexdigest()}


def map_provenance(material: dict[str, np.ndarray], all_conditions: list[dict[str, Any]],
                   role: str, label_root: Path):
    codes = np.full(len(material["gt_boxes"]), -1, np.int8)
    cache: dict[str, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    maximum = 0.0; unmatched = 0; ambiguous = 0; frame_files: set[str] = set()
    counts = Counter(); per_wait: dict[int, Counter] = defaultdict(Counter)
    identity = []
    for ordinal, dataset_index in enumerate(material["dataset_index"]):
        row = all_conditions[int(dataset_index)]
        if not (str(row["role"]) == role
                and str(row["sequence"]) == str(material["sequence"][ordinal])
                and int(row["wait_ms"]) == int(material["wait_ms"][ordinal])
                and int(row["receiver_timestamp_us"]) == int(material["receiver_timestamp_us"][ordinal])):
            raise RuntimeError((role, ordinal, "condition identity drift"))
        vehicle_frame = str(row["current_vehicle_frame"])
        frame_files.add(vehicle_frame + ".json")
        if vehicle_frame not in cache:
            path = label_root / f"{vehicle_frame}.json"
            objects = load(path)
            centers = np.asarray([[item["3d_location"]["x"], item["3d_location"]["y"]]
                                  for item in objects], np.float64)
            sides = np.asarray([SIDE_CODE[str(item["from_side"])] for item in objects], np.int8)
            cache[vehicle_frame] = centers, sides, cKDTree(centers)
        centers, sides, tree = cache[vehicle_frame]
        ga, gb = material["gt_offsets"][ordinal:ordinal + 2].astype(int)
        gt = material["gt_boxes"][ga:gb, :2].astype(np.float64)
        if not len(gt):
            continue
        distance, index = tree.query(gt, k=1)
        maximum = max(maximum, float(distance.max(initial=0.0)))
        valid = distance < 1e-3
        unmatched += int((~valid).sum())
        for point in gt[valid]:
            ambiguous += int(len(tree.query_ball_point(point, 1e-3)) != 1)
        codes[ga:gb][valid] = sides[index[valid]]
        for code in codes[ga:gb][valid].tolist():
            name = ("veh", "coop", "inf")[int(code)]
            counts[name] += 1; per_wait[int(material["wait_ms"][ordinal])][name] += 1
        identity.append((str(material["sequence"][ordinal]), int(material["wait_ms"][ordinal]),
                         int(material["receiver_timestamp_us"][ordinal]), vehicle_frame,
                         int(ga), int(gb)))
    return codes, {
        "conditions": len(material["wait_ms"]), "gt_occurrences": len(codes),
        "referenced_label_files": len(frame_files), "from_side_repeated_counts": dict(counts),
        "from_side_by_wait": {str(wait): dict(per_wait[wait]) for wait in WAITS},
        "unmatched": unmatched, "ambiguous": ambiguous,
        "maximum_nearest_center_distance_m": maximum,
        "mapping_threshold_m_strict_less_than": 1e-3,
        "condition_identity_sha256": hashlib.sha256(
            json.dumps(identity, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def masks(codes: np.ndarray, group: str) -> tuple[np.ndarray, np.ndarray]:
    if group == "all":
        return np.ones(len(codes), bool), np.zeros(len(codes), bool)
    if group == "veh_coop":
        return codes != SIDE_CODE["inf"], codes == SIDE_CODE["inf"]
    if group == "inf":
        return codes == SIDE_CODE["inf"], codes != SIDE_CODE["inf"]
    raise ValueError(group)


def outcome(boxes: np.ndarray, target: np.ndarray, ignore: np.ndarray, threshold: float):
    target_match = greedy(iou32(boxes, target), np.arange(len(boxes)), threshold)
    tp = target_match >= 0
    ignored = np.zeros(len(boxes), bool)
    remaining = ~tp
    if len(ignore) and remaining.any():
        overlap = iou32(boxes[remaining], ignore)
        ignored[remaining] = overlap.max(axis=1, initial=0.0) >= threshold
    fp = ~(tp | ignored)
    return tp.astype(np.int8), fp.astype(np.int8), ignored.astype(np.int8)


def metric(tp: np.ndarray, fp: np.ndarray, score: np.ndarray, gt: int) -> dict[str, Any]:
    if gt <= 0:
        return {"ap": 0.0, "tp": int(tp.sum()), "fp": int(fp.sum())}
    result = voc_ap(tp.tolist(), fp.tolist(), score.tolist(), int(gt), True)
    return {"ap": 100.0 * float(result["ap"]), "tp": int(result["tp"]),
            "fp": int(result["fp"])}


def evaluate_role(material: dict[str, np.ndarray], predictions: dict[str, np.ndarray],
                  boxes: dict[str, np.ndarray], provenance: np.ndarray, role: str):
    stats = defaultdict(lambda: {"score": [], "tp50": [], "fp50": [], "ignore50": [],
                                 "tp70": [], "fp70": [], "ignore70": [], "gt": 0,
                                 "ignore_gt": 0, "outputs": 0})
    seq_stats = defaultdict(lambda: {"score": [], "tp50": [], "fp50": [], "ignore50": [],
                                     "tp70": [], "fp70": [], "ignore70": [], "gt": 0,
                                     "ignore_gt": 0, "outputs": 0})
    saved: dict[str, np.ndarray] = {}
    full_exact = {}
    for method in METHODS:
        offsets = predictions[f"{method}__offset"].astype(np.int64)
        candidate = predictions[f"{method}__candidate"].astype(np.int64)
        score = predictions[f"{method}__score"].astype(np.float64)
        for group in GROUPS:
            for threshold in (50, 70):
                saved[f"{method}__{group}__tp{threshold}"] = np.zeros(len(candidate), np.int8)
                saved[f"{method}__{group}__fp{threshold}"] = np.zeros(len(candidate), np.int8)
                saved[f"{method}__{group}__ignore{threshold}"] = np.zeros(len(candidate), np.int8)
        for ordinal, wait in enumerate(material["wait_ms"]):
            a, b = material["condition_offsets"][ordinal:ordinal + 2].astype(int)
            ga, gb = material["gt_offsets"][ordinal:ordinal + 2].astype(int)
            pa, pb = offsets[ordinal:ordinal + 2].astype(int)
            ids = candidate[pa:pb]
            if len(ids) and (ids.min() < a or ids.max() >= b or len(np.unique(ids)) != len(ids)):
                raise RuntimeError((role, method, ordinal, "candidate identity/range"))
            pred_boxes = boxes[method][ids]
            gt_all = material["gt_boxes"][ga:gb]
            gt_roi = common_mask(gt_all)
            pred_roi = common_mask(pred_boxes)
            if not pred_roi.all():
                raise RuntimeError((role, method, ordinal, "saved prediction contains out-of-ROI box"))
            code = provenance[ga:gb]
            for group in GROUPS:
                target_mask, ignore_mask = masks(code, group)
                target = gt_all[gt_roi & target_mask]
                ignored_gt = gt_all[gt_roi & ignore_mask]
                aggregate = stats[(int(wait), method, group)]
                sequence = str(material["sequence"][ordinal])
                sequence_row = seq_stats[(sequence, int(wait), method, group)]
                for row in (aggregate, sequence_row):
                    row["score"].append(score[pa:pb]); row["gt"] += len(target)
                    row["ignore_gt"] += len(ignored_gt); row["outputs"] += len(ids)
                for threshold in (50, 70):
                    tp, fp, ignored = outcome(pred_boxes, target, ignored_gt, threshold / 100.0)
                    for row in (aggregate, sequence_row):
                        row[f"tp{threshold}"].append(tp); row[f"fp{threshold}"].append(fp)
                        row[f"ignore{threshold}"].append(ignored)
                    saved[f"{method}__{group}__tp{threshold}"][pa:pb] = tp
                    saved[f"{method}__{group}__fp{threshold}"][pa:pb] = fp
                    saved[f"{method}__{group}__ignore{threshold}"][pa:pb] = ignored
        for threshold in (50, 70):
            full_exact[f"{method}_tp{threshold}"] = exact(
                saved[f"{method}__all__tp{threshold}"], predictions[f"{method}__tp{threshold}"])
            full_exact[f"{method}_fp{threshold}"] = bool(np.array_equal(
                saved[f"{method}__all__fp{threshold}"],
                (1 - predictions[f"{method}__tp{threshold}"]).astype(np.int8)))
    panels = []
    for (wait, method, group), row in sorted(stats.items()):
        panel = {"role": role, "wait_ms": wait, "method": method, "source_group": group,
                 "gt": int(row["gt"]), "ignore_gt": int(row["ignore_gt"]),
                 "outputs_before_ignore": int(row["outputs"])}
        score = np.concatenate(row["score"])
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"]); fp = np.concatenate(row[f"fp{threshold}"])
            ignored = np.concatenate(row[f"ignore{threshold}"]).astype(bool)
            result = metric(tp[~ignored], fp[~ignored], score[~ignored], row["gt"])
            panel[f"ap{threshold}"] = result["ap"]; panel[f"tp{threshold}"] = result["tp"]
            panel[f"fp{threshold}"] = result["fp"]; panel[f"ignored_predictions{threshold}"] = int(ignored.sum())
        panels.append(panel)
    sequence_rows = []
    for (sequence, wait, method, group), row in sorted(seq_stats.items()):
        item = {"role": role, "sequence": sequence, "wait_ms": wait, "method": method,
                "source_group": group, "gt": int(row["gt"]), "ignore_gt": int(row["ignore_gt"]),
                "outputs_before_ignore": int(row["outputs"])}
        score = np.concatenate(row["score"])
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"]); fp = np.concatenate(row[f"fp{threshold}"])
            ignored = np.concatenate(row[f"ignore{threshold}"]).astype(bool)
            result = metric(tp[~ignored], fp[~ignored], score[~ignored], row["gt"])
            item[f"ap{threshold}"] = result["ap"]; item[f"tp{threshold}"] = result["tp"]
            item[f"fp{threshold}"] = result["fp"]; item[f"ignored_predictions{threshold}"] = int(ignored.sum())
        sequence_rows.append(item)
    return panels, sequence_rows, saved, full_exact


def bootstrap(sequence_rows: list[dict[str, Any]], seed: int = 20261031,
              replicates: int = 10000) -> list[dict[str, Any]]:
    table = {(row["role"], row["sequence"], int(row["wait_ms"]), row["source_group"], row["method"]): row
             for row in sequence_rows}
    rng = np.random.default_rng(seed); output = []
    contrasts = ((METHODS[1], METHODS[0]), (METHODS[3], METHODS[2]))
    for role in ROLES:
        sequences = sorted({row["sequence"] for row in sequence_rows if row["role"] == role})
        for wait in WAITS:
            for group in ("veh_coop", "inf"):
                eligible = [sequence for sequence in sequences
                            if table[(role, sequence, wait, group, METHODS[0])]["gt"] > 0]
                for left, right in contrasts:
                    values = np.asarray([
                        table[(role, sequence, wait, group, left)]["ap70"]
                        - table[(role, sequence, wait, group, right)]["ap70"]
                        for sequence in eligible], np.float64)
                    draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
                    output.append({
                        "role": role, "wait_ms": wait, "source_group": group,
                        "left": left, "right": right, "eligible_sequences": len(values),
                        "sequence_mean_delta_ap70_pp": float(values.mean()),
                        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
                        "ci95_low_pp": float(np.quantile(draws, .025)),
                        "ci95_high_pp": float(np.quantile(draws, .975)),
                        "probability_positive": float(np.mean(draws > 0)),
                    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); began = time.monotonic(); cpu_began = time.process_time()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = load(args.contract)
    if contract.get("task_id") != TASK or contract.get("owner") != OWNER \
            or contract.get("status") != "frozen_released_for_formal_execution":
        raise RuntimeError("contract identity/status")
    if evidence(Path(__file__).resolve()) != contract["producer"]:
        raise RuntimeError("producer drift")
    for spec in contract["inputs"].values():
        path = ROOT / spec["path"]
        if evidence(path) != spec:
            raise RuntimeError(("input drift", spec["path"]))
    label_root = ROOT / contract["label_roots"]["main_domain"]["path"]
    author_root = ROOT / contract["label_roots"]["author21"]["path"]
    main_aggregate = label_aggregate(label_root); author_aggregate = label_aggregate(author_root)
    if main_aggregate != contract["label_roots"]["main_domain"]["identity"] \
            or author_aggregate != contract["label_roots"]["author21"]["identity"] \
            or main_aggregate != author_aggregate:
        raise RuntimeError("label aggregate identity drift")
    args.output.mkdir(parents=True)
    conditions = {
        "mechanism_holdout14": load_jsonl(ROOT / contract["inputs"]["train_conditions"]["path"]),
        "val21": load_jsonl(ROOT / contract["inputs"]["val_conditions"]["path"]),
    }
    panels_all = []; sequence_all = []; mapping_audits = {}; outputs = {}; exact_checks = {}
    for role in ROLES:
        material = load_npz(ROOT / contract["inputs"][f"{role}_material"]["path"])
        state = load_npz(ROOT / contract["inputs"][f"{role}_state"]["path"])
        predictions = load_npz(ROOT / contract["inputs"][f"{role}_predictions"]["path"])
        for key in ("dataset_index", "sequence", "wait_ms", "receiver_timestamp_us"):
            if not exact(material[key], state[key]):
                raise RuntimeError((role, key, "state/material identity"))
        if len(predictions["method_names"]) != len(METHODS) or set(map(str, predictions["method_names"])) != set(METHODS):
            raise RuntimeError((role, "method identity"))
        boxes = {
            METHODS[0]: core.build_heading(material, state["baseline_scalar"])[0],
            METHODS[2]: core.build_heading(material, state["baseline_direction"])[0],
        }
        # Rebuild the actual per-candidate timing geometry, not a condition-wide delta.
        for method, key in ((METHODS[1], "timing_scalar_correction"),
                            (METHODS[3], "timing_direction_correction")):
            projected = core.build_heading(material, np.zeros_like(state["baseline_scalar"]))[0]
            remote = material["source"] == 1
            projected[remote, :2] += state[key][remote]
            boxes[method] = projected
        provenance, mapping = map_provenance(material, conditions[role], role, label_root)
        if mapping["unmatched"] or mapping["ambiguous"] or np.any(provenance < 0):
            raise RuntimeError((role, "GT provenance mapping incomplete", mapping))
        panels, sequence_rows, outcomes, full_exact = evaluate_role(
            material, predictions, boxes, provenance, role)
        if not all(full_exact.values()):
            raise RuntimeError((role, "official all-GT replay drift", full_exact))
        provenance_path = args.output / f"{role}_gt_provenance.npz"
        outcomes_path = args.output / f"{role}_source_split_outcomes.npz"
        save_npz(provenance_path, gt_from_side=provenance,
                 dataset_index=material["dataset_index"], gt_offsets=material["gt_offsets"])
        save_npz(outcomes_path, **outcomes)
        outputs[role] = {"gt_provenance": evidence(provenance_path),
                         "source_split_outcomes": evidence(outcomes_path)}
        mapping_audits[role] = mapping; exact_checks[role] = full_exact
        panels_all.extend(panels); sequence_all.extend(sequence_rows)
    accepted = load(ROOT / contract["inputs"]["timing_formal_summary"]["path"])
    accepted_table = {(row["role"], row["wait_ms"], row["method"]): row for row in accepted["panels"]}
    full_ap_replay = all(
        abs(row[f"ap{t}"] - accepted_table[(row["role"], row["wait_ms"], row["method"])][f"ap{t}"]) <= 1e-10
        for row in panels_all if row["source_group"] == "all" for t in (50, 70))
    if not full_ap_replay:
        raise RuntimeError("accepted timing all-GT AP replay drift")
    bootstrap_rows = bootstrap(sequence_all)
    panel_keys = {(row["role"], int(row["wait_ms"]), row["method"], row["source_group"])
                  for row in panels_all}
    expected_keys = {(role, wait, method, group) for role in ROLES for wait in WAITS
                     for method in METHODS for group in GROUPS}
    summary = {
        "schema": "peer07_spd_gt_source_split_ap_formal_producer_v1",
        "status": "formal_current_recipe_source_split_complete_review_pending",
        "created_at": contract["frozen_at"], "task_id": TASK, "owner": OWNER,
        "contract": evidence(args.contract), "formal_panels": panels_all,
        "per_sequence_panels": sequence_all, "paired_sequence_bootstrap": bootstrap_rows,
        "mapping_audits": mapping_audits, "official_full_gt_replay_checks": exact_checks,
        "outputs": outputs,
        "checks": {
            "accepted_timing_full_gt_ap_replayed": full_ap_replay,
            "official_all_gt_protocol_unchanged_and_exact": all(all(value.values()) for value in exact_checks.values()),
            "all_gt_mapped_uniquely_below_1e_3m": all(not value["unmatched"] and not value["ambiguous"] for value in mapping_audits.values()),
            "author21_and_main_domain_labels_byte_aggregate_identical": main_aggregate == author_aggregate,
            "formal_panel_coverage_exact": panel_keys == expected_keys,
            "other_source_is_ignore_after_target_matching": True,
            "no_role_wait_sequence_or_cell_method_selection": True,
            "paired_sequence_bootstrap_fixed_10000": len(bootstrap_rows) == 32 and all(row["bootstrap_replicates"] == 10000 for row in bootstrap_rows),
            "zero_detector_or_backbone_forward": True, "zero_optimizer_or_head_update": True,
            "zero_gpu": True, "shared_inputs_not_copied": True,
        },
        "label_identity": {"main_domain": main_aggregate, "author21": author_aggregate},
        "actual_cost": {
            "wall_seconds": time.monotonic() - began, "cpu_process_seconds": time.process_time() - cpu_began,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0, "gpu_seconds": 0,
            "new_output_bytes": sum(item.stat().st_size for item in args.output.glob("*.npz")),
        },
        "boundary": contract["boundary"],
        "remaining_coverage": contract["remaining_coverage"],
    }
    if not all(summary["checks"].values()):
        raise RuntimeError(summary["checks"])
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "panels": len(panels_all),
                      "mapping": mapping_audits}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
