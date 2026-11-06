#!/usr/bin/env python3
"""Strict-grid400 V2 native/native-prefix attribution by unchanged GT provenance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import gzip
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
BASE = ROOT / "preexperiment/research_baselines"
sys.path.insert(0, str(BASE))
from e1_receiver_value_core_v1_20261020 import greedy, iou32  # noqa: E402
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap  # noqa: E402


TASK = "v2_label_provenance_strict400_native_attribution_v1_20261029"
ROLES = ("mechanism_holdout14", "val21")
CATEGORIES = ("ego_only", "common_explicit_link", "remote_only",
              "common_with_fallback_collision")


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


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def lineage_map(path: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, int]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    totals = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["split"]), str(row["sequence"]), str(row["timestamp"]))
            if key in result:
                raise RuntimeError(("duplicate lineage frame", key))
            labels = row["formal_labels"]
            removed = sum(
                int(group.get("remote_fallback_in_formal_eval_xy", 0))
                for group in row["duplicate_groups"]
                if str(group["category"]) == "common_with_fallback_collision")
            unmatched = len(row["cache_to_raw_yaml_mapping"]["unmatched_raw_formal_indices"])
            result[key] = {
                "categories": [str(label["category"]) for label in labels],
                "boxes": np.asarray(row["formal_cache_boxes"], np.float32).reshape(-1, 7),
                "removed_collision_roi": removed,
                "unmatched_raw_cache_roi": unmatched,
            }
            totals.update({"frames": 1, "formal_labels": len(labels),
                           "removed_collision_roi": removed,
                           "unmatched_raw_cache_roi": unmatched})
    return result, dict(totals)


def read_predictions(path: Path, expected_methods: tuple[str, ...]) -> dict[tuple[str, str], dict[str, Any]]:
    frames: dict[tuple[str, str], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("protocol") != "strict_grid400" or row.get("method") not in expected_methods:
                raise RuntimeError((path, "protocol/method drift"))
            key = (str(row["sequence"]), str(row["timestamp"]))
            methods = frames.setdefault(key, {})
            method = str(row["method"])
            if method in methods:
                raise RuntimeError((path, key, method, "duplicate"))
            methods[method] = row
    for key, methods in frames.items():
        if set(methods) != set(expected_methods):
            raise RuntimeError((path, key, "incomplete methods", sorted(set(expected_methods) - set(methods))))
    return frames


def source_panels(path: Path) -> dict[str, dict[str, Any]]:
    summary = load(path)
    return {str(row["method"]): row for row in summary["panels"] if row["group"] == "all"}


def inverse_matches(matches: np.ndarray, gt_count: int) -> np.ndarray:
    inverse = np.full(gt_count, -1, np.int32)
    for candidate, gt_index in enumerate(matches):
        if gt_index >= 0:
            inverse[int(gt_index)] = candidate
    return inverse


def transition(left: np.ndarray, right: np.ndarray, use: np.ndarray) -> dict[str, int]:
    left_hit, right_hit = left >= 0, right >= 0
    return {
        "denominator": int(np.count_nonzero(use)),
        "rescued": int(np.count_nonzero(use & ~left_hit & right_hit)),
        "damaged": int(np.count_nonzero(use & left_hit & ~right_hit)),
        "retained": int(np.count_nonzero(use & left_hit & right_hit)),
        "unmatched": int(np.count_nonzero(use & ~left_hit & ~right_hit)),
    }


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic()
    cpu_started = time.process_time()
    output.mkdir(parents=True, exist_ok=False)
    lineage, full_lineage = lineage_map(ROOT / contract["lineage"]["path"])
    methods = tuple(contract["methods"])
    if not methods or methods[0] != "native" or any(not value.startswith("native_prefix_") for value in methods[1:]):
        raise RuntimeError("method identity")
    category_code = {name: index for index, name in enumerate(CATEGORIES)}
    split_by_role = {"mechanism_holdout14": "train", "val21": "val"}
    gt_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    additive_checks: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}
    coverage: dict[str, Any] = {}
    saved_match_exact = True
    source_ap_max_error = 0.0

    for role in ROLES:
        role_spec = contract["roles"][role]
        frames = read_predictions(ROOT / role_spec["predictions"], methods)
        source = source_panels(ROOT / role_spec["source_summary"])
        if set(source) != set(methods):
            raise RuntimeError((role, "source panel methods"))
        scores_by_method: dict[str, list[np.ndarray]] = defaultdict(list)
        tp_by_method: dict[str, list[np.ndarray]] = defaultdict(list)
        candidate_offsets = {method: [0] for method in methods}
        gt_offsets = [0]
        gt_categories: list[int] = []
        gt_matches: dict[str, list[np.ndarray]] = defaultdict(list)
        role_occurrence = Counter()
        fallback_frames = 0
        removed_total = 0
        unmatched_total = 0

        for frame_ordinal, ((sequence, timestamp), rows) in enumerate(sorted(frames.items())):
            frame_key = (split_by_role[role], sequence, timestamp)
            if frame_key not in lineage:
                raise RuntimeError((role, "lineage frame missing", frame_key))
            frame = lineage[frame_key]
            native_row = rows["native"]
            gt = np.asarray(native_row["gt_boxes"], np.float32).reshape(-1, 7)
            if not np.array_equal(gt, frame["boxes"]):
                raise RuntimeError((role, frame_key, "formal GT lineage drift"))
            categories = frame["categories"]
            if len(categories) != len(gt) or any(name not in category_code for name in categories):
                raise RuntimeError((role, frame_key, "category drift"))
            gt_offsets.append(gt_offsets[-1] + len(gt))
            gt_categories.extend(category_code[name] for name in categories)
            role_occurrence.update(categories)
            fallback_frames += int(bool(native_row["fallback"]))
            removed_total += int(frame["removed_collision_roi"])
            unmatched_total += int(frame["unmatched_raw_cache_roi"])
            inverse: dict[str, np.ndarray] = {}
            method_counts: dict[str, dict[str, int]] = {}
            for method in methods:
                row = rows[method]
                if (str(row["role"]) != role or row["sequence"] != sequence or
                        str(row["timestamp"]) != timestamp or
                        not np.array_equal(np.asarray(row["gt_boxes"], np.float32), gt)):
                    raise RuntimeError((role, frame_key, method, "row identity drift"))
                boxes = np.asarray(row["boxes"], np.float32).reshape(-1, 7)
                scores = np.asarray(row["scores"], np.float64)
                matches = greedy(iou32(boxes, gt), np.arange(len(boxes)), .7).astype(np.int32)
                saved = np.asarray(row["matched_gt70"], np.int32)
                saved_match_exact = saved_match_exact and np.array_equal(matches, saved)
                if not np.array_equal(matches, saved):
                    raise RuntimeError((role, frame_key, method, "saved match drift"))
                tp = (matches >= 0).astype(np.int8)
                scores_by_method[method].append(scores)
                tp_by_method[method].append(tp)
                candidate_offsets[method].append(candidate_offsets[method][-1] + len(tp))
                inv = inverse_matches(matches, len(gt))
                inverse[method] = inv
                gt_matches[method].append(inv + candidate_offsets[method][-2] * (inv >= 0))
                method_counts[method] = {"outputs": len(tp), "tp70": int(tp.sum()),
                                         "fp70": int(len(tp) - tp.sum())}
            for method in methods[1:]:
                left, right = inverse["native"], inverse[method]
                total = transition(left, right, np.ones(len(gt), bool))
                total["net_matched_change"] = total["rescued"] - total["damaged"]
                condition_rows.append({
                    "role": role, "protocol": "strict_grid400", "frame_ordinal": frame_ordinal,
                    "sequence": sequence, "timestamp": timestamp,
                    "comparison": f"native_to_{method}", "fallback": bool(native_row["fallback"]),
                    "gt": len(gt), "native_outputs": method_counts["native"]["outputs"],
                    "prefix_outputs": method_counts[method]["outputs"],
                    "removed_collision_roi": int(frame["removed_collision_roi"]),
                    "unmatched_raw_cache_roi": int(frame["unmatched_raw_cache_roi"]), **total,
                })
        gt_category_array = np.asarray(gt_categories, np.int8)
        archive[f"{role}__gt_offsets"] = np.asarray(gt_offsets, np.int64)
        archive[f"{role}__gt_category_code"] = gt_category_array
        for method in methods:
            score = np.concatenate(scores_by_method[method])
            tp = np.concatenate(tp_by_method[method])
            inverse = np.concatenate(gt_matches[method]).astype(np.int64)
            offsets = np.asarray(candidate_offsets[method], np.int64)
            archive[f"{role}__{method}__candidate_offsets"] = offsets
            archive[f"{role}__{method}__score"] = score
            archive[f"{role}__{method}__tp70"] = tp
            archive[f"{role}__{method}__gt_matched_candidate"] = inverse
            metric = voc_ap(tp.tolist(), (1 - tp).tolist(), score.tolist(), int(gt_offsets[-1]), True)
            source_row = source[method]
            error = abs(float(metric["ap"]) - float(source_row["ap70"]))
            source_ap_max_error = max(source_ap_max_error, error)
            panel = {
                "role": role, "protocol": "strict_grid400", "method": method,
                "ap70": 100.0 * float(metric["ap"]), "tp70": int(metric["tp"]),
                "fp70": int(metric["fp"]), "gt": int(metric["gt"]),
                "outputs": len(tp), "source_ap70_abs_error": error,
            }
            if (panel["tp70"], panel["fp70"], panel["gt"]) != (
                    int(source_row["tp70"]), int(source_row["fp70"]), int(source_row["gt"])):
                raise RuntimeError((role, method, "source count drift"))
            panels.append(panel)
        native_inverse = archive[f"{role}__native__gt_matched_candidate"]
        panel_table = {row["method"]: row for row in panels if row["role"] == role}
        for method in methods[1:]:
            right_inverse = archive[f"{role}__{method}__gt_matched_candidate"]
            comparison = f"native_to_{method}"
            for category in CATEGORIES:
                use = gt_category_array == category_code[category]
                row = {"comparison": comparison, "role": role, "protocol": "strict_grid400",
                       "category": category, **transition(native_inverse, right_inverse, use)}
                row["net_matched_change"] = row["rescued"] - row["damaged"]
                gt_rows.append(row)
            source_panel, target_panel = panel_table["native"], panel_table[method]
            selected = [row for row in gt_rows if row["comparison"] == comparison and row["role"] == role]
            net = sum(row["net_matched_change"] for row in selected)
            additive_checks.append({
                "comparison": comparison, "role": role,
                "category_denominator_sum": sum(row["denominator"] for row in selected),
                "gt": source_panel["gt"], "category_net_sum": net,
                "tp_delta": target_panel["tp70"] - source_panel["tp70"],
                "output_delta": target_panel["outputs"] - source_panel["outputs"],
                "fp_delta": target_panel["fp70"] - source_panel["fp70"],
                "tp_additive": net == target_panel["tp70"] - source_panel["tp70"],
                "fp_additive": (target_panel["outputs"] - source_panel["outputs"] - net) ==
                               target_panel["fp70"] - source_panel["fp70"],
                "denominator_additive": sum(row["denominator"] for row in selected) == source_panel["gt"],
            })
        coverage[role] = {
            "frames": len(frames), "gt": int(gt_offsets[-1]),
            "fallback_frames": fallback_frames, "category_denominators": dict(role_occurrence),
            "removed_collision_roi": removed_total, "unmatched_raw_cache_roi": unmatched_total,
        }

    gt_path = output / "gt_transition_ledger.jsonl"
    condition_path = output / "condition_ledger.jsonl"
    archive_path = output / "match_archive.npz"
    write_jsonl(gt_path, gt_rows)
    write_jsonl(condition_path, condition_rows)
    np.savez_compressed(archive_path, category_names=np.asarray(CATEGORIES, dtype="U48"), **archive)
    checks = {
        "strict_grid400_only": True,
        "v2_native_and_six_prefix_methods_complete": len(methods) == 7,
        "formal_role_frame_counts": (coverage["mechanism_holdout14"]["frames"] == 293 and
                                     coverage["val21"]["frames"] == 1029),
        "full_lineage_identity": full_lineage == {
            "frames": 4155, "formal_labels": 54576,
            "removed_collision_roi": 143, "unmatched_raw_cache_roi": 256},
        "formal_gt_lineage_exact": True,
        "saved_tp70_matches_replayed": saved_match_exact,
        "source_formal_ap_replayed": source_ap_max_error <= 1e-12,
        "all_additive_ledgers_close": all(row["tp_additive"] and row["fp_additive"] and
                                          row["denominator_additive"] for row in additive_checks),
        "all_categories_retained_including_zero_denominator": set(row["category"] for row in gt_rows) == set(CATEGORIES),
        "zero_formal_gt_modification": True,
        "zero_detector_or_backbone_forward": True,
        "zero_optimizer_or_head_update": True,
        "zero_gpu": True,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "source_ap_max_error": source_ap_max_error})
    summary = {
        "schema": "v2_label_provenance_strict400_native_attribution_producer_v1",
        "status": "complete_producer_review_pending", "created_at": now(),
        "task_id": TASK, "owner": contract["owner"], "contract": evidence(ROOT / contract["contract_path"]),
        "methods": list(methods), "comparisons": [f"native_to_{method}" for method in methods[1:]],
        "formal_panels": panels, "gt_transition_ledger": gt_rows,
        "condition_ledger_rows": len(condition_rows), "additive_checks": additive_checks,
        "coverage": {"full_lineage": full_lineage, "strict400_roles": coverage},
        "outputs": {"gt_transition_ledger": evidence(gt_path),
                    "condition_ledger": evidence(condition_path),
                    "match_archive": evidence(archive_path)},
        "source_ap_replay_max_abs_error": source_ap_max_error,
        "checks": checks, "prediction_before_result": contract["prediction_before_result"],
        "actual_cost": {"wall_seconds": time.monotonic() - started,
                        "cpu_process_seconds": time.process_time() - cpu_started,
                        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                        "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0,
                        "gpu_seconds": 0},
        "boundary": (
            "V2V4Real strict_grid400 saved native/native-prefix predictions and unchanged formal GT only. "
            "This is separate from the current same-candidate stale/KF/task/M1 population; provenance is "
            "annotation-coordinate lineage, not physical truth, repaired GT, or an inference gate."
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
        raise RuntimeError("runner drift")
    for item in contract["inputs"].values():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(("input drift", item["path"]))
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = compute(contract, args.output)
    print(json.dumps({"status": summary["status"], "panels": len(summary["formal_panels"]),
                      "comparisons": len(summary["comparisons"]),
                      "actual_cost": summary["actual_cost"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
