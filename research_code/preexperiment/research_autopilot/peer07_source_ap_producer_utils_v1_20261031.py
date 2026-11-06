#!/usr/bin/env python3
"""Formal SPD GT-provenance AP for the frozen current TraF recipe arms.

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

from e1_receiver_value_core_v1_20261020 import common_mask, greedy, iou32
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap


TASK = "peer07_spd_gt_source_split_ap_v1_20261031"
OWNER = "research_autopilot_peer07_spd_gt_source_split_ap_v1"
ROLES = ("mechanism_holdout14", "val21")
WAITS = (0, 100, 200, 400)
METHODS = (
    "heading_directional_information_existing_replay",
    "heading_isotropic_equal_information_existing_replay",
    "heading_scalar_equal_covariance_new",
    "heading_q005_base_exact_replay",
    "heading_anchor_gate_exact_replay",
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
