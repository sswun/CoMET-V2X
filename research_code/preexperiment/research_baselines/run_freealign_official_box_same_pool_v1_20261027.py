#!/usr/bin/env python3
"""Official FreeAlign box-only same-pool control on frozen TraF outputs.

This adapter calls the verified upstream ``get_pose_rotation`` entry with the
current ego's independent boxes and the legally arrived sender-local boxes.
It never calls ``freealign_training`` (whose acceptance check consumes GT
poses).  A successful upstream pose maps sender-local boxes into the receiver
frame.  Insufficient/no-match/exception cases retain the existing nominal
stale boxes so every arm has exactly the same source population.

This is intentionally a box-registration module control, not a reproduction
of the full FreeAlign system (no learned graph matcher, temporal search, or
feature-fusion integration).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "preexperiment/research_baselines"
FREEALIGN = ROOT / "third_party/external/FreeAlign_box_control_20261027"
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(FREEALIGN))

from e1_receiver_value_core_v1_20261020 import atomic, greedy, iou32, sha256
from object_history_selector_core_v1_20261021 import PackedDataset
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap
from run_shared_offset_optional_formal_v1_20261026 import (
    direct_scores, load_npz, ordered, source_map, update_features,
)
from opencood.utils.box_utils import boxes_to_corners_3d, corner_to_center
from freealign.match import match_v7_with_detection as official


WAITS = (0, 100, 200, 400)
BETAS = (1.0, 1.2, 1.4)
BASE_METHODS = ("freealign_raw", "freealign_source_u", "freealign_direct_u")
METHODS = BASE_METHODS + tuple(
    f"{name}_beta{beta:.1f}" for name in BASE_METHODS for beta in BETAS
)
STATUS_NAMES = (
    "estimated", "insufficient_boxes_stale_fallback",
    "no_common_subgraph_stale_fallback", "exception_stale_fallback",
    "nonfinite_stale_fallback",
)
STATUS_CODE = {name: index for index, name in enumerate(STATUS_NAMES)}


def angle_wrap(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


@lru_cache(maxsize=64)
def raw_boxes(path_text: str) -> np.ndarray:
    with np.load(path_text, allow_pickle=False) as archive:
        return archive["pred_boxes"].copy()


def capture_official_pose(ego_corners: np.ndarray, source_corners: np.ndarray
                          ) -> tuple[np.ndarray, int, list[list[int]], str | None, float]:
    """Call the unmodified official entry while observing its MCS return."""
    captured: dict[str, Any] = {}
    original = official.find_common_subgraph

    def observe(graph1, graph2, opt):
        result = original(graph1, graph2, opt)
        captured["mcs_size"] = int(result[0])
        captured["matches"] = [[int(a), int(b)] for a, b in result[1]]
        return result

    official.find_common_subgraph = observe
    started = time.perf_counter()
    error = None
    try:
        pose = np.asarray(official.get_pose_rotation(ego_corners, source_corners), np.float64)
    except Exception as exc:  # Preserve upstream failures; the population-safe adapter falls back.
        pose = np.zeros(6, np.float64)
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.perf_counter() - started
        official.find_common_subgraph = original
    return pose, int(captured.get("mcs_size", 0)), captured.get("matches", []), error, elapsed


def apply_absolute_pose(source_boxes: np.ndarray, pose: np.ndarray) -> np.ndarray:
    output = source_boxes.astype(np.float32).copy()
    theta = math.radians(float(pose[4]))
    rotation = np.asarray([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta), math.cos(theta)]], np.float64)
    output[:, :2] = (source_boxes[:, :2].astype(np.float64) @ rotation.T + pose[:2]).astype(np.float32)
    output[:, 6] = angle_wrap(source_boxes[:, 6].astype(np.float64) + theta).astype(np.float32)
    return output


def select_pilot(indices: list[int], meta: list[dict], per_wait: int) -> list[int]:
    chosen: list[int] = []
    for wait in WAITS:
        group = [index for index in indices if int(meta[index]["wait_ms"]) == wait]
        if len(group) <= per_wait:
            chosen.extend(group)
        else:
            positions = np.linspace(0, len(group) - 1, per_wait).round().astype(int)
            chosen.extend(group[int(position)] for position in positions)
    return chosen


def input_boxes(dataset: PackedDataset, index: int, pool: dict[str, np.ndarray],
                common: dict[str, np.ndarray], condition: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, str, float, float]:
    """Recover exact selected ego/source-local boxes and the nominal stale arm."""
    meta = dataset.meta[index]
    a, b = common["condition_offsets"][condition:condition + 2].astype(int)
    source = common["source"][a:b]
    ego_local = np.flatnonzero(source == 0)
    remote_local = np.flatnonzero(source == 1)
    nominal_stale = common["stale_boxes"][a:b][remote_local].copy()
    if not len(remote_local):
        return (common["stale_boxes"][a:b][ego_local].copy(),
                np.empty((0, 7), np.float32), nominal_stale, remote_local, True,
                "empty_legal_packet", 0.0, 0.0)
    if meta.get("source") is None:
        raise AssertionError("remote population exists without a legal source packet")
    pool_kind = pool["hypothesis_type"]
    pool_ego = np.flatnonzero(pool_kind == 0)
    pool_stale = np.flatnonzero(pool_kind == 1)
    assert len(pool_ego) == len(ego_local) and len(pool_stale) == len(remote_local)
    ego_ids = pool["source_detection_indices"][pool_ego].astype(int)
    source_ids = pool["source_detection_indices"][pool_stale].astype(int)
    frozen_ego = common["stale_boxes"][a:b][ego_local].copy()
    if meta.get("ego_raw_path"):
        ego_from_file = raw_boxes(str(meta["ego_raw_path"]))[ego_ids].astype(np.float32)
        ego_identity = np.array_equal(ego_from_file, frozen_ego)
    else:
        ego_identity = True
    ego_raw = frozen_ego
    if not ego_identity:
        raise AssertionError("selected current-ego boxes differ from the frozen pool")
    # The 39D cache permanently stores sender-local x/y/z and sin/cos(yaw).
    # Dimensions survive rigid projection and are copied from the stale arm.
    stale_features = pool["features"][pool_stale]
    reconstructed = pool["boxes"][pool_stale].astype(np.float32).copy()
    reconstructed[:, :3] = stale_features[:, 12:15].astype(np.float32)
    reconstructed[:, 6] = np.arctan2(stale_features[:, 16], stale_features[:, 17]).astype(np.float32)
    if meta.get("source_raw_path"):
        source_raw = raw_boxes(str(meta["source_raw_path"]))[source_ids].astype(np.float32)
        center_max = float(np.max(np.abs(source_raw[:, :3] - reconstructed[:, :3]), initial=0.0))
        yaw_max = float(np.max(np.abs(angle_wrap(
            source_raw[:, 6].astype(np.float64) - reconstructed[:, 6].astype(np.float64))), initial=0.0))
        origin = "source_raw_file"
    else:
        source_raw = reconstructed
        center_max = 0.0; yaw_max = 0.0
        origin = "frozen_39d_sender_fields"
    return ego_raw, source_raw, nominal_stale, remote_local, ego_identity, origin, center_max, yaw_max


def materialize_role(dataset: PackedDataset, role: str, common: dict[str, np.ndarray],
                     selected_conditions: list[int] | None, output: Path) -> dict[str, Any]:
    full_indices = common["dataset_index"].astype(int).tolist()
    if selected_conditions is None:
        condition_ids = list(range(len(full_indices)))
    else:
        wanted = set(selected_conditions)
        condition_ids = [condition for condition, index in enumerate(full_indices) if index in wanted]
    boxes_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    transforms: list[np.ndarray] = []
    mcs_sizes: list[int] = []
    statuses: list[int] = []
    official_seconds: list[float] = []
    adapter_seconds: list[float] = []
    nominal_xy_error: list[float] = []
    nominal_yaw_error: list[float] = []
    source_age_ms: list[float] = []
    offsets = [0]
    dataset_indices: list[int] = []
    condition_waits: list[int] = []
    timestamps: list[int] = []
    sequences: list[str] = []
    exception_rows: list[dict[str, Any]] = []
    match_rows: list[list[list[int]]] = []
    identity_fallback_exact = True
    legal_time = True
    population_exact = True
    raw_input_identity = True
    source_input_origins: dict[str, int] = defaultdict(int)
    reconstructed_center_max = 0.0
    reconstructed_yaw_max = 0.0
    started = time.monotonic()

    for local_count, condition in enumerate(condition_ids):
        index = full_indices[condition]
        meta = dataset.meta[index]
        pool = dataset.pool(index)
        a, b = common["condition_offsets"][condition:condition + 2].astype(int)
        base_boxes = common["stale_boxes"][a:b]
        base_features = common["stale_features"][a:b]
        raw = common["raw_scores"][a:b]
        source = common["source"][a:b]
        protected = common["protected"][a:b]
        ego_raw, source_raw, stale_remote, remote_local, ego_identity, source_origin, center_error, yaw_error = input_boxes(
            dataset, index, pool, common, condition)
        raw_input_identity &= bool(ego_identity)
        source_input_origins[source_origin] += 1
        reconstructed_center_max = max(reconstructed_center_max, center_error)
        reconstructed_yaw_max = max(reconstructed_yaw_max, yaw_error)
        adapter_started = time.perf_counter()
        ego_corners = boxes_to_corners_3d(ego_raw, "hwl")
        source_corners = boxes_to_corners_3d(source_raw, "hwl")
        pose, mcs_size, matches, error, call_seconds = capture_official_pose(
            ego_corners, source_corners)
        if error is not None:
            status = "exception_stale_fallback"
        elif not np.all(np.isfinite(pose)):
            status = "nonfinite_stale_fallback"
        elif len(ego_raw) <= 2 or len(source_raw) <= 2:
            status = "insufficient_boxes_stale_fallback"
        elif mcs_size < 3:
            status = "no_common_subgraph_stale_fallback"
        else:
            status = "estimated"
        if status == "estimated":
            remote = apply_absolute_pose(source_raw, pose)
        else:
            remote = stale_remote.copy()
            identity_fallback_exact &= np.array_equal(remote, stale_remote)
        if len(remote) != int((source == 1).sum()):
            population_exact = False
            raise AssertionError("FreeAlign arm changed the frozen remote population")

        combined = base_boxes.copy()
        combined[remote_local] = remote
        protected_boxes = base_boxes[np.flatnonzero(protected)]
        protected_scores = raw[np.flatnonzero(protected)]
        features = update_features(base_features, combined, protected_boxes, protected_scores)
        adapter_elapsed = time.perf_counter() - adapter_started

        nominal_value = meta.get("source", {}).get("source_to_current_ego") if meta.get("source") else None
        nominal = np.asarray(nominal_value, np.float64) if nominal_value is not None else None
        if status == "estimated" and nominal is not None:
            nominal_yaw = math.degrees(math.atan2(float(nominal[1, 0]), float(nominal[0, 0])))
            nominal_xy_error.append(float(np.linalg.norm(pose[:2] - nominal[:2, 3])))
            nominal_yaw_error.append(abs(float(angle_wrap(math.radians(pose[4] - nominal_yaw)))))
        else:
            nominal_xy_error.append(float("nan")); nominal_yaw_error.append(float("nan"))
        if meta.get("source"):
            receiver = int(meta["receiver_timestamp_us"])
            sender = int(meta["source"]["source_timestamp_us"])
            age = (receiver - sender) / 1000.0
            legal_time &= sender <= receiver and abs(age - float(meta["source"]["actual_age_ms"])) <= 0.0011
        else:
            age = 0.0

        boxes_rows.append(combined.astype(np.float32))
        feature_rows.append(features.astype(np.float32))
        transforms.append(pose.astype(np.float64))
        mcs_sizes.append(mcs_size); statuses.append(STATUS_CODE[status])
        official_seconds.append(call_seconds); adapter_seconds.append(adapter_elapsed)
        source_age_ms.append(age); offsets.append(offsets[-1] + len(combined))
        dataset_indices.append(index); condition_waits.append(int(meta["wait_ms"]))
        timestamps.append(int(meta["receiver_timestamp_us"])); sequences.append(str(meta["sequence"]))
        match_rows.append(matches)
        if error is not None:
            exception_rows.append({"dataset_index": index, "error": error})
        if (local_count + 1) % 1000 == 0:
            atomic(output / f"{role}_progress.json", {
                "stage": "official_registration", "completed": local_count + 1,
                "total": len(condition_ids), "wall_seconds": time.monotonic() - started,
            })

    archive = {
        "freealign_boxes": np.concatenate(boxes_rows) if boxes_rows else np.empty((0, 7), np.float32),
        "freealign_features": np.concatenate(feature_rows) if feature_rows else np.empty((0, 39), np.float32),
        "condition_offsets": np.asarray(offsets, np.int64),
        "dataset_index": np.asarray(dataset_indices, np.int64),
        "sequence": np.asarray(sequences, dtype="U16"),
        "wait_ms": np.asarray(condition_waits, np.int16),
        "receiver_timestamp_us": np.asarray(timestamps, np.int64),
        "source_age_ms": np.asarray(source_age_ms, np.float64),
        "pose_source_to_ego_xyyaw_deg": np.asarray(transforms, np.float64),
        "mcs_size": np.asarray(mcs_sizes, np.int16),
        "status_code": np.asarray(statuses, np.int8),
        "status_names": np.asarray(STATUS_NAMES, dtype="U40"),
        "official_call_seconds": np.asarray(official_seconds, np.float64),
        "adapter_seconds": np.asarray(adapter_seconds, np.float64),
        "nominal_pose_xy_error_m": np.asarray(nominal_xy_error, np.float64),
        "nominal_pose_yaw_error_rad": np.asarray(nominal_yaw_error, np.float64),
    }
    materialized_path = output / f"{role}_freealign_materialized.npz"
    np.savez_compressed(materialized_path, **archive)
    with (output / f"{role}_matches.jsonl").open("w") as stream:
        for index, matches in zip(dataset_indices, match_rows):
            stream.write(json.dumps({"dataset_index": index, "matches": matches}, separators=(",", ":")) + "\n")
    with (output / f"{role}_exceptions.jsonl").open("w") as stream:
        for row in exception_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    counts = {name: int((archive["status_code"] == code).sum()) for name, code in STATUS_CODE.items()}
    calls = archive["official_call_seconds"]
    adapters = archive["adapter_seconds"]
    return {
        "role": role, "conditions": len(condition_ids), "candidates": len(archive["freealign_boxes"]),
        "status_counts": counts, "success_rate": counts["estimated"] / max(len(condition_ids), 1),
        "identity_fallback_exact": bool(identity_fallback_exact),
        "same_population": bool(population_exact), "legal_packet_times": bool(legal_time),
        "ego_raw_selected_identity_exact": bool(raw_input_identity),
        "source_input_origins": dict(source_input_origins),
        "raw_backed_sender_reconstruction_center_max_abs_m": reconstructed_center_max,
        "raw_backed_sender_reconstruction_yaw_max_abs_rad": reconstructed_yaw_max,
        "official_call_seconds": {"sum": float(calls.sum()), "mean": float(calls.mean()),
            "median": float(np.median(calls)), "p95": float(np.percentile(calls, 95)),
            "max": float(calls.max())},
        "adapter_seconds": {"sum": float(adapters.sum()), "mean": float(adapters.mean()),
            "median": float(np.median(adapters)), "p95": float(np.percentile(adapters, 95)),
            "max": float(adapters.max())},
        "materialized": str(materialized_path.relative_to(ROOT)),
        "materialized_sha256": sha256(materialized_path),
        "matches_sha256": sha256(output / f"{role}_matches.jsonl"),
        "exceptions_sha256": sha256(output / f"{role}_exceptions.jsonl"),
    }


def evaluate_role(dataset: PackedDataset, role: str, common: dict[str, np.ndarray],
                  materialized: dict[str, np.ndarray], readout: Path, output: Path
                  ) -> dict[str, Any]:
    maps = json.loads((readout / "source_maps.json").read_text())
    calibration = json.loads((readout / "calibration.json").read_text())
    direct = direct_scores({"freealign": materialized["freealign_features"]}, readout)["freealign"]
    source_u_all = source_map(
        common["raw_scores"], common["source"], maps["maps"]["raw_to_utility"])
    direct_u_all = source_map(direct, common["source"], maps["maps"]["direct_to_utility"])
    results = defaultdict(lambda: {name: {"scores": [], "tp50": [], "tp70": [],
        "gt": 0, "outputs": 0, "remote": 0} for name in METHODS})
    saved = {name: {"candidate": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
             for name in METHODS}
    condition_outputs = np.zeros((len(METHODS), len(materialized["dataset_index"])), np.int32)
    condition_tp70 = np.zeros_like(condition_outputs)
    condition_fp70 = np.zeros_like(condition_outputs)
    common_index_to_condition = {int(index): condition for condition, index in enumerate(common["dataset_index"])}

    for condition, index_value in enumerate(materialized["dataset_index"]):
        index = int(index_value)
        common_condition = common_index_to_condition[index]
        ca, cb = common["condition_offsets"][common_condition:common_condition + 2].astype(int)
        a, b = materialized["condition_offsets"][condition:condition + 2].astype(int)
        assert cb - ca == b - a
        pool = dataset.pool(index)
        raw = common["raw_scores"][ca:cb]
        source = common["source"][ca:cb]
        protected = common["protected"][ca:cb]
        boxes = materialized["freealign_boxes"][a:b]
        source_u = source_u_all[ca:cb]
        direct_u = direct_u_all[a:b]
        wait = int(materialized["wait_ms"][condition])
        for method_index, method in enumerate(METHODS):
            base, _, beta_text = method.partition("_beta")
            family = base.split("_", 1)[1]
            scores = raw if family == "raw" else source_u if family == "source_u" else direct_u
            threshold = None
            if beta_text:
                calibration_name = {"raw": "protected_raw", "source_u": "protected_source_u",
                                    "direct_u": "protected_direct_u"}[family]
                threshold = calibration["methods"][calibration_name]["working_points"][beta_text]
            ids, ranks = ordered(raw, source, protected, scores, threshold)
            ious = iou32(boxes[ids], pool["gt_boxes"])
            match50 = greedy(ious, np.arange(len(ids)), .5)
            match70 = greedy(ious, np.arange(len(ids)), .7)
            row = results[wait][method]
            row["scores"].append(ranks); row["tp50"].append((match50 >= 0).astype(np.int8))
            row["tp70"].append((match70 >= 0).astype(np.int8)); row["gt"] += len(pool["gt_boxes"])
            row["outputs"] += len(ids); row["remote"] += int((source[ids] == 1).sum())
            record = saved[method]
            record["candidate"].append((ca + ids).astype(np.int64)); record["score"].append(ranks)
            record["tp50"].append((match50 >= 0).astype(np.int8)); record["tp70"].append((match70 >= 0).astype(np.int8))
            record["offset"].append(record["offset"][-1] + len(ids))
            condition_outputs[method_index, condition] = len(ids)
            condition_tp70[method_index, condition] = int((match70 >= 0).sum())
            condition_fp70[method_index, condition] = int((match70 < 0).sum())

    panels = []
    for wait in WAITS:
        for method in METHODS:
            row = results[wait][method]
            scores = np.concatenate(row["scores"]) if row["scores"] else np.empty(0)
            panel = {"wait_ms": wait, "method": method, "outputs": row["outputs"],
                     "remote": row["remote"]}
            for threshold in (50, 70):
                tp = np.concatenate(row[f"tp{threshold}"]) if row[f"tp{threshold}"] else np.empty(0, np.int8)
                metric = voc_ap(tp.tolist(), (1 - tp).tolist(), scores.tolist(), row["gt"], True)
                panel[f"ap{threshold}"] = metric["ap"] * 100.0
                if threshold == 70:
                    panel.update(tp=metric["tp"], fp=metric["fp"], gt=metric["gt"])
            panels.append(panel)
    archive: dict[str, np.ndarray] = {"method_names": np.asarray(METHODS, dtype="U48")}
    for method, record in saved.items():
        for key in ("candidate", "score", "tp50", "tp70"):
            archive[f"{method}__{key}"] = np.concatenate(record[key])
        archive[f"{method}__offset"] = np.asarray(record["offset"], np.int64)
    archive.update(condition_outputs=condition_outputs, condition_tp70=condition_tp70,
                   condition_fp70=condition_fp70, dataset_index=materialized["dataset_index"])
    predictions_path = output / f"{role}_freealign_predictions.npz"
    np.savez_compressed(predictions_path, **archive)
    return {"role": role, "panels": panels, "predictions": str(predictions_path.relative_to(ROOT)),
            "predictions_sha256": sha256(predictions_path)}


def synthetic_probes() -> dict[str, Any]:
    source = np.asarray([
        [0., 0., 0., 1.5, 1.8, 4.0, 0.1], [8., 1., 0., 1.6, 1.9, 4.2, -.2],
        [-3., 7., 0., 1.7, 2.0, 4.5, .5], [12., -6., 0., 1.4, 1.7, 3.8, -.6],
        [-8., -5., 0., 1.6, 1.8, 4.1, .8],
    ], np.float32)
    expected = np.asarray([1.2, -.7, 0., 0., 6.0, 0.], np.float64)
    target = apply_absolute_pose(source, expected)
    source_corners = boxes_to_corners_3d(source, "hwl")
    target_corners = boxes_to_corners_3d(target, "hwl")
    pose, mcs, _, error, elapsed = capture_official_pose(target_corners, source_corners)
    reverse, reverse_mcs, _, reverse_error, _ = capture_official_pose(source_corners, target_corners)
    zero = apply_absolute_pose(source, np.zeros(6, np.float64))
    roundtrip = corner_to_center(source_corners, "hwl")
    return {
        "source_to_ego_expected": expected.tolist(), "source_to_ego_estimated": pose.tolist(),
        "source_to_ego_max_abs_error": float(np.max(np.abs(pose - expected))),
        "source_to_ego_mcs_size": mcs, "source_to_ego_error": error,
        "reverse_estimated": reverse.tolist(), "reverse_mcs_size": reverse_mcs,
        "reverse_error": reverse_error,
        "direction_distinguished": bool(np.max(np.abs(reverse - expected)) > 1.0),
        "zero_pose_application_bit_exact": bool(np.array_equal(zero, source)),
        "hwl_corner_roundtrip_center_dimensions_max_abs_error": float(np.max(np.abs(roundtrip[:, :6] - source[:, :6]))),
        "official_call_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input-addendum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--pilot-per-wait", type=int, default=32)
    args = parser.parse_args()
    args.contract = args.contract.resolve()
    args.input_addendum = args.input_addendum.resolve()
    args.output = args.output.resolve()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    started = time.monotonic()
    contract = json.loads(args.contract.read_text())
    addendum = json.loads(args.input_addendum.read_text())
    assert contract["owner"] == "research_autopilot_freealign_box_same_pool_v1"
    assert contract["upstream_entry"] == "get_pose_rotation"
    assert contract["gt_pose_entry_excluded"] is True
    assert addendum["base_contract_sha256"] == sha256(args.contract)
    assert addendum["recovery_scope"] == "val21_sender_local_input_reconstruction_only"
    for identity in contract["frozen_identity"].values():
        path = ROOT / identity["path"]
        assert sha256(path) == identity["sha256"], path
    assert "embodied-research.slice" in Path("/proc/self/cgroup").read_text()
    assert os.environ.get("EMBODIED_THERMAL_THROTTLE") == "1"

    train_path = ROOT / contract["datasets"]["mechanism_holdout14"]["path"]
    val_path = ROOT / contract["datasets"]["val21"]["path"]
    formal_path = ROOT / contract["existing_formal_materialized_root"]
    readout = ROOT / contract["readout_root"]
    roles = {
        "mechanism_holdout14": (PackedDataset(train_path), load_npz(formal_path / "mechanism_holdout14_materialized.npz")),
        "val21": (PackedDataset(val_path), load_npz(formal_path / "val21_materialized.npz")),
    }
    probes = synthetic_probes()
    assert probes["source_to_ego_max_abs_error"] < 1e-5
    assert probes["direction_distinguished"] and probes["zero_pose_application_bit_exact"]
    summaries: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for role, (dataset, common) in roles.items():
        selected = None
        if args.mode == "pilot":
            full = common["dataset_index"].astype(int).tolist()
            selected = select_pilot(full, dataset.meta, args.pilot_per_wait)
        summaries[role] = materialize_role(dataset, role, common, selected, args.output)
        if args.mode == "full":
            materialized = load_npz(ROOT / summaries[role]["materialized"])
            evaluations[role] = evaluate_role(dataset, role, common, materialized, readout, args.output)

    result = {
        "schema": f"freealign_official_box_same_pool_{args.mode}_v1",
        "status": f"complete_{args.mode}", "mode": args.mode,
        "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": str(args.contract.relative_to(ROOT)), "contract_sha256": sha256(args.contract),
        "input_addendum": str(args.input_addendum.relative_to(ROOT)),
        "input_addendum_sha256": sha256(args.input_addendum),
        "official_scope": "verified official get_pose_rotation box-registration module only; not full FreeAlign",
        "adapter_semantics": {
            "input": "all frozen current ego independent boxes and same-population legally arrived sender-local boxes",
            "box_order": "hwl; [x,y,z,h,w,l,yaw_radians] converted with official boxes_to_corners_3d",
            "direction": "estimated full planar source-local to current-ego transform",
            "success": "apply upstream x/y/yaw exactly to sender-local source identities",
            "fallback": "insufficient/no-match/exception/nonfinite retains the already reviewed nominal stale receiver boxes",
            "population": "same source identities and raw scores as stale/CV/constant/KF; protected current-ego prefix unchanged",
            "time": "source packet timestamp must not exceed receiver timestamp; no future packet or temporal search",
            "gt": "GT used only after geometry is frozen for AP matching; freealign_training and its GT-pose acceptance are never called",
        },
        "synthetic_probes": probes, "roles": summaries, "evaluations": evaluations,
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0,
            "gpu_seconds": 0, "cpu_threads": 1,
        },
        "script_sha256": sha256(Path(__file__)),
    }
    summary_path = args.output / "summary.json"
    atomic(summary_path, result)
    atomic(args.output / "terminal.json", {
        "schema": f"freealign_official_box_same_pool_{args.mode}_terminal_v1",
        "status": "complete", "summary": str(summary_path.relative_to(ROOT)),
        "summary_sha256": sha256(summary_path), "task_id": contract["task_id"],
        "owner": contract["owner"], "scientific_task_closed": args.mode == "full",
    })
    print(json.dumps({"status": result["status"], "roles": {
        role: {"conditions": value["conditions"], "status_counts": value["status_counts"],
               "official_call_seconds": value["official_call_seconds"]}
        for role, value in summaries.items()}, "wall_seconds": result["cost"]["wall_seconds"]},
        sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
