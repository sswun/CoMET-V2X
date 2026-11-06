#!/usr/bin/env python3
"""Actual V2XBPKT1 budgets for the fixed60 shared-KF + fresh-39D system.

Every source message is encoded and decoded with the existing wire codec.
Receiver history contains only messages that were delivered at an earlier
receiver decision in the same (role, sequence, wait, sender policy, budget)
stream.  The causal shared-KF is reconstructed from those decoded records;
full-cache boxes are never corrected and then truncated.

The all-calibration-disabled arm retains exactly the same decoded detection
candidates and history geometry as CV, but withholds every legal co-visible
anchor from the shared offset state.  It is the paired fit/cal diagnostic and
formal indirect-calibration-value control, not a different sender packet.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "preexperiment/research_baselines"
sys.path.insert(0, str(BASELINES))

from audit_common_offset_peer_v1_20261026 import pairs
from e1_receiver_value_core_v1_20261020 import common_mask, greedy, iou32, unique_ego_scores
from object_history_selector_core_v1_20261021 import PackedDataset, make_condition
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap
import run_shared_offset_optional_formal_v1_20261026 as optional
from v2xseq_real_motion_packets import encode_packet as encode_wire
import v2xvit_sender_packet_budget_v2_20261026 as packet_codec


TASK_ROOT = ROOT / "artifacts/research_autopilot/status_audit_20261027_2130/evening_successors_v1/shared_correction_actual_packet_budget_20261027_evening"
CONTRACT = TASK_ROOT / "contract_v1.json"
DATA_ROOT = ROOT / "artifacts/research_autopilot/status_audit_20261026_2155/traf_strict60_shared_offset_successor_v1/timestamp_adapter_recovery_v1"
TRAIN_DATASET = DATA_ROOT / "train_dataset"
VAL_DATASET = DATA_ROOT / "val_dataset"
CACHE = ROOT / "artifacts/research_autopilot/status_audit_20261026_1235/traf_strict60_current_method_successor_v1/scientific_run_recovery_v4/stages/cache_v1"
READOUT = ROOT / "artifacts/research_autopilot/status_audit_20261026_2155/traf_strict60_shared_offset_successor_v1/scientific_run_recovery_v2/stages/readout_v1"
CHECKPOINT_SHA256 = "d6811946bda673b6adbf00640c8e6bee09b9ffe9308aa77fc072c7828ecc190d"
ROLES = ("fit24", "calibration8", "mechanism_holdout14", "val21")
FORMAL_ROLES = ("mechanism_holdout14", "val21")
WAITS = (0, 400)
BUDGETS: tuple[int | None, ...] = (1024, 2048, 4096, None)
POLICIES = ("score_priority", "spatial_coverage_10m")
METHODS = (
    "ego_raw", "stale_raw", "stale_direct_u", "cv_direct_u",
    "kf_no_calibration_direct_u", "kf_raw", "kf_direct_u",
)
GEOMETRY_FOR_METHOD = {
    "ego_raw": "stale", "stale_raw": "stale", "stale_direct_u": "stale",
    "cv_direct_u": "cv", "kf_no_calibration_direct_u": "cv",
    "kf_raw": "kf", "kf_direct_u": "kf",
}


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def evidence(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def append(stream: Any, value: Any) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def budget_name(value: int | None) -> str:
    return "unbounded" if value is None else str(value)


class FrozenCache:
    def __init__(self) -> None:
        summary = json.loads((CACHE / "summary.json").read_text())
        if summary["status"] != "passed" or summary["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise RuntimeError("fixed60 cache identity failed")
        if not all(summary["checks"].values()):
            raise RuntimeError("fixed60 cache review checks failed")
        self.summary = summary
        with np.load(CACHE / "predictions_and_gt.npz", allow_pickle=False) as archive:
            self.data = {key: archive[key].copy() for key in archive.files}
        self.manifest = list(rows(CACHE / "manifest.jsonl"))
        if len(self.manifest) != 21304:
            raise RuntimeError("fixed60 manifest population changed")
        self.by_task = {int(row["task_index"]): row for row in self.manifest}
        self.sources: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in self.manifest:
            if row["sensor_role"] != "source":
                continue
            audit = row["input_audit"]
            key = (str(row["split"]), str(row["sequence"]), str(row["frame"]))
            raw = self.raw(int(row["task_index"]))
            self.sources[key] = {
                "sequence": str(row["sequence"]), "frame": str(row["frame"]),
                "timestamp_us": int(audit["source_timestamp_us"]),
                "pose": np.asarray(audit["source_pose_lidar_to_world"], np.float64),
                "raw": {name: raw[name] for name in ("pred_boxes", "pred_scores", "pred_labels")},
                "task_index": int(row["task_index"]),
                "processed_tensor_sha256": row["processed_tensor_sha256"],
                "input_identity_policy": row["input_identity_policy"],
            }
        if len(self.sources) != 10543:
            raise RuntimeError("fixed60 source population changed")

    def raw(self, task_index: int) -> dict[str, np.ndarray]:
        pa, pb = self.data["prediction_offsets"][task_index:task_index + 2].astype(int)
        ga, gb = self.data["gt_offsets"][task_index:task_index + 2].astype(int)
        return {
            "pred_boxes": self.data["pred_boxes"][pa:pb].copy(),
            "pred_scores": self.data["pred_scores"][pa:pb].copy(),
            "pred_labels": self.data["pred_labels"][pa:pb].copy(),
            "gt_boxes": self.data["gt_boxes"][ga:gb].copy(),
            "gt_labels": self.data["gt_labels"][ga:gb].copy(),
        }

    def ego(self, task_index: int) -> dict[str, np.ndarray]:
        row = self.by_task[task_index]
        if row["sensor_role"] != "ego" or row["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise RuntimeError("ego cache identity failed")
        return self.raw(task_index)

    def source(self, split: str, sequence: str, frame: str) -> dict[str, Any]:
        return self.sources[(split, sequence, frame)]


def score_order(source: dict[str, Any]) -> np.ndarray:
    raw = source["raw"]
    return packet_codec.sender_order(
        raw["pred_boxes"], raw["pred_scores"], raw["pred_labels"],
        np.arange(len(raw["pred_boxes"]), dtype=np.uint32),
    )


def spatial_order(source: dict[str, Any]) -> np.ndarray:
    """One highest-score record per sender-local 10m cell, then score refill."""
    ordered = score_order(source)
    boxes = source["raw"]["pred_boxes"]
    seen: set[tuple[int, int]] = set()
    coverage: list[int] = []
    refill: list[int] = []
    for index_value in ordered:
        index = int(index_value)
        cell = (math.floor(float(boxes[index, 0]) / 10.0),
                math.floor(float(boxes[index, 1]) / 10.0))
        if cell not in seen:
            coverage.append(index); seen.add(cell)
        else:
            refill.append(index)
    return np.asarray(coverage + refill, np.int64)


class PacketStore:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.binary_path = output / "packets.bin"
        self.index_path = output / "packet_index.jsonl"
        self.binary = self.binary_path.open("wb")
        self.index = self.index_path.open("w", encoding="utf-8")
        self.cache: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        self.offset = 0
        self.packet_count = 0

    def get(self, split: str, source: dict[str, Any], policy: str,
            budget: int | None) -> dict[str, Any]:
        key = (split, source["sequence"], source["frame"], policy, budget_name(budget))
        if key in self.cache:
            return self.cache[key]
        raw = source["raw"]
        original_order = score_order(source) if policy == "score_priority" else spatial_order(source)
        if policy == "score_priority":
            payload = packet_codec.encode_budgeted_packet(
                sequence=source["sequence"], source_frame=source["frame"],
                source_timestamp_us=source["timestamp_us"], source_pose=source["pose"],
                boxes=raw["pred_boxes"], scores=raw["pred_scores"], labels=raw["pred_labels"],
                budget_bytes=budget, original_indices=np.arange(len(raw["pred_boxes"])),
            )
        else:
            empty = encode_wire(
                sequence=source["sequence"], source_frame=source["frame"],
                source_timestamp_us=source["timestamp_us"], source_pose=source["pose"],
                boxes=np.empty((0, 7), np.float32), scores=np.empty(0, np.float32),
                labels=np.empty(0, np.int32),
            )
            if budget is not None and len(empty) > budget:
                payload = b""; count = 0
            else:
                count = len(original_order) if budget is None else min(
                    len(original_order), (int(budget) - len(empty)) // packet_codec.LOGICAL_RECORD_BYTES)
                # Spatial coverage is a truncation policy, not an alternative
                # wire-order treatment.  When every record fits, serialize in
                # the exact score-priority order so the two policies are input
                # equivalent (including stable downstream tie behavior).
                if count == len(original_order):
                    original_order = score_order(source)
                chosen = original_order[:count]
                payload = encode_wire(
                    sequence=source["sequence"], source_frame=source["frame"],
                    source_timestamp_us=source["timestamp_us"], source_pose=source["pose"],
                    boxes=raw["pred_boxes"][chosen], scores=raw["pred_scores"][chosen],
                    labels=raw["pred_labels"][chosen],
                )
        decoded = packet_codec.decode_budgeted_packet(payload)
        count = 0 if decoded is None else len(decoded["boxes"])
        selected = original_order[:count].astype(np.int32)
        if decoded is not None:
            if not (np.array_equal(decoded["boxes"], raw["pred_boxes"][selected])
                    and np.array_equal(decoded["scores"], raw["pred_scores"][selected])
                    and np.array_equal(decoded["labels"], raw["pred_labels"][selected])):
                raise RuntimeError("wire roundtrip changed selected detector records")
        if budget is not None and len(payload) > budget:
            raise RuntimeError("actual payload exceeded budget")
        digest = hashlib.sha256(payload).hexdigest()
        start = self.offset
        self.binary.write(payload); self.offset += len(payload)
        metadata_bytes = 0 if decoded is None else len(payload) - packet_codec.LOGICAL_RECORD_BYTES * count
        record = {
            "split": split, "sequence": source["sequence"], "source_frame": source["frame"],
            "source_task_index": source["task_index"], "checkpoint_sha256": CHECKPOINT_SHA256,
            "policy": policy, "budget_bytes": budget, "packet_offset": start,
            "packet_bytes": len(payload), "metadata_and_prefix_bytes": metadata_bytes,
            "records": count, "payload_sha256": digest,
            "selected_original_detector_rows": selected.tolist(),
            "source_timestamp_us": source["timestamp_us"],
        }
        append(self.index, record)
        receiver = None
        if decoded is not None:
            receiver = {
                "sequence": decoded["sequence"], "frame": decoded["source_frame"],
                "timestamp_us": int(decoded["source_timestamp_us"]),
                "pose": decoded["source_pose_lidar_to_world"],
                "raw": {"pred_boxes": decoded["boxes"], "pred_scores": decoded["scores"],
                        "pred_labels": decoded["labels"]},
                "payload_sha256": digest, "packet_bytes": len(payload),
                "metadata_and_prefix_bytes": metadata_bytes,
                "selected_original_detector_rows": selected,
            }
        value = {**record, "decoded": receiver}
        self.cache[key] = value; self.packet_count += 1
        return value

    def close(self) -> dict[str, Any]:
        self.binary.flush(); os.fsync(self.binary.fileno()); self.binary.close()
        self.index.flush(); os.fsync(self.index.fileno()); self.index.close()
        return {
            "unique_encoded_packets": self.packet_count,
            "packets": evidence(self.binary_path), "index": evidence(self.index_path),
        }


def decode_source(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {key: value[key] for key in ("sequence", "frame", "timestamp_us", "pose", "raw")}


def materialize(dataset: PackedDataset, cache: FrozenCache, store: PacketStore,
                role: str, policy: str, budget: int | None, output: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    split = "val" if role == "val21" else "train"
    selected = [index for index, meta in enumerate(dataset.meta)
                if meta["role"] == role and int(meta["wait_ms"]) in WAITS]
    selected.sort(key=lambda index: (str(dataset.meta[index]["sequence"]),
                                     int(dataset.meta[index]["wait_ms"]),
                                     int(dataset.meta[index]["receiver_timestamp_us"])))
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    condition_offsets = [0]; gt_offsets = [0]
    condition_index: list[int] = []; waits: list[int] = []; sequences: list[str] = []
    receiver_times: list[int] = []; packet_bytes: list[int] = []; new_wire_bytes: list[int] = []
    metadata_bytes: list[int] = []; history_bytes: list[int] = []; packet_records: list[int] = []
    packet_sha: list[str] = []; history_sha: list[str] = []; anchor_counts: list[int] = []
    state: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    delivered: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    newly_delivered: set[tuple[str, int, str]] = set()
    ledger_path = output / f"{role}__{policy}__{budget_name(budget)}__condition_ledger.jsonl"
    ledger = ledger_path.open("w", encoding="utf-8")
    started = time.monotonic()
    for ordinal, index in enumerate(selected, 1):
        meta = dataset.meta[index]
        sequence = str(meta["sequence"]); wait = int(meta["wait_ms"])
        stream_key = (sequence, wait); source_meta = meta.get("source")
        entry = None; current = None; history = None; is_new = False
        if source_meta is not None:
            source = cache.source(split, sequence, str(source_meta["source_frame"]))
            if int(source_meta["source_timestamp_us"]) != source["timestamp_us"]:
                raise RuntimeError("condition/cache source timestamp mismatch")
            entry = store.get(split, source, policy, budget)
            current = entry["decoded"]
            if current is not None:
                earlier = [row for row in delivered[stream_key].values()
                           if int(row["timestamp_us"]) < int(current["timestamp_us"])]
                history = max(earlier, key=lambda row: int(row["timestamp_us"])) if earlier else None
                delivery_key = (sequence, wait, current["frame"])
                if delivery_key not in newly_delivered:
                    delivered[stream_key][current["frame"]] = current
                    newly_delivered.add(delivery_key); is_new = True
        receiver_timestamp = int(meta["receiver_timestamp_us"])
        if current is not None:
            age_ms = (receiver_timestamp - int(current["timestamp_us"])) / 1000.0
            if abs(age_ms - float(source_meta["actual_age_ms"])) > .0011 or age_ms < 0:
                raise RuntimeError("decoded packet violates recorded arrival")
        else:
            age_ms = 0.0
        ego = cache.ego(int(meta["ego_task_index"]))
        pool = make_condition(
            ego, decode_source(current), decode_source(history),
            np.asarray(meta["current_ego_pose_lidar_to_world"], np.float64),
            receiver_timestamp, age_ms, 0.3,
        )
        kind = pool["hypothesis_type"]
        ego_ids = np.flatnonzero(kind == 0); stale_ids = np.flatnonzero(kind == 1)
        prop_lookup = {int(pool["source_detection_indices"][candidate]): int(candidate)
                       for candidate in np.flatnonzero(kind == 2)}
        cv_ids = np.asarray([prop_lookup.get(int(pool["source_detection_indices"][candidate]), int(candidate))
                             for candidate in stale_ids], np.int64)
        stale_boxes = np.concatenate((pool["boxes"][ego_ids], pool["boxes"][stale_ids])).astype(np.float32)
        cv_boxes = np.concatenate((pool["boxes"][ego_ids], pool["boxes"][cv_ids])).astype(np.float32)
        raw = np.r_[pool["raw_scores"][ego_ids], pool["raw_scores"][stale_ids]].astype(np.float64)
        protected = np.r_[pool["is_protected"][ego_ids], np.zeros(len(stale_ids), np.int8)]
        source_flag = np.r_[np.zeros(len(ego_ids), np.int8), np.ones(len(stale_ids), np.int8)]
        stale_features = np.concatenate((pool["features"][ego_ids], pool["features"][stale_ids])).astype(np.float32)
        cv_features = np.concatenate((pool["features"][ego_ids], pool["features"][cv_ids])).astype(np.float32)
        protected_local = np.flatnonzero(protected[:len(ego_ids)] == 1)
        protected_boxes = cv_boxes[protected_local]
        protected_scores = raw[protected_local]
        remote_cv = cv_boxes[len(ego_ids):]
        anchor_remote, anchor_ego = pairs(remote_cv, protected_boxes, 2.5, True)
        residual = protected_boxes[anchor_ego, :2].astype(np.float64) - remote_cv[anchor_remote, :2]
        mean, variance = state.get(stream_key, (np.zeros(2, np.float64), 1.0))
        prior = float(variance) + .15 ** 2
        if len(residual):
            gain = prior / (prior + .6 ** 2 / len(residual))
            mean = mean + gain * (residual.mean(axis=0) - mean)
            variance = (1.0 - gain) * prior
        else:
            variance = prior
        state[stream_key] = (mean.copy(), float(variance))
        kf_boxes = cv_boxes.copy(); kf_boxes[len(ego_ids):, :2] += mean.astype(np.float32)
        kf_features = cv_features.copy()
        kf_features[len(ego_ids):] = optional.update_features(
            cv_features[len(ego_ids):], kf_boxes[len(ego_ids):],
            protected_boxes, protected_scores,
        )
        arrays["stale_boxes"].append(stale_boxes); arrays["cv_boxes"].append(cv_boxes)
        arrays["kf_boxes"].append(kf_boxes); arrays["stale_features"].append(stale_features)
        arrays["cv_features"].append(cv_features); arrays["kf_features"].append(kf_features)
        arrays["raw_scores"].append(raw); arrays["source"].append(source_flag)
        arrays["protected"].append(protected); arrays["gt_boxes"].append(pool["gt_boxes"])
        if current is None:
            original_ids = np.full(len(stale_ids), -1, np.int32)
        else:
            original_ids = current["selected_original_detector_rows"][pool["source_detection_indices"][stale_ids]]
        arrays["sender_original_id"].append(np.r_[np.full(len(ego_ids), -1, np.int32), original_ids])
        condition_offsets.append(condition_offsets[-1] + len(raw)); gt_offsets.append(gt_offsets[-1] + len(pool["gt_boxes"]))
        condition_index.append(index); waits.append(wait); sequences.append(sequence); receiver_times.append(receiver_timestamp)
        size = 0 if entry is None else int(entry["packet_bytes"])
        packet_bytes.append(size); new_wire_bytes.append(size if is_new else 0)
        metadata_bytes.append(0 if entry is None else int(entry["metadata_and_prefix_bytes"]))
        history_bytes.append(0 if history is None else int(history["packet_bytes"]))
        packet_records.append(0 if entry is None else int(entry["records"]))
        packet_sha.append("" if entry is None else str(entry["payload_sha256"]))
        history_sha.append("" if history is None else str(history["payload_sha256"]))
        anchor_counts.append(len(residual))
        append(ledger, {
            "condition_ordinal": ordinal - 1, "dataset_index": index, "sequence": sequence,
            "wait_ms": wait, "receiver_timestamp_us": receiver_timestamp,
            "source_frame": None if source_meta is None else source_meta["source_frame"],
            "packet_sha256": packet_sha[-1], "packet_bytes": size,
            "new_wire_bytes": new_wire_bytes[-1], "decoded_records": packet_records[-1],
            "history_packet_sha256": history_sha[-1], "history_cached_bytes": history_bytes[-1],
            "history_retransmitted_bytes": 0, "feedback_bytes": 0,
            "anchors_used": len(residual), "kf_dx": float(mean[0]), "kf_dy": float(mean[1]),
            "same_output_candidates_in_no_calibration_arm": True,
        })
        if ordinal % 1000 == 0:
            print(json.dumps({"stage": "actual_packet_materialize", "role": role,
                              "policy": policy, "budget": budget_name(budget),
                              "completed": ordinal, "total": len(selected)}), flush=True)
    ledger.flush(); os.fsync(ledger.fileno()); ledger.close()
    material = {key: np.concatenate(value) for key, value in arrays.items()}
    material.update(
        condition_offsets=np.asarray(condition_offsets, np.int64),
        gt_offsets=np.asarray(gt_offsets, np.int64), dataset_index=np.asarray(condition_index, np.int64),
        wait_ms=np.asarray(waits, np.int16), sequence=np.asarray(sequences, dtype="U16"),
        receiver_timestamp_us=np.asarray(receiver_times, np.int64),
        packet_bytes=np.asarray(packet_bytes, np.int32), new_wire_bytes=np.asarray(new_wire_bytes, np.int32),
        metadata_and_prefix_bytes=np.asarray(metadata_bytes, np.int32),
        history_cached_bytes=np.asarray(history_bytes, np.int32), packet_records=np.asarray(packet_records, np.int16),
        packet_sha256=np.asarray(packet_sha, dtype="U64"), history_packet_sha256=np.asarray(history_sha, dtype="U64"),
        anchor_count=np.asarray(anchor_counts, np.int16),
    )
    return material, {"conditions": len(selected), "candidates": len(material["raw_scores"]),
                      "ledger": evidence(ledger_path), "wall_seconds": time.monotonic() - started}


def score_and_evaluate(material: dict[str, np.ndarray], role: str, policy: str,
                       budget: int | None, output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    direct_raw = optional.direct_scores({name: material[f"{name}_features"]
                                         for name in ("stale", "cv", "kf")}, READOUT)
    maps = json.loads((READOUT / "source_maps.json").read_text())
    direct = {name: optional.source_map(value, material["source"], maps["maps"]["direct_to_utility"])
              for name, value in direct_raw.items()}
    candidates_path = output / f"{role}__{policy}__{budget_name(budget)}__candidates.npz"
    np.savez_compressed(
        candidates_path,
        **{key: value for key, value in material.items() if not key.endswith("_features")},
        stale_direct_u=direct["stale"], cv_direct_u=direct["cv"], kf_direct_u=direct["kf"],
    )
    stats = defaultdict(lambda: {"score": [], "tp50": [], "tp70": [], "gt": 0,
                                 "outputs": 0, "remote": 0, "dropped_prediction_roi": 0,
                                 "dropped_gt_roi": 0})
    saved = {method: {"candidate": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
             for method in METHODS}
    for condition, wait_value in enumerate(material["wait_ms"]):
        a, b = material["condition_offsets"][condition:condition + 2].astype(int)
        ga, gb = material["gt_offsets"][condition:condition + 2].astype(int)
        raw = material["raw_scores"][a:b]; source = material["source"][a:b]
        protected = material["protected"][a:b]; gt_all = material["gt_boxes"][ga:gb]
        gt_keep = common_mask(gt_all); gt = gt_all[gt_keep]
        for method in METHODS:
            geometry = GEOMETRY_FOR_METHOD[method]
            boxes = material[f"{geometry}_boxes"][a:b]
            if method == "ego_raw":
                ids = np.flatnonzero(source == 0); rank = unique_ego_scores(raw[ids], stable=True)
                order = np.argsort(-rank, kind="stable"); ids, rank = ids[order], rank[order]
            else:
                score = raw if method in ("stale_raw", "kf_raw") else (
                    direct["stale"][a:b] if method == "stale_direct_u" else
                    direct["kf"][a:b] if method == "kf_direct_u" else direct["cv"][a:b])
                ids, rank = optional.ordered(raw, source, protected, score, None)
            keep = common_mask(boxes[ids]); dropped = int((~keep).sum())
            ids, rank = ids[keep], rank[keep]
            matrix = iou32(boxes[ids], gt)
            tp50 = (greedy(matrix, np.arange(len(ids)), .5) >= 0).astype(np.int8)
            tp70 = (greedy(matrix, np.arange(len(ids)), .7) >= 0).astype(np.int8)
            row = stats[(int(wait_value), method)]
            row["score"].append(rank); row["tp50"].append(tp50); row["tp70"].append(tp70)
            row["gt"] += len(gt); row["outputs"] += len(ids); row["remote"] += int((source[ids] == 1).sum())
            row["dropped_prediction_roi"] += dropped; row["dropped_gt_roi"] += int((~gt_keep).sum())
            record = saved[method]; record["candidate"].append((a + ids).astype(np.int64))
            record["score"].append(rank.astype(np.float64)); record["tp50"].append(tp50); record["tp70"].append(tp70)
            record["offset"].append(record["offset"][-1] + len(ids))
    panels: list[dict[str, Any]] = []
    for (wait, method), row in sorted(stats.items()):
        panel = {"role": role, "policy": policy, "budget_bytes": budget, "wait_ms": wait, "method": method,
                 **{key: int(value) for key, value in row.items() if not isinstance(value, list)}}
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"]); scores = np.concatenate(row["score"])
            metric = voc_ap(tp.tolist(), (1 - tp).tolist(), scores.tolist(), row["gt"], True)
            panel[f"ap{threshold}"] = 100.0 * float(metric["ap"])
            panel[f"tp{threshold}"] = int(metric["tp"]); panel[f"fp{threshold}"] = int(metric["fp"])
        panels.append(panel)
    prediction_path = output / f"{role}__{policy}__{budget_name(budget)}__predictions.npz"
    archive: dict[str, np.ndarray] = {"method_names": np.asarray(METHODS, dtype="U40")}
    for method, record in saved.items():
        for field in ("candidate", "score", "tp50", "tp70"):
            archive[f"{method}__{field}"] = np.concatenate(record[field])
        archive[f"{method}__offset"] = np.asarray(record["offset"], np.int64)
    np.savez_compressed(prediction_path, **archive)
    communication = []
    for wait in WAITS:
        use = material["wait_ms"] == wait
        communication.append({
            "role": role, "policy": policy, "budget_bytes": budget, "wait_ms": wait,
            "conditions": int(use.sum()), "logical_packet_bytes_sum": int(material["packet_bytes"][use].sum()),
            "actual_new_wire_bytes_sum": int(material["new_wire_bytes"][use].sum()),
            "metadata_and_prefix_bytes_sum": int(material["metadata_and_prefix_bytes"][use].sum()),
            "decoded_records_sum": int(material["packet_records"][use].sum()),
            "cached_history_bytes_referenced_sum": int(material["history_cached_bytes"][use].sum()),
            "history_retransmitted_bytes": 0, "feedback_bytes": 0,
            "history_hit_conditions": int((material["history_cached_bytes"][use] > 0).sum()),
            "no_packet_conditions": int((material["packet_bytes"][use] == 0).sum()),
            "mean_packet_bytes_per_condition": float(material["packet_bytes"][use].mean()),
            "mean_decoded_records_per_condition": float(material["packet_records"][use].mean()),
        })
    return panels, {"candidates": evidence(candidates_path), "predictions": evidence(prediction_path),
                    "communication": communication}


def comparisons(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key = lambda row: (row["role"], row["policy"], budget_name(row["budget_bytes"]), row["wait_ms"], row["method"])
    table = {key(row): row for row in panels}
    result = []
    pairs_to_report = (
        ("kf_direct_u", "kf_no_calibration_direct_u", "indirect_calibration_value_same_packet"),
        ("kf_direct_u", "cv_direct_u", "shared_kf_vs_delivered_history_cv"),
        ("cv_direct_u", "stale_direct_u", "delivered_history_cv_vs_stale"),
        ("kf_direct_u", "stale_raw", "full_cached_module_stack_vs_simple_same_packet"),
        ("stale_raw", "ego_raw", "direct_new_target_supply_simple_fusion"),
        ("kf_direct_u", "kf_raw", "readout_contribution_same_kf_geometry"),
    )
    for row_key, candidate in table.items():
        role, policy, budget, wait, method = row_key
        for candidate_name, reference_name, interpretation in pairs_to_report:
            if method != candidate_name:
                continue
            reference = table[(role, policy, budget, wait, reference_name)]
            result.append({
                "role": role, "policy": policy,
                "budget_bytes": None if budget == "unbounded" else int(budget), "wait_ms": wait,
                "candidate": candidate_name, "reference": reference_name, "interpretation": interpretation,
                "delta_ap50_pp": candidate["ap50"] - reference["ap50"],
                "delta_ap70_pp": candidate["ap70"] - reference["ap70"],
                "delta_tp70": candidate["tp70"] - reference["tp70"],
                "delta_fp70": candidate["fp70"] - reference["fp70"],
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True); started = time.monotonic()
    if "embodied-research.slice" not in Path("/proc/self/cgroup").read_text():
        raise RuntimeError("runner outside embodied-research.slice")
    if os.environ.get("EMBODIED_THERMAL_THROTTLE") != "1":
        raise RuntimeError("continuous pressure guard not active")
    contract = json.loads(CONTRACT.read_text())
    if contract["owner"] != "research_autopilot_actual_packet_kf_value_v1":
        raise RuntimeError("owner identity changed")
    policy_contract = {
        "schema": "strict60_actual_packet_sender_policy_contract_v1",
        "status": "frozen_before_fit_cal_or_formal_outcomes", "created_at": now(),
        "task_id": contract["task_id"], "owner": contract["owner"],
        "checkpoint_sha256": CHECKPOINT_SHA256, "budgets_bytes": [1024, 2048, 4096, None],
        "formal_roles": list(FORMAL_ROLES), "formal_waits_ms": list(WAITS),
        "sender_policies": {
            "score_priority": "score descending with the published deterministic sender-only tie order",
            "spatial_coverage_10m": (
                "sender-local floor(x/10m),floor(y/10m) cells; take the highest-score record "
                "from each occupied cell in descending cell-top-score order, then refill in score order; "
                "if every record fits, serialize in exact score-priority order"
            ),
        },
        "wire": packet_codec.layout_identity(),
        "history": "strictly earlier messages already decoded and delivered in the same role/sequence/wait/policy/budget stream",
        "paired_ablation": (
            "same decoded current packet, same delivered-history CV geometry and same output candidates; "
            "withhold every co-visible anchor from shared-KF state updates"
        ),
        "selection_forbidden_inputs": ["receiver detections", "ground truth", "future frames", "validation outcomes"],
        "effect_gate": None,
    }
    atomic(output / "frozen_policy_contract.json", policy_contract)
    cache = FrozenCache(); train = PackedDataset(TRAIN_DATASET); val = PackedDataset(VAL_DATASET)
    for dataset, expected in ((train, 29780), (val, 13264)):
        if len(dataset.meta) != expected:
            raise RuntimeError("packed dataset population changed")
    store = PacketStore(output)
    panels: list[dict[str, Any]] = []; run_inventory: dict[str, Any] = {}; communication: list[dict[str, Any]] = []
    configurations: list[tuple[str, str, int | None, str]] = [
        ("fit24", "score_priority", None, "development_fit_cal_ablation"),
        ("calibration8", "score_priority", None, "development_fit_cal_ablation"),
    ]
    configurations.extend((role, policy, budget, "formal") for role in FORMAL_ROLES
                          for policy in POLICIES for budget in BUDGETS)
    for role, policy, budget, kind in configurations:
        dataset = val if role == "val21" else train
        print(json.dumps({"stage": "configuration_start", "kind": kind, "role": role,
                          "policy": policy, "budget": budget_name(budget)}), flush=True)
        material, material_summary = materialize(dataset, cache, store, role, policy, budget, output)
        result_panels, inventory = score_and_evaluate(material, role, policy, budget, output)
        for row in result_panels:
            row["result_kind"] = kind
        panels.extend(result_panels); communication.extend(inventory.pop("communication"))
        run_inventory[f"{role}__{policy}__{budget_name(budget)}"] = {
            "kind": kind, "materialization": material_summary, **inventory,
        }
        del material
    packet_inventory = store.close()
    signed = comparisons(panels)
    formal_panels = [row for row in panels if row["result_kind"] == "formal"]
    development_panels = [row for row in panels if row["result_kind"] != "formal"]
    panel_table = {
        (row["role"], row["policy"], budget_name(row["budget_bytes"]), row["wait_ms"], row["method"]): row
        for row in panels
    }
    paired_cv_identity = []
    for role, policy, budget_value, wait, method in panel_table:
        if method != "cv_direct_u":
            continue
        cv = panel_table[(role, policy, budget_value, wait, "cv_direct_u")]
        no_cal = panel_table[(role, policy, budget_value, wait, "kf_no_calibration_direct_u")]
        paired_cv_identity.append(all(cv[field] == no_cal[field] for field in (
            "ap50", "ap70", "tp50", "fp50", "tp70", "fp70", "gt", "outputs", "remote"
        )))
    checks = {
        "fixed60_checkpoint_exact": cache.summary["checkpoint_sha256"] == CHECKPOINT_SHA256,
        "all_actual_payloads_saved": packet_inventory["unique_encoded_packets"] > 0,
        "formal_two_roles_two_waits_two_policies_four_budgets_seven_methods": len(formal_panels) == 2 * 2 * 2 * 4 * 7,
        "canonical_formal_gt_counts": all(row["gt"] == (22637 if row["role"] == "mechanism_holdout14" else 31123)
                                          for row in formal_panels),
        "same_packet_no_calibration_is_cv": bool(paired_cv_identity) and all(paired_cv_identity),
        "no_history_retransmission_or_feedback": all(row["history_retransmitted_bytes"] == 0
                                                       and row["feedback_bytes"] == 0 for row in communication),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {
        "schema": "strict60_actual_packet_kf_value_v1", "status": "complete_producer_review_pending",
        "created_at": now(), "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": evidence(CONTRACT), "policy_contract": evidence(output / "frozen_policy_contract.json"),
        "input_identity": {
            "checkpoint_sha256": CHECKPOINT_SHA256, "cache_summary": evidence(CACHE / "summary.json"),
            "cache_predictions": evidence(CACHE / "predictions_and_gt.npz"),
            "cache_manifest": evidence(CACHE / "manifest.jsonl"),
            "train_dataset_summary": evidence(TRAIN_DATASET / "summary.json"),
            "train_dataset_pools": evidence(TRAIN_DATASET / "pools.npz"),
            "train_dataset_conditions": evidence(TRAIN_DATASET / "conditions.jsonl"),
            "val_dataset_summary": evidence(VAL_DATASET / "summary.json"),
            "val_dataset_pools": evidence(VAL_DATASET / "pools.npz"),
            "val_dataset_conditions": evidence(VAL_DATASET / "conditions.jsonl"),
            "readout_weights": evidence(READOUT / "direct_u.pth"),
            "readout_schedule": evidence(READOUT / "training_schedule.npz"),
            "readout_source_maps": evidence(READOUT / "source_maps.json"),
            "wire_codec": evidence(BASELINES / "v2xseq_real_motion_packets.py"),
            "budget_wrapper": evidence(BASELINES / "v2xvit_sender_packet_budget_v2_20261026.py"),
        },
        "formal_panels": formal_panels, "development_fit_cal_panels": development_panels,
        "signed_comparisons": signed, "communication": communication,
        "packet_inventory": packet_inventory, "run_inventory": run_inventory, "checks": checks,
        "actual_cost": {
            "detector_or_backbone_forwards": 0, "detector_updates": 0, "head_updates": 0,
            "gpu_seconds": 0, "cpu_threads": 1, "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        },
        "cost_boundary": (
            "This run measures cached packet/KF/39D module execution. The fixed60 cache identity records "
            "the upstream 21,304 detector forwards; this CPU-only increment is not complete-system latency."
        ),
        "communication_boundary": (
            "logical_packet_bytes is the size of the source packet referenced by a condition; "
            "actual_new_wire_bytes counts it only on first delivery in that stream. Earlier packet bytes "
            "are reported as cached-history references and are never counted as current retransmission."
        ),
        "scientific_boundary": (
            "Formal rows are exposed strict60 hold14/val21 development roles at waits 0/400. "
            "All effect signs are retained; no hidden-test, all-domain, or full-system claim is made."
        ),
        "script": evidence(Path(__file__).resolve()),
    }
    atomic(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "formal_panels": len(formal_panels),
                      "signed_comparisons": len(signed), "summary": rel(output / "summary.json"),
                      "actual_cost": summary["actual_cost"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
