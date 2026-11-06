#!/usr/bin/env python3
"""Materialize exact 39D V2V4Real inputs for the current shared-offset readout.

This is a zero-forward adapter.  It preserves every box, score, hypothesis,
label, IoU and source identity from the reviewed V2V4Real shared-offset pools.
Only their documented zero-width feature placeholder is replaced.  The first
30 fields are reconstructed from the earlier native 31D pool (whose last field
was only ``is_remote``); the final nine motion fields are reconstructed from
the reviewed legal-history inputs and their frozen raw source records.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

from diagnose_v2v_legal_history_geometry_v1_20261023 import (
    associate, common_mask, local_artifact, sensor_manifests,
)
from e1_receiver_value_core_v1_20261020 import (
    atomic, load_npz, read_rows, sha256,
)
from fit_evaluate_candidate_d_heads_v1_20261019 import feature_matrix
from object_history_selector_core_v1_20261021 import FEATURE_NAMES, PackWriter


class Packed:
    def __init__(self, directory: Path):
        self.directory = directory
        self.meta = list(read_rows(directory / "conditions.jsonl"))
        self.data = load_npz(directory / "pools.npz")

    def pool(self, index: int) -> dict[str, np.ndarray]:
        row = self.meta[index]
        a, b = int(row["start"]), int(row["stop"])
        ga, gb = int(row["gt_start"]), int(row["gt_stop"])
        out = {
            key: value[a:b]
            for key, value in self.data.items()
            if key not in {"gt_boxes", "gt_original_indices", "ious"}
        }
        out["gt_boxes"] = self.data["gt_boxes"][ga:gb]
        out["gt_original_indices"] = self.data["gt_original_indices"][ga:gb]
        out["ious"] = self.data["ious"][int(row["iou_start"]):int(row["iou_stop"])].reshape(
            b - a, gb - ga
        )
        return out


def checked(directory: Path) -> tuple[Packed, dict[str, Any]]:
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert sha256(directory / "conditions.jsonl") == summary["conditions_sha256"]
    assert sha256(directory / "pools.npz") == summary["pools_sha256"]
    return Packed(directory), summary


def motion_for_condition(
    root: Path,
    history_meta: dict[str, Any],
    manifest: dict[int, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    current_manifest = manifest[int(history_meta["source_task_index"])]
    current_path = local_artifact(root, history_meta["current_raw_path"])
    assert sha256(current_path) == history_meta["current_raw_sha256"]
    current = load_npz(current_path)
    source_pose = np.asarray(current_manifest["source_pose_lidar_to_world"], np.float64)
    previous = None
    previous_pose = None
    dt_s = 0.0
    if history_meta["previous_task_index"] is not None:
        previous_manifest = manifest[int(history_meta["previous_task_index"])]
        previous_path = local_artifact(root, history_meta["previous_raw_path"])
        assert sha256(previous_path) == history_meta["previous_raw_sha256"]
        previous = load_npz(previous_path)
        previous_pose = np.asarray(previous_manifest["source_pose_lidar_to_world"], np.float64)
        dt_s = 0.1
    assert float(history_meta["history_dt_s"]) == dt_s
    horizon_s = float(history_meta["effective_age_ms"]) / 1000.0
    assert float(history_meta["horizon_s"]) == horizon_s
    motion = associate(current, previous, source_pose, previous_pose, dt_s, horizon_s)
    return current, motion, source_pose, np.asarray(previous_pose) if previous_pose is not None else np.empty((0, 0))


def extras(motion: dict[str, np.ndarray], source_ids: np.ndarray,
           horizon_s: float, dt_s: float) -> np.ndarray:
    confirmed = motion["confirmed"][source_ids]
    out = np.zeros((len(source_ids), 9), np.float64)
    out[:, 2] = confirmed
    out[:, 3] = horizon_s
    if np.any(confirmed):
        ids = np.flatnonzero(confirmed)
        chosen = source_ids[ids]
        out[ids, 4] = dt_s
        out[ids, 5] = np.linalg.norm(motion["velocity"][chosen], axis=1)
        out[ids, 6] = motion["distance"][chosen]
        out[ids, 7] = motion["ratio"][chosen]
        out[ids, 8] = np.linalg.norm(motion["shift"][chosen], axis=1)
    return out


def build_split(
    root: Path,
    destination: Path,
    source_dir: Path,
    adapted_dir: Path,
    history_dir: Path,
    manifest: dict[int, dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    source, source_summary = checked(source_dir)
    adapted, adapted_summary = checked(adapted_dir)
    history, history_summary = checked(history_dir)
    assert len(source.meta) == len(adapted.meta) == len(history.meta)
    assert source.data["features"].shape[1] == 0
    assert adapted.data["features"].shape[1] == 31
    assert history.data["association_features"].shape[1] == 17
    writer = PackWriter(destination)
    type_counts: Counter[int] = Counter()
    role_wait_counts: Counter[str] = Counter()
    maxima = Counter()
    confirmed_stale = propagated = 0
    for index, (source_meta, adapted_meta, history_meta) in enumerate(
        zip(source.meta, adapted.meta, history.meta)
    ):
        for key in ("sequence", "global_index", "wait_ms", "role", "effective_age_ms"):
            assert source_meta[key] == adapted_meta[key] == history_meta[key]
        assert int(source_meta["condition_index"]) == int(adapted_meta["condition_index"]) == index
        assert int(history_meta["source_condition_index"]) == index
        source_pool = source.pool(index)
        adapted_pool = adapted.pool(index)
        hs, he = int(history_meta["start"]), int(history_meta["stop"])
        history_pool = {key: value[hs:he] for key, value in history.data.items()
                        if key not in {"gt_boxes", "stale_ious", "prop_ious"}}

        source_kind = source_pool["hypothesis_type"].astype(np.int8)
        ego = np.flatnonzero(source_kind == 0)
        stale = np.flatnonzero(source_kind == 1)
        prop = np.flatnonzero(source_kind == 2)
        adapted_ego = np.flatnonzero(adapted_pool["is_remote"] == 0)
        adapted_remote = np.flatnonzero(adapted_pool["is_remote"] == 1)
        assert len(ego) == len(adapted_ego)
        assert len(stale) == len(adapted_remote) == he - hs
        assert np.array_equal(source_pool["source_detection_indices"][ego],
                              adapted_pool["source_detection_indices"][adapted_ego])
        assert np.array_equal(source_pool["source_detection_indices"][stale],
                              adapted_pool["source_detection_indices"][adapted_remote])
        assert np.array_equal(source_pool["source_detection_indices"][stale],
                              history_pool["source_detection_indices"])
        maxima["ego_box"] = max(maxima["ego_box"], float(np.max(np.abs(
            source_pool["boxes"][ego] - adapted_pool["boxes"][adapted_ego]), initial=0.0)))
        maxima["stale_box"] = max(maxima["stale_box"], float(np.max(np.abs(
            source_pool["boxes"][stale] - adapted_pool["boxes"][adapted_remote]), initial=0.0)))
        maxima["history_stale_box"] = max(maxima["history_stale_box"], float(np.max(np.abs(
            source_pool["boxes"][stale] - history_pool["stale_boxes"]), initial=0.0)))
        assert np.array_equal(source_pool["raw_scores"][ego], adapted_pool["raw_scores"][adapted_ego])
        assert np.array_equal(source_pool["raw_scores"][stale], adapted_pool["raw_scores"][adapted_remote])
        assert np.array_equal(source_pool["raw_scores"][stale], history_pool["scores"])

        current, motion, source_pose, _ = motion_for_condition(root, history_meta, manifest)
        source_ids = source_pool["source_detection_indices"][stale].astype(np.int32)
        horizon_s = float(source_meta["effective_age_ms"]) / 1000.0
        dt_s = float(history_meta["history_dt_s"])
        receiver_pose = np.asarray(adapted_meta["receiver_ego_pose"], np.float64)
        assert np.max(np.abs(source_pose - np.asarray(adapted_meta["source_pose"], np.float64)), initial=0.0) == 0.0
        transform = np.linalg.solve(receiver_pose, source_pose)
        maxima["source_to_receiver"] = max(maxima["source_to_receiver"], float(np.max(np.abs(
            transform - np.asarray(adapted_meta["source_to_receiver"], np.float64)), initial=0.0)))

        propagated_valid = motion["confirmed"][source_ids] & common_mask(history_pool["prop_boxes"])
        assert np.array_equal(propagated_valid, history_pool["prop_valid"])
        valid_source_ids = source_ids[propagated_valid]
        assert len(prop) == int(propagated_valid.sum())
        assert np.array_equal(source_pool["source_detection_indices"][prop], valid_source_ids)
        maxima["prop_box"] = max(maxima["prop_box"], float(np.max(np.abs(
            source_pool["boxes"][prop] - history_pool["prop_boxes"][propagated_valid]), initial=0.0)))
        assert np.array_equal(source_pool["raw_scores"][prop],
                              history_pool["scores"][propagated_valid])

        feature = np.empty((len(source_kind), 39), np.float32)
        ego_extra = np.zeros((len(ego), 9), np.float64); ego_extra[:, 0] = 1.0
        feature[ego] = np.column_stack((adapted_pool["features"][adapted_ego, :30], ego_extra))
        stale_extra = extras(motion, source_ids, horizon_s, dt_s)
        feature[stale] = np.column_stack((adapted_pool["features"][adapted_remote, :30], stale_extra))
        protected_ego = ego[source_pool["is_protected"][ego].astype(bool)]
        prop_base = feature_matrix(
            source_pool["boxes"][prop], current["pred_boxes"][valid_source_ids],
            source_pool["raw_scores"][prop], float(source_meta["effective_age_ms"]),
            transform, source_pool["boxes"][protected_ego],
            source_pool["raw_scores"][protected_ego],
        )
        prop_extra = stale_extra[propagated_valid].copy(); prop_extra[:, 1] = 1.0
        feature[prop] = np.column_stack((prop_base, prop_extra))
        assert np.isfinite(feature).all()
        assert np.array_equal(feature[:, 30] > .5, source_kind == 0)
        assert np.array_equal(feature[:, 31] > .5, source_kind == 2)
        assert np.array_equal(feature[stale, 32] > .5, motion["confirmed"][source_ids])
        assert np.array_equal(feature[prop, 32] > .5, np.ones(len(prop), bool))

        value = {key: array.copy() for key, array in source_pool.items()}
        value["features"] = feature
        value["diagnostic"] = {
            "ego": int(len(ego)), "stale": int(len(stale)),
            "propagated": int(len(prop)),
            "causal_associated_stale": int(motion["confirmed"][source_ids].sum()),
            "formal_shared_offset_geometry_preserved": True,
        }
        receiver_timestamp_us = int(source_meta["timestamp"]) * 100000
        assert receiver_timestamp_us == int(source_meta["receiver_timestamp_us"])
        metadata = {
            key: value for key, value in source_meta.items()
            if key not in {"condition_index", "start", "stop", "gt_start", "gt_stop",
                           "iou_start", "iou_stop", "protected_ids", "variable_ids", "diagnostic"}
        }
        metadata.update(
            current_ego_pose_lidar_to_world=receiver_pose.tolist(),
            source={
                "source_frame": str(source_meta["source_timestamp"]),
                "source_timestamp_us": int(source_meta["source_timestamp"]) * 100000,
                "source_pose_lidar_to_world": source_pose.tolist(),
                "source_task_index": int(history_meta["source_task_index"]),
            },
            feature_identity="exact39D_current_shared_offset_actual_readout_v1",
        )
        writer.add(metadata, value)
        type_counts.update(map(int, source_kind))
        role_wait_counts[f"{source_meta['role']}|{int(source_meta['wait_ms'])}"] += 1
        confirmed_stale += int(motion["confirmed"][source_ids].sum())
        propagated += len(prop)
        if (index + 1) % 1000 == 0:
            print(json.dumps({"stage": f"materialize_{split}", "completed": index + 1,
                              "total": len(source.meta)}), flush=True)

    assert all(value == 0.0 for value in maxima.values()), maxima
    summary = writer.finish({
        "schema": "v2v4real_shared_offset_39d_actual_readout_inputs_v1",
        "split": split,
        "source_geometry_schema": source_summary["schema"],
        "feature_names": list(FEATURE_NAMES),
        "feature_dimensions": 39,
        "hypothesis_type_counts": {str(key): int(value) for key, value in sorted(type_counts.items())},
        "causal_associated_stale": confirmed_stale,
        "propagated_hypotheses": propagated,
        "role_wait_counts": dict(sorted(role_wait_counts.items())),
        "identity_max_abs_errors": dict(maxima),
        "input_sha256": {
            "source_conditions": source_summary["conditions_sha256"],
            "source_pools": source_summary["pools_sha256"],
            "adapted_conditions": adapted_summary["conditions_sha256"],
            "adapted_pools": adapted_summary["pools_sha256"],
            "history_conditions": history_summary["conditions_sha256"],
            "history_pools": history_summary["pools_sha256"],
        },
        "geometry_and_target_policy": "bit-preserve reviewed shared-offset geometry/targets; replace only zero-width feature placeholder",
        "receiver_timestamp_rule": "formal V2V4Real 10Hz frame proxy int(timestamp)*100000",
        "new_detector_or_backbone_forwards": 0,
        "detector_updates": 0,
        "head_updates": 0,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-inputs", required=True, type=Path)
    parser.add_argument("--adapted-train", required=True, type=Path)
    parser.add_argument("--adapted-val", required=True, type=Path)
    parser.add_argument("--history-train", required=True, type=Path)
    parser.add_argument("--history-val", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); root = args.workspace.resolve()
    assert not args.output.exists(); args.output.mkdir(parents=True)
    started = time.monotonic(); manifest = sensor_manifests(root)
    train = build_split(root, args.output / "train_dataset", args.source_inputs / "train_dataset",
                        args.adapted_train, args.history_train, manifest, "train")
    val = build_split(root, args.output / "val_dataset", args.source_inputs / "val_dataset",
                      args.adapted_val, args.history_val, manifest, "val")
    summary = {
        "schema": "v2v4real_shared_offset_39d_actual_readout_materialization_v1",
        "status": "complete_review_pending",
        "task_id": "shared_offset_v2v4real_actual_39d_readout_20261027",
        "owner": "research_autopilot_shared_offset_v2v4real_actual_39d_readout_v1",
        "splits": {"train": train, "val": val},
        "script_sha256": sha256(Path(__file__)),
        "wall_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
        "detector_or_backbone_forwards": 0, "head_updates": 0, "gpu_seconds": 0,
        "cost_boundary": "CPU cached input materialization only; not system inference latency",
    }
    atomic(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "wall_seconds": summary["wall_seconds"],
                      "train_candidates": train["candidates"], "val_candidates": val["candidates"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
