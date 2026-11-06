#!/usr/bin/env python3
"""Pre-registered native TraF-Align V2V4Real pilot without third-party edits."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | os.PathLike[str]) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default)
        stream.write("\n")
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    if not isinstance(protocol, dict):
        raise ValueError("Protocol config must be a mapping")
    return protocol


def validate_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    checkpoint = resolve(protocol["checkpoint"]["model"])
    config = resolve(protocol["checkpoint"]["config"])
    actual = {"checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(config)}
    expected = {
        "checkpoint_sha256": protocol["checkpoint"]["model_sha256"],
        "config_sha256": protocol["checkpoint"]["config_sha256"],
    }
    if actual != expected:
        raise RuntimeError(f"Frozen artifact hash mismatch: actual={actual}, expected={expected}")
    return actual


def scenario_layout(dataset: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = 0
    for scene_id, cumulative in enumerate(dataset.len_record):
        scene = dataset.scenario_database[scene_id]
        ego_id = next(cav_id for cav_id, content in scene.items() if content["ego"])
        timestamps = [key for key in scene[ego_id].keys() if key != "ego"]
        rows.append(
            {
                "scene_id": scene_id,
                "scene_name": Path(dataset.scenario_folders[scene_id]).name,
                "global_start": previous,
                "frame_count": cumulative - previous,
                "timestamps": timestamps,
            }
        )
        previous = cumulative
    return rows


def selected_rows(layout: list[dict[str, Any]], protocol: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode == "smoke":
        scene_ids = protocol["smoke"]["scene_ids"]
        local_indices = protocol["smoke"]["local_indices"]
    else:
        selection = protocol["selection"]
        scene_ids = selection["scene_ids"]
        local_indices = range(selection["local_index_start"], selection["local_index_end_inclusive"] + 1)
    by_id = {row["scene_id"]: row for row in layout}
    output: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        row = by_id[scene_id]
        for local_index in local_indices:
            if local_index >= row["frame_count"]:
                raise RuntimeError(f"Scene {scene_id} lacks local index {local_index}")
            output.append(
                {
                    "scene_id": scene_id,
                    "scene_name": row["scene_name"],
                    "local_index": local_index,
                    "global_index": row["global_start"] + local_index,
                    "timestamp": row["timestamps"][local_index],
                    "timestamps": row["timestamps"],
                }
            )
    keys = [(row["scene_id"], row["local_index"]) for row in output]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate selected scene/frame keys")
    return output


def blank_stat(eval_class: str, thresholds: Iterable[float]) -> dict[str, Any]:
    return {
        eval_class: {
            float(iou): {"tp": [], "fp": [], "gt": 0, "score": [], "frame": [], "iou": []}
            for iou in thresholds
        }
    }


def merge_stat(target: dict[str, Any], source: dict[str, Any]) -> None:
    for class_name, threshold_stats in source.items():
        for threshold, values in threshold_stats.items():
            destination = target[class_name][threshold]
            for key in ("tp", "fp", "score", "frame", "iou"):
                destination[key].extend(values[key])
            destination["gt"] += values["gt"]


def stat_metrics(eval_utils: Any, stat: dict[str, Any], eval_class: str, thresholds: Iterable[float]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"upstream_frame_order": {}, "global_score_sort": {}}
    for threshold in thresholds:
        key = float(threshold)
        for label, global_sort in (("upstream_frame_order", False), ("global_score_sort", True)):
            ap, recall, precision = eval_utils.calculate_ap(stat[eval_class], key, global_sort)
            metrics[label][f"ap{int(round(key * 100))}"] = float(ap)
            metrics[label][f"final_recall_at_{int(round(key * 100))}"] = float(recall[-2]) if len(recall) >= 2 else 0.0
            metrics[label][f"final_precision_at_{int(round(key * 100))}"] = float(precision[-2]) if len(precision) >= 2 else 0.0
    return metrics


def mask_detection_range(detection: dict[str, Any], hypes: dict[str, Any]) -> dict[str, Any]:
    boxes = detection["box3d_lidar"]
    bounds = hypes["dataset"]["eval_range"]
    mask = (
        (boxes[:, 0] >= bounds[0])
        & (boxes[:, 0] <= bounds[3])
        & (boxes[:, 1] >= bounds[1])
        & (boxes[:, 1] <= bounds[4])
    )
    output: dict[str, Any] = {}
    for key, value in detection.items():
        output[key] = value[mask] if hasattr(value, "shape") and value.shape[:1] == boxes.shape[:1] else value
    return output


def extract_frame_gt(torch: Any, gt_batch: Any, hypes: dict[str, Any]) -> Any:
    """Remove the batch column while preserving the empty-frame schema."""
    if gt_batch.ndim == 1:
        if gt_batch.numel() != 0:
            raise RuntimeError(f"Unexpected 1-D nonempty label tensor: {tuple(gt_batch.shape)}")
        gt = gt_batch.new_empty((0, 8))
    elif gt_batch.ndim == 2 and gt_batch.shape[1] >= 9:
        gt = gt_batch[gt_batch[:, 0] == 0, 1:].clone()
    else:
        raise RuntimeError(f"Unexpected label tensor shape: {tuple(gt_batch.shape)}")
    if gt.shape[0]:
        class_map = torch.as_tensor(hypes["dataset"]["cls_map"], device=gt.device)
        gt[:, 7] = class_map[gt[:, 7].long()]
    return gt


def evaluate_frame(
    torch: Any,
    box_utils: Any,
    eval_utils: Any,
    detection: dict[str, Any],
    gt_batch: Any,
    hypes: dict[str, Any],
    eval_class: str,
    thresholds: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    detection = mask_detection_range(detection, hypes)
    gt = extract_frame_gt(torch, gt_batch, hypes)
    det_corners = box_utils.boxes_to_corners2d(detection["box3d_lidar"], order="hwl")[:, :, :2]
    gt_corners = box_utils.boxes_to_corners2d(gt[:, :7], order="hwl")[:, :, :2]
    stat = blank_stat(eval_class, thresholds)
    # Upstream sorts its local scores in place across the threshold loop,
    # but leaves boxes in their original order. Separate calls preserve the
    # box/score association for unsorted (e.g. ego-first) detections at every
    # threshold, without changing detections or third-party algorithm code.
    for threshold in thresholds:
        eval_utils.caluclate_tp_fp(
            det_corners,
            detection["scores"],
            detection["label_preds"],
            gt_corners,
            gt[:, 7],
            stat,
            __import__("numpy").asarray([[threshold]]),
            [eval_class],
        )
    counts = {"detections": int(detection["scores"].shape[0]), "gt": int(gt.shape[0])}
    for threshold in thresholds:
        values = stat[eval_class][float(threshold)]
        suffix = int(round(float(threshold) * 100))
        counts[f"tp{suffix}"] = int(sum(values["tp"]))
        counts[f"fp{suffix}"] = int(sum(values["fp"]))
    return stat, counts


def frame_path_audit(
    batch: dict[str, Any], row: dict[str, Any], delay_ms: int, ego_only: bool, lidar_frequency_hz: int
) -> dict[str, Any]:
    paths = [str(item) for item in batch["lidar_path"]]
    if len(paths) != 6:
        raise RuntimeError(f"Expected 6 flattened lidar slots, got {len(paths)}")
    delay_frames = delay_ms * lidar_frequency_hz // 1000
    timestamps = row["timestamps"]
    index = row["local_index"]
    expected_ego = [timestamps[index - offset] for offset in range(2)]
    expected_cav = [timestamps[index - delay_frames - offset] for offset in range(4)]
    actual_ego = [Path(path).stem if path else "" for path in paths[:2]]
    actual_cav = [Path(path).stem if path else "" for path in paths[2:]]
    raw_delays = batch["time_delays"][0]
    actual_delays = [float(item) for item in raw_delays]
    if actual_ego != expected_ego:
        raise RuntimeError(f"Ego path audit failed: {actual_ego} != {expected_ego}")
    if ego_only:
        if actual_cav != ["", "", "", ""] or actual_delays != [0.0, 0.0]:
            raise RuntimeError(f"Ego-only audit failed: paths={actual_cav}, delays={actual_delays}")
    else:
        if actual_cav != expected_cav or actual_delays != [0.0, float(delay_frames)]:
            raise RuntimeError(
                f"Collaborative path audit failed: paths={actual_cav}/{expected_cav}, delays={actual_delays}"
            )
    return {
        "paths": paths,
        "actual_path_timestamps": {"ego": actual_ego, "cooperative": actual_cav},
        "expected_path_timestamps": {"ego": expected_ego, "cooperative": [] if ego_only else expected_cav},
        "actual_time_delay_frames": actual_delays,
        "expected_cooperative_delay_frames": 0 if ego_only else delay_frames,
        "ego_only": ego_only,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=json_default) + "\n")
        stream.flush()


def build_hypes(yaml_utils: Any, protocol: dict[str, Any], delay_ms: int) -> dict[str, Any]:
    hypes = yaml_utils.load_yaml(str(resolve(protocol["checkpoint"]["config"])))
    hypes["validate_dir"] = str(resolve(protocol["dataset"]["root"]))
    hypes["dataset"]["infer_range"] = True
    hypes["train_params"]["train_batch_size"] = 1
    hypes["train_params"]["val_batch_size"] = 1
    hypes["wild_setting"]["agent_i_delay"] = delay_ms
    hypes["delay_ego"] = int(protocol["conditions"]["ego_delay_ms"])
    return yaml_utils.check_pillar_params(hypes)


def environment(torch: Any, np: Any, spconv: Any, protocol_path: Path) -> dict[str, Any]:
    gpu_index = torch.cuda.current_device()
    return {
        "timestamp": utc_now(),
        "hostname": platform.node(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "spconv": getattr(spconv, "__version__", "unknown"),
        "cuda_device_index": gpu_index,
        "cuda_device_name": torch.cuda.get_device_name(gpu_index),
        "protocol_config": str(protocol_path),
        "protocol_config_sha256": sha256(protocol_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("validate", "smoke", "pilot"), default="pilot")
    args = parser.parse_args()

    protocol_path = resolve(args.protocol_config)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(protocol_path)
    hashes = validate_hashes(protocol)

    source_root = resolve(protocol["source"]["root"])
    sys.path.insert(0, str(source_root))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import numpy as np
    import spconv
    import torch
    from datasets import build_dataset
    from hypes_yaml import yaml_utils
    from utils import box_utils, eval_utils, train_utils

    seed = int(protocol["runtime"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = bool(protocol["runtime"]["deterministic_cudnn"])
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_index = int(protocol["runtime"]["device_index"])
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    first_delay = int(protocol["conditions"]["delay_ms"][0])
    selection_dataset = build_dataset(build_hypes(yaml_utils, protocol, first_delay), set="test")
    layout = scenario_layout(selection_dataset)
    rows = selected_rows(layout, protocol, "smoke" if args.mode == "smoke" else "pilot")
    total_timestamps = sum(row["frame_count"] for row in layout)
    selection_report = {
        "status": "passed",
        "mode": args.mode,
        "scenario_count": len(layout),
        "total_timestamps": total_timestamps,
        "selected_frames_per_condition": len(rows),
        "selected_keys_unique": len(rows) == len({(row["scene_id"], row["local_index"]) for row in rows}),
        "layout": [{key: value for key, value in row.items() if key != "timestamps"} for row in layout],
        "selection": [{key: value for key, value in row.items() if key != "timestamps"} for row in rows],
        "artifact_hashes": hashes,
    }
    if len(layout) != int(protocol["dataset"]["expected_total_scenes"]):
        raise RuntimeError("Unexpected total scenario count")
    if total_timestamps != int(protocol["dataset"]["expected_total_timestamps"]):
        raise RuntimeError("Unexpected total timestamp count")
    if args.mode == "pilot" and len(rows) != int(protocol["selection"]["expected_frames_per_condition"]):
        raise RuntimeError("Unexpected pilot selection count")
    atomic_json(output_dir / "selection_audit.json", selection_report)
    print(json.dumps({key: selection_report[key] for key in ("status", "mode", "scenario_count", "total_timestamps", "selected_frames_per_condition")}, indent=2))
    if args.mode == "validate":
        return
    del selection_dataset

    delays = protocol["smoke"]["delay_ms"] if args.mode == "smoke" else protocol["conditions"]["delay_ms"]
    eval_class = protocol["evaluation"]["class_name"]
    thresholds = [float(value) for value in protocol["evaluation"]["iou_thresholds"]]
    ego_only_scenes = set(protocol["selection"]["ego_only_scene_ids"])
    environment_report = environment(torch, np, spconv, protocol_path)
    environment_report.update({"source_commit_declared": protocol["source"]["commit"], **hashes})
    atomic_json(output_dir / "environment.json", environment_report)

    hypes_for_model = build_hypes(yaml_utils, protocol, int(delays[0]))
    model = train_utils.create_model(hypes_for_model).float().to(device)
    checkpoint = torch.load(resolve(protocol["checkpoint"]["model"]), map_location="cpu")
    incompatible = model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    strict_report = {
        "strict": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "model_state_keys": len(model.state_dict()),
        "status": "passed",
    }
    atomic_json(output_dir / "checkpoint_load.json", strict_report)
    model.eval()

    frame_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    condition_summaries: dict[str, Any] = {}
    audit_path = output_dir / "frame_audit.jsonl"
    if audit_path.exists():
        raise RuntimeError(f"Refusing to append to existing frame audit: {audit_path}")

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for delay_ms_raw in delays:
        delay_ms = int(delay_ms_raw)
        hypes = build_hypes(yaml_utils, protocol, delay_ms)
        dataset = build_dataset(hypes, set="test")
        current_layout = scenario_layout(dataset)
        current_rows = selected_rows(current_layout, protocol, "smoke" if args.mode == "smoke" else "pilot")
        condition_stat = blank_stat(eval_class, thresholds)
        scene_stats: dict[int, dict[str, Any]] = {
            scene_id: blank_stat(eval_class, thresholds) for scene_id in sorted({row["scene_id"] for row in current_rows})
        }
        for ordinal, row in enumerate(current_rows, start=1):
            item = dataset[row["global_index"]]
            batch = dataset.collate_batch([item])
            audit = frame_path_audit(
                batch,
                row,
                delay_ms,
                row["scene_id"] in ego_only_scenes,
                int(protocol["conditions"]["lidar_frequency_hz"]),
            )
            torch.cuda.synchronize(device)
            frame_started = time.perf_counter()
            batch_device = train_utils.to_device(batch, device)
            with torch.no_grad():
                _, detections = model(batch_device)
            torch.cuda.synchronize(device)
            inference_s = time.perf_counter() - frame_started
            frame_stat, counts = evaluate_frame(
                torch,
                box_utils,
                eval_utils,
                detections[0],
                batch_device["label_dict"],
                hypes,
                eval_class,
                thresholds,
            )
            merge_stat(condition_stat, frame_stat)
            merge_stat(scene_stats[row["scene_id"]], frame_stat)
            voxel_count = int(batch_device["processed_lidar"]["voxel_coords"].shape[0])
            frame_row = {
                "delay_ms": delay_ms,
                "scene_id": row["scene_id"],
                "scene_name": row["scene_name"],
                "local_index": row["local_index"],
                "global_index": row["global_index"],
                "timestamp": row["timestamp"],
                "ego_only": row["scene_id"] in ego_only_scenes,
                "voxel_count": voxel_count,
                "inference_s": round(inference_s, 6),
                **counts,
            }
            frame_rows.append(frame_row)
            append_jsonl(audit_path, {**frame_row, **audit})
            if ordinal == 1 or ordinal % 16 == 0 or ordinal == len(current_rows):
                print(f"delay={delay_ms}ms progress={ordinal}/{len(current_rows)} scene={row['scene_id']} frame={row['local_index']}", flush=True)

        condition_metrics = stat_metrics(eval_utils, condition_stat, eval_class, thresholds)
        np.save(output_dir / f"result_stat_{delay_ms}ms.npy", condition_stat)
        per_scene_primary: dict[str, list[float]] = {f"ap{int(value * 100)}": [] for value in thresholds}
        per_scene_global: dict[str, list[float]] = {f"ap{int(value * 100)}": [] for value in thresholds}
        for scene_id, stat in scene_stats.items():
            metrics = stat_metrics(eval_utils, stat, eval_class, thresholds)
            row = next(item for item in current_rows if item["scene_id"] == scene_id)
            scene_row: dict[str, Any] = {
                "delay_ms": delay_ms,
                "scene_id": scene_id,
                "scene_name": row["scene_name"],
                "ego_only": scene_id in ego_only_scenes,
                "frames": sum(item["scene_id"] == scene_id for item in current_rows),
            }
            for sort_name, sort_metrics in metrics.items():
                for metric_name, value in sort_metrics.items():
                    scene_row[f"{sort_name}_{metric_name}"] = value
            for key in per_scene_primary:
                per_scene_primary[key].append(metrics["upstream_frame_order"][key])
                per_scene_global[key].append(metrics["global_score_sort"][key])
            scene_rows.append(scene_row)
        scene_equal = {
            "upstream_frame_order": {key: float(np.mean(values)) for key, values in per_scene_primary.items()},
            "global_score_sort": {key: float(np.mean(values)) for key, values in per_scene_global.items()},
        }
        condition_summaries[str(delay_ms)] = {
            "frames": len(current_rows),
            "global_aggregate": condition_metrics,
            "scene_equal_mean": scene_equal,
        }
        atomic_json(output_dir / "partial_summary.json", {"status": "running", "completed_conditions": condition_summaries})
        del dataset

    write_csv(output_dir / "frame_metrics.csv", frame_rows)
    write_csv(output_dir / "scene_metrics.csv", scene_rows)
    delta: dict[str, Any] = {}
    if {"0", "400"}.issubset(condition_summaries):
        for aggregation in ("global_aggregate", "scene_equal_mean"):
            delta[aggregation] = {}
            for sort_name in ("upstream_frame_order", "global_score_sort"):
                delta[aggregation][sort_name] = {}
                for metric in ("ap50", "ap70"):
                    delta[aggregation][sort_name][metric] = (
                        condition_summaries["400"][aggregation][sort_name][metric]
                        - condition_summaries["0"][aggregation][sort_name][metric]
                    )
    expected_total = len(rows) * len(delays)
    unique_keys = {(row["delay_ms"], row["scene_id"], row["local_index"]) for row in frame_rows}
    summary = {
        "status": "completed",
        "mode": args.mode,
        "protocol_id": protocol["protocol_id"],
        "started_at": environment_report["timestamp"],
        "finished_at": utc_now(),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "expected_frame_conditions": expected_total,
        "completed_frame_conditions": len(frame_rows),
        "unique_frame_condition_keys": len(unique_keys),
        "coverage_passed": len(frame_rows) == expected_total == len(unique_keys),
        "conditions": condition_summaries,
        "delta_400ms_minus_0ms": delta,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "artifacts": {
            "frame_metrics": "frame_metrics.csv",
            "scene_metrics": "scene_metrics.csv",
            "frame_audit": "frame_audit.jsonl",
            "environment": "environment.json",
            "checkpoint_load": "checkpoint_load.json",
        },
    }
    if not summary["coverage_passed"]:
        raise RuntimeError(f"Coverage audit failed: {summary}")
    atomic_json(output_dir / "summary.json", summary)
    partial = output_dir / "partial_summary.json"
    if partial.exists():
        partial.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
