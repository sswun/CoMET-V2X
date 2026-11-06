#!/usr/bin/env python3
"""Fixed-epoch raw evaluator for same-loader adapted V2X-Seq baselines.

One invocation evaluates one frozen delay arm and saves every post-processed
prediction before metric-range masking.  The companion CPU reviewer recomputes
all matching and AP values from these NPZ files without importing model code.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXPECTED_EPOCH = 60
EXPECTED_FRAMES = 3316
ALLOWED_DELAYS_MS = (0, 100, 200, 400)
SEED = 2023


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def install_corrected_index(dataset: Any, np: Any) -> None:
    """Repair the released non-first-sequence -1 boundary lookup."""
    original = dataset.retrieve_base_data

    def corrected(idx: int, cur_ego_pose_flag: bool = False) -> Any:
        scene_index = int(np.where((idx < dataset.len_record) == True)[0][0])
        if scene_index == 0:
            return original(idx, cur_ego_pose_flag=cur_ego_pose_flag)
        dataset.len_record[scene_index - 1] -= 1
        try:
            return original(idx, cur_ego_pose_flag=cur_ego_pose_flag)
        finally:
            dataset.len_record[scene_index - 1] += 1

    dataset.retrieve_base_data = corrected


def dataset_rows(dataset: Any) -> list[dict[str, Any]]:
    sequences = list(dataset.scenario_folders.keys())
    rows: list[dict[str, Any]] = []
    previous = 0
    for scene_id, cumulative_raw in enumerate(dataset.len_record):
        cumulative = int(cumulative_raw)
        scene = dataset.scenario_database[scene_id]
        ego_id = next(cav_id for cav_id, content in scene.items() if isinstance(cav_id, int) and content["ego"])
        timestamps = [key for key in scene[ego_id] if key not in {"ego", "agent"}]
        for local_index in range(cumulative - previous):
            rows.append(
                {
                    "scene_id": scene_id,
                    "sequence": Path(str(sequences[scene_id])).name,
                    "global_index": previous + local_index,
                    "local_index": local_index,
                    "timestamps": timestamps,
                }
            )
        previous = cumulative
    return rows


def physical_history(start: str, count: int, side_infos: dict[str, Any]) -> list[str]:
    return [
        candidate if candidate in side_infos else start
        for offset in range(count)
        for candidate in [str(int(start) - offset).zfill(len(start))]
    ]


def path_audit(
    dataset: Any,
    batch: dict[str, Any],
    row: dict[str, Any],
    delay_ms: int,
    data_root: Path,
    frame_his: int,
    cav_frame_his: int,
    lidar_frequency_hz: int,
) -> dict[str, Any]:
    scene = dataset.scenario_database[row["scene_id"]]
    ego_id = next(cav_id for cav_id, content in scene.items() if isinstance(cav_id, int) and content["ego"])
    cooperative_id = next(cav_id for cav_id, content in scene.items() if isinstance(cav_id, int) and not content["ego"])
    current_key = row["timestamps"][row["local_index"]]
    delay_frames = int(delay_ms) * int(lidar_frequency_hz) // 1000
    cooperative_local = max(0, row["local_index"] - delay_frames)
    cooperative_key = row["timestamps"][cooperative_local]
    label_frame = str(scene[ego_id][current_key]["timestamp"])
    vehicle_frame = str(scene[ego_id][current_key]["timestamp"])
    infrastructure_frame = str(scene[cooperative_id][cooperative_key]["timestamp"])
    expected_ego = physical_history(vehicle_frame, frame_his, dataset.vehicle_infos[row["sequence"]])
    expected_cooperative = physical_history(
        infrastructure_frame, cav_frame_his, dataset.infra_infos[row["sequence"]]
    )
    paths = [str(value) for value in batch["lidar_path"]]
    stems = [Path(path).stem if path else "" for path in paths]
    expected = expected_ego + expected_cooperative
    if stems != expected:
        raise RuntimeError(
            f"path audit failed delay={delay_ms} scene={row['scene_id']} local={row['local_index']}: "
            f"{stems} != {expected}"
        )
    lexical_root = os.path.abspath(data_root)
    if any(not Path(path).is_file() or os.path.commonpath([lexical_root, os.path.abspath(path)]) != lexical_root for path in paths):
        raise RuntimeError("input path missing or lexically escaped the frozen shared view")
    actual_delays = [float(value) for value in batch["time_delays"][0]]
    if actual_delays[:2] != [0.0, float(delay_frames)]:
        raise RuntimeError(f"time-delay audit failed: {actual_delays}")
    return {
        "input_path_stems": stems,
        "expected_path_stems": expected,
        "effective_label_frame": label_frame,
        "effective_vehicle_frame": vehicle_frame,
        "effective_infrastructure_frame": infrastructure_frame,
        "actual_time_delay_frames": actual_delays,
        "expected_remote_delay_frames": delay_frames,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delay-ms", type=int, choices=ALLOWED_DELAYS_MS, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    config = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    split = Path(args.split).resolve()
    data_manifest = Path(args.data_manifest).resolve()
    protocol_path = Path(args.protocol).resolve()
    training_run = Path(args.training_run).resolve()
    output = Path(args.output).resolve()
    status_path = output / "status.json"
    raw_dir = output / "raw"
    if output.exists():
        raise FileExistsError(f"evaluation output already exists and will not be overwritten: {output}")
    raw_dir.mkdir(parents=True)

    state: dict[str, Any] = {
        "schema": "traf_v2xseq_adapted_fixed60_raw_arm_v1",
        "created_at": now(),
        "status": "initializing",
        "delay_ms": args.delay_ms,
        "training_run": str(training_run),
        "output": str(output),
        "checkpoint_selected_by_AP": False,
        "public_exposed_validation_only": True,
    }
    atomic_json(status_path, state)

    try:
        for path in (source, config, data_root, split, data_manifest, protocol_path, training_run):
            if not path.exists():
                raise FileNotFoundError(path)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
        training = json.loads((training_run / "status.json").read_text(encoding="utf-8"))
        checkpoint = training_run / f"net_epoch{EXPECTED_EPOCH}.pth"
        if protocol.get("protocol_id") != "v2xseq_main_domain_shared_external_benchmarks_v1":
            raise RuntimeError("unexpected common protocol")
        if manifest.get("status") != "passed" or len(manifest.get("val_sequences", [])) != 21:
            raise RuntimeError("shared data manifest did not pass the expected val21 boundary")
        if protocol["input_semantics"]["evaluation_remote_delay_ms"] != list(ALLOWED_DELAYS_MS):
            raise RuntimeError("protocol delay arms changed")
        if args.delay_ms not in protocol["input_semantics"]["evaluation_remote_delay_ms"]:
            raise RuntimeError("delay is outside the frozen protocol")
        if not (
            training.get("status") == "completed"
            and training.get("mode") == "full"
            and training.get("target_epochs") == EXPECTED_EPOCH
            and training.get("completed_epoch") == EXPECTED_EPOCH
            and training.get("checkpoint_selected_by_AP") is False
            and training.get("test_dataset_built") is False
        ):
            raise RuntimeError("training run has not reached the fixed full60 terminal state")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        epoch_record = next((row for row in training.get("epochs", []) if row.get("epoch") == EXPECTED_EPOCH), None)
        if not epoch_record or epoch_record.get("model_checkpoint_sha256") != sha256(checkpoint):
            raise RuntimeError("fixed epoch60 checkpoint hash is absent or mismatched")
        frozen_paths = (config, split, data_manifest, protocol_path)
        frozen_hashes = {str(path): sha256(path) for path in frozen_paths}
        if any(training.get("input_sha256", {}).get(path) != digest for path, digest in frozen_hashes.items()):
            raise RuntimeError("training/evaluation frozen-input identity mismatch")

        sys.path.insert(0, str(source))
        os.environ.setdefault("MPLBACKEND", "Agg")
        import numpy as np
        import spconv
        import torch
        from datasets import build_dataset
        from hypes_yaml import yaml_utils
        from utils import box_utils, eval_utils, train_utils
        import traf_align_v2v4real_pilot as common

        if np.__version__ != "1.23.5" or getattr(spconv, "__version__", "unknown") != "2.1.25":
            raise RuntimeError(
                f"environment mismatch numpy={np.__version__} spconv={getattr(spconv, '__version__', 'unknown')}"
            )
        torch.cuda.set_device(args.device_index)
        device = torch.device(f"cuda:{args.device_index}")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(2)

        hypes = yaml_utils.load_yaml(str(config))
        hypes["root_dir"] = str(data_root) + os.sep
        hypes["split_dir"] = str(split)
        hypes["train_params"]["train_batch_size"] = 1
        hypes["train_params"]["val_batch_size"] = 1
        hypes["wild_setting"]["agent_i_delay"] = int(args.delay_ms)
        hypes["delay_ego"] = 0
        hypes = yaml_utils.check_pillar_params(hypes)
        if hypes["fusion"]["dataset"] != "v2xseq" or hypes["wild_setting"].get("async") is not True:
            raise RuntimeError("evaluation config is not the frozen asynchronous V2X-Seq family")
        if int(hypes["wild_setting"]["lidar_frequency"]) != 10:
            raise RuntimeError("unexpected LiDAR frequency")
        if list(hypes["wild_setting"].get("agent_i_delay_train_aug", [])) != [0, 0]:
            raise RuntimeError("training delay augmentation identity changed")
        effective_path = output / "effective_config.yaml"
        with effective_path.open("w", encoding="utf-8") as stream:
            yaml.dump(hypes, stream, sort_keys=True)

        model = train_utils.create_model(hypes).float().to(device)
        checkpoint_value = torch.load(checkpoint, map_location="cpu")
        incompatible = model.load_state_dict(checkpoint_value, strict=True)
        del checkpoint_value
        model.eval()
        if list(incompatible.missing_keys) or list(incompatible.unexpected_keys):
            raise RuntimeError("strict checkpoint load unexpectedly returned incompatible keys")
        if not all(bool(torch.isfinite(value).all().item()) for value in model.state_dict().values()):
            raise RuntimeError("checkpoint contains non-finite model tensors")

        dataset = build_dataset(hypes, set="val")
        install_corrected_index(dataset, np)
        if len(dataset) != EXPECTED_FRAMES:
            raise RuntimeError(f"validation cardinality changed: {len(dataset)}")
        rows = dataset_rows(dataset)
        if len(rows) != EXPECTED_FRAMES or len({row["global_index"] for row in rows}) != EXPECTED_FRAMES:
            raise RuntimeError("corrected-index row layout is incomplete or non-unique")
        observed_sequences = sorted({row["sequence"] for row in rows})
        if observed_sequences != sorted(manifest["val_sequences"]):
            raise RuntimeError("loader sequences differ from the frozen val21 manifest")

        eval_class = "vehicle"
        thresholds = [0.5, 0.7]
        stat = common.blank_stat(eval_class, thresholds)
        frame_rows: list[dict[str, Any]] = []
        prediction_digest = hashlib.sha256()
        gt_digest = hashlib.sha256()
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        state.update(
            status="running",
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
            input_sha256=frozen_hashes,
            evaluator_sha256=sha256(Path(__file__).resolve()),
            method_config_name=hypes.get("name"),
            model_class=type(model).__name__,
            model_parameters=sum(parameter.numel() for parameter in model.parameters()),
            validation_frames=EXPECTED_FRAMES,
            sequences=observed_sequences,
            eval_range=[float(value) for value in hypes["dataset"]["eval_range"]],
            raw_semantics="postprocessed method outputs before evaluator range masking; current-frame cooperative GT",
        )
        atomic_json(status_path, state)

        for ordinal, row in enumerate(rows, start=1):
            item = dataset[row["global_index"]]
            batch = dataset.collate_batch([item])
            audit = path_audit(
                dataset,
                batch,
                row,
                args.delay_ms,
                data_root,
                int(hypes["dataset"]["frame_his"]),
                int(hypes["dataset"]["cav_frame_his"]),
                int(hypes["wild_setting"]["lidar_frequency"]),
            )
            batch_device = train_utils.to_device(batch, device)
            torch.cuda.synchronize(device)
            frame_started = time.monotonic()
            with torch.no_grad():
                output_value = model(batch_device)
            torch.cuda.synchronize(device)
            inference_s = time.monotonic() - frame_started
            if not isinstance(output_value, tuple) or len(output_value) != 2:
                raise RuntimeError("evaluation model did not return predictions and detections")
            _, detections = output_value
            if not isinstance(detections, list) or len(detections) != 1:
                raise RuntimeError("batch1 evaluation returned unexpected detection cardinality")
            detection = detections[0]
            required = ("box3d_lidar", "scores", "label_preds")
            if any(key not in detection for key in required):
                raise RuntimeError(f"detection keys missing: {sorted(detection)}")
            gt = common.extract_frame_gt(torch, batch_device["label_dict"], hypes)
            arrays = {
                "pred_boxes": detection["box3d_lidar"].detach().cpu().numpy(),
                "pred_scores": detection["scores"].detach().cpu().numpy(),
                "pred_labels": detection["label_preds"].detach().cpu().numpy(),
                "gt_boxes": gt[:, :7].detach().cpu().numpy(),
                "gt_labels": gt[:, 7].detach().cpu().numpy(),
            }
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise RuntimeError(f"non-finite raw array at frame {row['global_index']}")
            raw_path = raw_dir / f"frame_{row['global_index']:04d}.npz"
            np.savez_compressed(raw_path, **arrays)
            raw_sha = sha256(raw_path)
            prediction_digest.update(arrays["pred_boxes"].tobytes())
            prediction_digest.update(arrays["pred_scores"].tobytes())
            prediction_digest.update(arrays["pred_labels"].tobytes())
            gt_digest.update(arrays["gt_boxes"].tobytes())
            gt_digest.update(arrays["gt_labels"].tobytes())

            frame_stat, counts = common.evaluate_frame(
                torch, box_utils, eval_utils, detection, batch_device["label_dict"], hypes, eval_class, thresholds
            )
            counts["gt_vehicle"] = int((arrays["gt_labels"].astype(np.int64) == 0).sum())
            common.merge_stat(stat, frame_stat)
            record = {
                "delay_ms": args.delay_ms,
                "scene_id": row["scene_id"],
                "sequence": row["sequence"],
                "local_index": row["local_index"],
                "global_index": row["global_index"],
                "raw_path": str(raw_path),
                "raw_sha256": raw_sha,
                "inference_s": inference_s,
                "voxel_count": int(batch_device["processed_lidar"]["voxel_coords"].shape[0]),
                "communication_statistic": None,
                "communication_statistic_reason": "not exposed by this repository model forward API",
                **counts,
                **audit,
            }
            frame_rows.append({key: value for key, value in record.items() if key not in {"input_path_stems", "expected_path_stems", "actual_time_delay_frames"}})
            append_jsonl(output / "frame_manifest.jsonl", record)
            if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(rows):
                print(f"delay={args.delay_ms} progress={ordinal}/{len(rows)}", flush=True)
            if ordinal % 100 == 0:
                torch.cuda.empty_cache()

        metrics = common.stat_metrics(eval_utils, stat, eval_class, thresholds)
        write_csv(output / "frame_metrics.csv", frame_rows)
        np.save(output / "runner_result_stat.npy", stat)
        summary = {
            "schema": "traf_v2xseq_adapted_fixed60_raw_arm_summary_v1",
            "status": "completed",
            "finished_at": now(),
            "delay_ms": args.delay_ms,
            "frames": len(frame_rows),
            "unique_frames": len({row["global_index"] for row in frame_rows}),
            "sequences": len({row["sequence"] for row in frame_rows}),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "fixed_epoch": EXPECTED_EPOCH,
            "checkpoint_selected_by_AP": False,
            "index_semantics": "corrected_index",
            "dataset_split": "public_labeled_val21_exposed_not_hidden_test",
            "metrics": metrics,
            "prediction_content_digest": prediction_digest.hexdigest(),
            "gt_content_digest": gt_digest.hexdigest(),
            "eval_range": [float(value) for value in hypes["dataset"]["eval_range"]],
            "elapsed_s": time.monotonic() - started,
            "peak_cuda_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "environment": {
                "hostname": os.uname().nodename,
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "numpy": np.__version__,
                "spconv": getattr(spconv, "__version__", "unknown"),
                "gpu": torch.cuda.get_device_name(device),
                "source_commit": subprocess.run(
                    ["git", "-C", str(source), "rev-parse", "HEAD"],
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
            },
            "claim_boundary": "Adapted same-loader public validation result at fixed epoch60; not an author checkpoint or hidden-test score unless separately established.",
        }
        if not (summary["frames"] == summary["unique_frames"] == EXPECTED_FRAMES and summary["sequences"] == 21):
            raise RuntimeError("terminal evaluation coverage failed")
        atomic_json(output / "summary.json", summary)
        state.update(
            status="completed",
            finished_at=summary["finished_at"],
            summary=str(output / "summary.json"),
            frames=summary["frames"],
            prediction_content_digest=summary["prediction_content_digest"],
            gt_content_digest=summary["gt_content_digest"],
        )
        atomic_json(status_path, state)
        del dataset, model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        state.update(
            status="failed_preserved",
            failed_at=now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            partial_raw_files=len(list(raw_dir.glob("frame_*.npz"))) if raw_dir.exists() else 0,
        )
        atomic_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
