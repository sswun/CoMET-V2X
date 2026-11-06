#!/usr/bin/env python3
"""Native saved-shard source AP; accepted author corners, target-first ignore."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "preexperiment/research_baselines"), str(ROOT / "third_party/third_party/related_works/TraF-Align")]
from utils import box_utils, common_utils
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap
from audit_v2xseq_strict_common_roi_20261010 import mask as common_mask


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def polygons(boxes):
    return list(common_utils.convert_format(box_utils.boxes_to_corners2d(torch.from_numpy(boxes.copy()), order="hwl")[:, :, :2].numpy()))


def source_codes(gt, objects):
    if not len(gt): return np.empty(0, np.int8), 0.0
    centers = np.asarray([[obj["3d_location"]["x"], obj["3d_location"]["y"]] for obj in objects], np.float64)
    tree = cKDTree(centers)
    distance, indices = tree.query(gt[:, :2].astype(np.float64))
    if np.any(distance >= 1e-3) or any(len(tree.query_ball_point(point, 1e-3)) != 1 for point in gt[:, :2]):
        raise RuntimeError("source mapping ambiguous or unmatched")
    codes = np.asarray([{"veh": 0, "coop": 1, "inf": 2}[obj["from_side"]] for obj in objects], np.int8)
    return codes[indices], float(distance.max())


def matrix(left, right):
    return np.stack([common_utils.compute_iou(poly, right) for poly in left]) if len(left) else np.empty((0, len(right)), np.float32)


def outcomes(overlap, sides, group, threshold):
    target_mask = np.ones(len(sides), bool) if group == "all" else (sides == 2 if group == "inf" else sides != 2)
    remaining = np.flatnonzero(target_mask).tolist()
    ignored_indices = np.flatnonzero(~target_mask)
    tp = np.zeros(len(overlap), np.int8); fp = np.zeros(len(overlap), np.int8); ignored = np.zeros(len(overlap), np.int8)
    for index, row in enumerate(overlap):
        if remaining and float(row[remaining].max()) >= threshold:
            best = int(np.argmax(row[remaining])); remaining.pop(best); tp[index] = 1
        elif len(ignored_indices) and float(row[ignored_indices].max()) >= threshold:
            ignored[index] = 1
        else:
            fp[index] = 1
    return tp, fp, ignored, int(target_mask.sum())


def metric(tp, fp, scores, count):
    result = voc_ap(tp, fp, scores, int(count), True)
    return {"ap": 100 * result["ap"], "tp": result["tp"], "fp": result["fp"], "gt": int(count)}


def compute(contract, output):
    records = load(ROOT / contract["records"]["path"])
    identity = {int(row["global_index"]): row for row in (json.loads(line) for line in (ROOT / contract["history_manifest"]["path"]).read_text().splitlines())}
    accepted = load(ROOT / contract["accepted_review"]["path"])
    accepted_panels = {int(row["wait_ms"]): row for row in accepted["panels"]}
    labels = ROOT / contract["label_root"]
    frame_cache = {}; panels = []; sequence_rows = []; result_arrays = {}; max_mapping_error = 0.0
    full_replay = []; outcome_digest = hashlib.sha256(); condition_digest = hashlib.sha256()
    for wait in (0, 100, 200, 400):
        selected = [row for row in records if int(row["wait_ms"]) == wait]
        if len(selected) != 3316 or len({int(row["global_index"]) for row in selected}) != 3316:
            raise RuntimeError("native population mismatch")
        stats = defaultdict(lambda: {"tp": [], "fp": [], "scores": [], "ignored": [], "gt": 0})
        perseq = defaultdict(lambda: {"tp": [], "fp": [], "scores": [], "gt": 0})
        shard_path = ROOT / contract["shards"][str(wait)]["path"]
        with np.load(shard_path, allow_pickle=False) as shard:
            for ordinal, record in enumerate(selected):
                row = identity[int(record["global_index"])]
                if str(row["sequence"]) != str(record["sequence"]) or int(row["decision_us"]) != int(record["decision_us"]):
                    raise RuntimeError("native receiver identity mismatch")
                frame = str(row["current_vehicle_frame"])
                if frame not in frame_cache: frame_cache[frame] = load(labels / (frame + ".json"))
                index = int(record["shard_case"])
                gt = shard[f"c{index}_gt"]
                side, error = source_codes(gt, frame_cache[frame]); max_mapping_error = max(max_mapping_error, error)
                condition_digest.update(json.dumps([wait, record["global_index"], record["sequence"], record["decision_us"], frame], separators=(",", ":")).encode())
                for roi in ("native", "common"):
                    keep_gt = gt[:, 7] == 0
                    if roi == "common": keep_gt &= common_mask(gt[:, :7])
                    target = gt[keep_gt, :7]; source = side[keep_gt]
                    target_polygons = polygons(target)
                    for arm in contract["arms"]:
                        boxes = shard[f"c{index}_{arm}_box3d_lidar"]
                        scores = shard[f"c{index}_{arm}_scores"]
                        classes = shard[f"c{index}_{arm}_label_preds"]
                        # Same author prediction-range filter as accepted native review.
                        bounds = contract["native_eval_range"]
                        keep = (classes == 0) & (boxes[:, 0] >= bounds[0]) & (boxes[:, 0] <= bounds[3]) & (boxes[:, 1] >= bounds[1]) & (boxes[:, 1] <= bounds[4])
                        if roi == "common": keep &= common_mask(boxes)
                        boxes, scores = boxes[keep], scores[keep]
                        order = np.argsort(-scores)
                        scores = scores[order].astype(np.float64)
                        overlap = matrix(polygons(boxes[order]), target_polygons)
                        for group in ("all", "veh_coop", "inf"):
                            for threshold in (.5, .7):
                                tp, fp, ignore, count = outcomes(overlap, source, group, threshold)
                                key = (roi, arm, group, threshold)
                                value = stats[key]; value["gt"] += count
                                for field, array in (("tp", tp), ("fp", fp), ("ignored", ignore), ("scores", scores)):
                                    value[field].extend(array.tolist())
                                seq = perseq[(record["sequence"], *key)]; seq["gt"] += count
                                for field, array in (("tp", tp), ("fp", fp), ("scores", scores)): seq[field].extend(array.tolist())
                                outcome_digest.update(tp.tobytes()); outcome_digest.update(fp.tobytes()); outcome_digest.update(ignore.tobytes())
                if ordinal % 256 == 0:
                    atomic(output / "status.json", {"status": "running", "wait_ms": wait, "completed_in_wait": ordinal + 1, "zero_forwards": True})
        for (roi, arm, group, threshold), value in stats.items():
            metrics = metric(value["tp"], value["fp"], value["scores"], value["gt"])
            panels.append({"wait_ms": wait, "roi": roi, "arm": arm, "source_group": group, "threshold": threshold, **metrics, "ignored": int(sum(value["ignored"]))})
            prefix = f"w{wait}_{roi}_{arm}_{group}_{round(100*threshold)}"
            for field in ("tp", "fp", "scores", "ignored"):
                result_arrays[prefix + "_" + field] = np.asarray(value[field], np.float64 if field == "scores" else np.int8)
            if group == "all":
                expected = 100 * accepted_panels[wait]["metrics"][roi][arm]["global_score_sort"][f"ap{round(100*threshold)}"]
                full_replay.append(abs(metrics["ap"] - expected) < 1e-8)
        for (sequence, roi, arm, group, threshold), value in perseq.items():
            sequence_rows.append({"sequence": sequence, "wait_ms": wait, "roi": roi, "arm": arm,
                                  "source_group": group, "threshold": threshold,
                                  **metric(value["tp"], value["fp"], value["scores"], value["gt"])})
        if not all(full_replay): raise RuntimeError(("all-GT native/common AP replay mismatch", wait))
    lookup = {(row["wait_ms"], row["roi"], row["arm"], row["source_group"], row["threshold"], row["sequence"]): row for row in sequence_rows}
    sequences = sorted({str(row["sequence"]) for row in records}); bootstrap = []
    rng = np.random.default_rng(20261031)
    left, right = contract["contrast"]
    for wait in (0, 100, 200, 400):
        for roi in ("native", "common"):
            for group in ("veh_coop", "inf"):
                values = np.asarray([lookup[(wait, roi, left, group, .7, seq)]["ap"] - lookup[(wait, roi, right, group, .7, seq)]["ap"] for seq in sequences if lookup[(wait, roi, left, group, .7, seq)]["gt"] > 0])
                draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
                bootstrap.append({"wait_ms": wait, "roi": roi, "source_group": group,
                    "contrast": left + "-" + right, "eligible_sequences": len(values), "replicates": 10000,
                    "estimand": "macro-sequence AP70 difference, not pooled-AP confidence interval",
                    "mean_pp": float(values.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]})
    checks = {"all_gt_native_common_AP_replay": all(full_replay), "all_13264_conditions_unique": len(records) == 13264,
              "source_mapping_unique_under_1mm": max_mapping_error < 1e-3,
              "panels_exact_96": len(panels) == 96, "bootstrap_16": len(bootstrap) == 16,
              "zero_GPU_and_forward_update": not torch.cuda.is_initialized(), "no_method_selection": True}
    if not all(checks.values()): raise RuntimeError(checks)
    return {"status": "producer_complete_pending_independent_review", "panels": panels, "per_sequence_panels": sequence_rows,
            "paired_sequence_bootstrap": bootstrap, "checks": checks, "max_source_center_error_m": max_mapping_error,
            "outcome_digest": outcome_digest.hexdigest(), "condition_digest": condition_digest.hexdigest()}, result_arrays


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); tick = time.monotonic(); cpu = time.process_time()
    if args.output.exists(): raise FileExistsError(args.output)
    contract = load(args.contract)
    for item in contract["frozen_inputs"]:
        if sha(ROOT / item["path"]) != item["sha256"]: raise RuntimeError(("input drift", item["path"]))
    args.output.mkdir(parents=True)
    result, arrays = compute(contract, args.output)
    array_path = args.output / "outcomes.npz"; np.savez_compressed(array_path, **arrays)
    result.update(contract_sha256=sha(args.contract), task_id=contract["task_id"],
                  producer_sha256=sha(Path(__file__)), outcomes={"path": str(array_path.relative_to(ROOT)), "sha256": sha(array_path)},
                  actual_cost={"wall_seconds": time.monotonic()-tick, "cpu_process_seconds": time.process_time()-cpu,
                               "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                               "new_forwards": 0, "updates": 0, "gpu_seconds": 0}, boundary=contract["boundary"])
    atomic(args.output / "summary.json", result)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}), flush=True)


if __name__ == "__main__": main()
