#!/usr/bin/env python3
"""Run official RoCo/CBM box-pose modules on the frozen TraF/SPD same pool."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "preexperiment/research_baselines"
sys.path.insert(0, str(BASE))

# Import the reviewed CoAlign/g2o overlay before FreeAlign changes sys.path.
import run_roco_official_v2v_pose_qualification_v1_20261028 as roco
import run_cbm_official_v2v_pose_full_v1_20261028 as cbm
import run_freealign_official_box_same_pool_v1_20261027 as freealign
from v2xseq_real_motion_packets import transform_boxes7d


WAITS = (0, 100, 200, 400)
METHODS = ("stale", "causal_kf", "freealign", "roco", "cbm")
READOUTS = ("raw", "fresh39d")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def evidence(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def nominal_pose(meta: dict[str, Any]) -> np.ndarray:
    if meta.get("source") is None:
        return np.eye(4, dtype=np.float64)
    value = np.asarray(meta["source"]["source_to_current_ego"], np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise RuntimeError("invalid nominal source_to_current_ego")
    return roco.planar(value)


def apply_pose(source_boxes: np.ndarray, pose: np.ndarray) -> np.ndarray:
    if not len(source_boxes):
        return np.empty((0, 7), np.float32)
    return transform_boxes7d(source_boxes, pose, np.eye(4, dtype=np.float64)).astype(np.float32)


def materialize(dataset: Any, role: str, common: dict[str, np.ndarray], roco_module: Any,
                cbm_matcher: Any, output: Path) -> dict[str, Any]:
    roco_boxes: list[np.ndarray] = []; roco_features: list[np.ndarray] = []
    cbm_boxes: list[np.ndarray] = []; cbm_features: list[np.ndarray] = []
    roco_status: list[str] = []; cbm_status: list[str] = []
    roco_seconds: list[float] = []; cbm_seconds: list[float] = []; cbm_matches: list[int] = []
    source_origins: Counter[str] = Counter(); center_error = 0.0; yaw_error = 0.0
    population_exact = True; legal_packet_times = True
    started = time.monotonic()
    for condition, index_value in enumerate(common["dataset_index"]):
        index = int(index_value); pool = dataset.pool(index); meta = dataset.meta[index]
        a, b = common["condition_offsets"][condition:condition + 2].astype(int)
        base_boxes = common["stale_boxes"][a:b].copy()
        base_features = common["stale_features"][a:b]
        raw = common["raw_scores"][a:b]; protected = common["protected"][a:b]
        ego, source, stale_remote, remote_local, ego_ok, origin, c_error, y_error = freealign.input_boxes(
            dataset, index, pool, common, condition)
        if not ego_ok:
            raise RuntimeError((role, index, "ego identity mismatch"))
        source_origins[origin] += 1; center_error = max(center_error, c_error); yaw_error = max(yaw_error, y_error)
        nominal = nominal_pose(meta)
        if meta.get("source"):
            receiver = int(meta["receiver_timestamp_us"]); sender = int(meta["source"]["source_timestamp_us"])
            legal_packet_times &= sender <= receiver

        if not len(remote_local):
            roco_estimate = nominal; roco_case = "empty_remote_identity"
            roco_elapsed = 0.0
        else:
            roco_estimate = nominal; roco_case = "estimated"; call_started = time.perf_counter()
            try:
                refined = roco.official_call(roco_module, ego.astype(np.float64),
                                             source.astype(np.float64), nominal.copy())
                roco_estimate = roco.pose3_to_matrix(refined[1])
            except Exception as exception:
                roco_case = f"exception_nominal_fallback:{type(exception).__name__}"
                roco_estimate = nominal
            roco_elapsed = time.perf_counter() - call_started
        roco_remote = apply_pose(source, roco_estimate) if roco_case == "estimated" else stale_remote.copy()

        cbm_estimate = nominal; matches = np.empty((0, 2), np.int64)
        if not len(remote_local):
            cbm_case = "empty_remote_identity"; cbm_elapsed = 0.0
        else:
            cbm_case = "estimated"; call_started = time.perf_counter()
            try:
                value = cbm_matcher(ego.astype(np.float64).copy(), source.astype(np.float64).copy(), nominal.copy())
                matches = value.detach().cpu().numpy().astype(np.int64)
                if len(matches) < 2:
                    cbm_case = "insufficient_matches_nominal_fallback"
                else:
                    cbm_estimate = cbm.planar_svd(source[matches[:, 1], :2], ego[matches[:, 0], :2])
            except Exception as exception:
                cbm_case = f"exception_nominal_fallback:{type(exception).__name__}"
                cbm_estimate = nominal
            cbm_elapsed = time.perf_counter() - call_started
        cbm_remote = apply_pose(source, cbm_estimate) if cbm_case == "estimated" else stale_remote.copy()

        if len(roco_remote) != len(remote_local) or len(cbm_remote) != len(remote_local):
            population_exact = False; raise RuntimeError((role, index, "population changed"))
        protected_boxes = base_boxes[np.flatnonzero(protected)]
        protected_scores = raw[np.flatnonzero(protected)]
        row_roco = base_boxes.copy(); row_roco[remote_local] = roco_remote
        row_cbm = base_boxes.copy(); row_cbm[remote_local] = cbm_remote
        roco_boxes.append(row_roco.astype(np.float32)); cbm_boxes.append(row_cbm.astype(np.float32))
        roco_features.append(freealign.update_features(base_features, row_roco, protected_boxes, protected_scores))
        cbm_features.append(freealign.update_features(base_features, row_cbm, protected_boxes, protected_scores))
        roco_status.append(roco_case); cbm_status.append(cbm_case)
        roco_seconds.append(roco_elapsed); cbm_seconds.append(cbm_elapsed); cbm_matches.append(len(matches))
        if (condition + 1) % 1000 == 0:
            atomic(output / f"{role}_progress.json", {"role": role, "completed": condition + 1,
                   "total": len(common["dataset_index"]), "wall_seconds": time.monotonic() - started})

    archive = {
        "condition_offsets": common["condition_offsets"], "dataset_index": common["dataset_index"],
        "sequence": common["sequence"], "wait_ms": common["wait_ms"],
        "receiver_timestamp_us": common["receiver_timestamp_us"],
        "roco_boxes": np.concatenate(roco_boxes), "roco_features": np.concatenate(roco_features),
        "cbm_boxes": np.concatenate(cbm_boxes), "cbm_features": np.concatenate(cbm_features),
        "roco_status": np.asarray(roco_status, dtype="U64"),
        "cbm_status": np.asarray(cbm_status, dtype="U64"),
        "roco_call_seconds": np.asarray(roco_seconds, np.float64),
        "cbm_call_seconds": np.asarray(cbm_seconds, np.float64),
        "cbm_match_count": np.asarray(cbm_matches, np.int32),
    }
    path = output / f"{role}_module_material.npz"; np.savez_compressed(path, **archive)
    return {"role": role, "conditions": int(len(common["dataset_index"])),
            "candidates": int(len(archive["roco_boxes"])),
            "same_population": bool(population_exact), "legal_packet_times": bool(legal_packet_times),
            "source_input_origins": dict(sorted(source_origins.items())),
            "sender_reconstruction_center_max_abs_m": center_error,
            "sender_reconstruction_yaw_max_abs_rad": yaw_error,
            "roco_status_counts": dict(sorted(Counter(roco_status).items())),
            "cbm_status_counts": dict(sorted(Counter(cbm_status).items())),
            "cbm_match_count_mean": float(np.mean(cbm_matches)),
            "roco_official_call_seconds": float(np.sum(roco_seconds)),
            "cbm_official_call_seconds": float(np.sum(cbm_seconds)),
            "material": evidence(path)}


def evaluate(dataset: Any, role: str, common: dict[str, np.ndarray], module: dict[str, np.ndarray],
             freealign_material: dict[str, np.ndarray], readout: Path, output: Path
             ) -> dict[str, Any]:
    if not (np.array_equal(common["condition_offsets"], module["condition_offsets"])
            and np.array_equal(common["dataset_index"], module["dataset_index"])
            and np.array_equal(common["condition_offsets"], freealign_material["condition_offsets"])
            and np.array_equal(common["dataset_index"], freealign_material["dataset_index"])):
        raise RuntimeError((role, "material population mismatch"))
    boxes = {"stale": common["stale_boxes"], "causal_kf": common["kf_boxes"],
             "freealign": freealign_material["freealign_boxes"],
             "roco": module["roco_boxes"], "cbm": module["cbm_boxes"]}
    features = {"stale": common["stale_features"], "causal_kf": common["kf_features"],
                "freealign": freealign_material["freealign_features"],
                "roco": module["roco_features"], "cbm": module["cbm_features"]}
    direct = freealign.direct_scores(features, readout)
    maps = load(readout / "source_maps.json")
    utility = {name: freealign.source_map(value, common["source"], maps["maps"]["direct_to_utility"])
               for name, value in direct.items()}
    aggregate = defaultdict(lambda: {"score": [], "tp50": [], "tp70": [], "gt": 0, "outputs": 0})
    saved = {(method, scoring): {"candidate": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
             for method in METHODS for scoring in READOUTS}
    for condition, index_value in enumerate(common["dataset_index"]):
        index = int(index_value); pool = dataset.pool(index); gt = pool["gt_boxes"]
        a, b = common["condition_offsets"][condition:condition + 2].astype(int)
        raw = common["raw_scores"][a:b]; source = common["source"][a:b]
        protected = common["protected"][a:b]; wait = int(common["wait_ms"][condition])
        for method in METHODS:
            for scoring in READOUTS:
                score_input = raw if scoring == "raw" else utility[method][a:b]
                ids, score = freealign.ordered(raw, source, protected, score_input, None)
                matrix = freealign.iou32(boxes[method][a:b][ids], gt)
                match50 = freealign.greedy(matrix, np.arange(len(ids)), .5)
                match70 = freealign.greedy(matrix, np.arange(len(ids)), .7)
                row = aggregate[(wait, method, scoring)]; row["score"].append(score)
                row["tp50"].append((match50 >= 0).astype(np.int8)); row["tp70"].append((match70 >= 0).astype(np.int8))
                row["gt"] += len(gt); row["outputs"] += len(ids)
                record = saved[(method, scoring)]; record["candidate"].append((a + ids).astype(np.int64))
                record["score"].append(score.astype(np.float64)); record["tp50"].append((match50 >= 0).astype(np.int8))
                record["tp70"].append((match70 >= 0).astype(np.int8)); record["offset"].append(record["offset"][-1] + len(ids))
    panels: list[dict[str, Any]] = []
    for (wait, method, scoring), row in sorted(aggregate.items()):
        panel = {"role": role, "wait_ms": wait, "method": method, "readout": scoring,
                 "gt": int(row["gt"]), "outputs": int(row["outputs"])}
        scores = np.concatenate(row["score"])
        for threshold in (50, 70):
            tp = np.concatenate(row[f"tp{threshold}"])
            metric = freealign.voc_ap(tp.tolist(), (1 - tp).tolist(), scores.tolist(), row["gt"], True)
            panel[f"ap{threshold}"] = 100.0 * metric["ap"]
            panel[f"tp{threshold}"] = int(metric["tp"]); panel[f"fp{threshold}"] = int(metric["fp"])
        panels.append(panel)
    archive: dict[str, np.ndarray] = {"method_names": np.asarray(METHODS, dtype="U24"),
                                     "readout_names": np.asarray(READOUTS, dtype="U16")}
    for (method, scoring), record in saved.items():
        prefix = f"{method}__{scoring}"
        for field in ("candidate", "score", "tp50", "tp70"):
            archive[f"{prefix}__{field}"] = np.concatenate(record[field])
        archive[f"{prefix}__offset"] = np.asarray(record["offset"], np.int64)
    prediction_path = output / f"{role}_predictions.npz"; np.savez_compressed(prediction_path, **archive)
    return {"role": role, "panels": panels, "predictions": evidence(prediction_path)}


def comparisons(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["role"], row["wait_ms"], row["method"], row["readout"]): row for row in panels}
    rows = []
    for row in panels:
        if row["method"] not in ("roco", "cbm"):
            continue
        item = dict(row)
        for reference in ("stale", "causal_kf", "freealign"):
            base = index[(row["role"], row["wait_ms"], reference, row["readout"])]
            item[f"minus_{reference}_ap50_pp"] = row["ap50"] - base["ap50"]
            item[f"minus_{reference}_ap70_pp"] = row["ap70"] - base["ap70"]
        rows.append(item)
    return rows


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    for name, item in contract["inputs"].items():
        if isinstance(item, dict) and "sha256" in item and sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(("input identity drift", name, item["path"]))
    roco_module = roco.load_roco(ROOT / contract["inputs"]["roco_source"]["path"])
    cbm_matcher = cbm.load_cbm(ROOT / contract["inputs"]["cbm_source"]["path"])
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    readout = ROOT / contract["readout_root"]
    common_root = ROOT / contract["common_material_root"]
    freealign_root = ROOT / contract["freealign_material_root"]
    role_summaries = {}; all_panels = []
    for role in contract["roles"]:
        dataset = freealign.PackedDataset(ROOT / contract["datasets"][role])
        common = load_npz(common_root / f"{role}_materialized.npz")
        material_summary = materialize(dataset, role, common, roco_module, cbm_matcher, output)
        module = load_npz(ROOT / material_summary["material"]["path"])
        fa_material = load_npz(freealign_root / f"{role}_freealign_materialized.npz")
        evaluation = evaluate(dataset, role, common, module, fa_material, readout, output)
        role_summaries[role] = {"material": material_summary, "evaluation": evaluation}
        all_panels.extend(evaluation["panels"])
    result = {"schema": "spd_current_official_roco_cbm_same_pool_summary_v1",
              "status": "complete_producer_review_pending", "created_at": contract["frozen_at"],
              "task_id": contract["task_id"], "owner": contract["owner"],
              "contract": evidence(ROOT / contract["contract_path"]),
              "identity": contract["identity"], "roles": role_summaries, "panels": all_panels,
              "comparisons": comparisons(all_panels), "input_and_weight_identity": contract["inputs"],
              "prediction_before_result": contract["prediction_before_result"],
              "actual_cost": {"wall_seconds": time.monotonic() - started,
                              "cpu_process_seconds": time.process_time(),
                              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                              "official_roco_calls": sum(v["material"]["conditions"] for v in role_summaries.values()),
                              "official_cbm_calls": sum(v["material"]["conditions"] for v in role_summaries.values()),
                              "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0,
                              "gpu_seconds": 0},
              "boundary": "Official cached box/pose module controls on exposed TraF/SPD roles; not full RoCo/CBM, hidden confirmation, feature fusion, or complete-system latency."}
    atomic(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "panels": len(all_panels),
                      "comparisons": len(result["comparisons"]), "actual_cost": result["actual_cost"]},
                     sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    contract = load(args.contract)
    if contract.get("status") != "frozen_released_for_execution": raise RuntimeError("contract not released")
    if sha256(Path(__file__)) != contract["runner"]["sha256"]: raise RuntimeError("runner identity drift")
    compute(contract, args.output)


if __name__ == "__main__":
    main()
