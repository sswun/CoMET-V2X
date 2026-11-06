#!/usr/bin/env python3
"""Formal TraF hold14/val21 optional shared-offset same-readout controls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

from audit_common_offset_peer_v1_20261026 import pairs
from e1_receiver_value_core_v1_20261020 import atomic, greedy, iou32, sha256, unique_ego_scores
from object_history_selector_core_v1_20261021 import PackedDataset
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap
from run_traf_receiver_local_source_monotone_calibration_v1_20261021 import apply_map


WAITS = (0, 100, 200, 400)
GEOMETRIES = ("stale", "cv", "huber", "supervised_constant", "kf")
GEOMETRY_CODE = {name: index for index, name in enumerate(GEOMETRIES)}
BASE_METHODS = (
    "ego",
    "stale_raw", "stale_source_u", "stale_direct_u",
    "cv_raw", "cv_source_u", "cv_direct_u",
    "cv_raw_softnms_sigma0.5", "cv_source_q_softnms_sigma0.5",
    "huber_raw", "huber_source_u", "huber_direct_u",
    "supervised_constant_raw", "supervised_constant_source_u", "supervised_constant_direct_u",
    "kf_raw", "kf_source_u", "kf_direct_u",
    "optional_raw", "optional_source_u", "optional_direct_u",
)
THRESHOLDED = (
    "cv_raw", "cv_source_u", "cv_direct_u",
    "huber_raw", "huber_source_u", "huber_direct_u",
    "kf_raw", "kf_source_u", "kf_direct_u",
    "optional_raw", "optional_source_u", "optional_direct_u",
)
BETAS = (1.0, 1.2, 1.4)
METHODS = BASE_METHODS + tuple(f"{name}_beta{beta:.1f}" for name in THRESHOLDED for beta in BETAS)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def checked_dataset(path: Path) -> tuple[PackedDataset, dict]:
    summary = json.loads((path / "summary.json").read_text())
    assert sha256(path / "pools.npz") == summary["pools_sha256"]
    assert sha256(path / "conditions.jsonl") == summary["conditions_sha256"]
    return PackedDataset(path), summary


def identities(pool: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kind = pool["hypothesis_type"]
    ego = np.flatnonzero(kind == 0); stale = np.flatnonzero(kind == 1)
    propagated = np.flatnonzero(kind == 2)
    lookup = {int(pool["source_detection_indices"][j]): int(j) for j in propagated}
    chosen = np.asarray([lookup.get(int(pool["source_detection_indices"][j]), int(j))
                         for j in stale], np.int64)
    assert np.array_equal(pool["source_detection_indices"][stale],
                          pool["source_detection_indices"][chosen])
    return ego, stale, chosen


def update_features(base: np.ndarray, boxes: np.ndarray, protected: np.ndarray,
                    protected_scores: np.ndarray) -> np.ndarray:
    out = base.astype(np.float32).copy()
    if not len(out):
        return out
    out[:, 2:5] = boxes[:, :3]; out[:, 5:8] = boxes[:, 3:6]
    out[:, 8] = np.sin(boxes[:, 6]); out[:, 9] = np.cos(boxes[:, 6])
    out[:, 10] = np.linalg.norm(boxes[:, :2], axis=1); out[:, 11] = np.abs(boxes[:, 1])
    if len(protected):
        delta = boxes[:, None, :2] - protected[None, :, :2]
        distance = np.linalg.norm(delta, axis=2); nearest = distance.argmin(axis=1)
        overlap = iou32(boxes, protected); rows = np.arange(len(boxes))
        out[:, 24] = distance[rows, nearest]
        out[:, 25] = np.abs(delta[rows, nearest, 0]); out[:, 26] = np.abs(delta[rows, nearest, 1])
        out[:, 27] = overlap.max(axis=1); out[:, 28] = protected_scores[nearest]
        out[:, 29] = (distance < 10.0).sum(axis=1)
    else:
        out[:, 24:30] = np.asarray([200., 200., 200., 0., 0., 0.], np.float32)
    return out


def source_map(score: np.ndarray, source: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    out = np.empty(len(score), np.float64)
    for source_id, source_name in enumerate(("ego", "remote")):
        use = source == source_id; out[use] = apply_map(score[use], spec[source_name])
    return out


def direct_scores(features: dict[str, np.ndarray], readout: Path) -> dict[str, np.ndarray]:
    import torch
    from torch import nn
    schedule = load_npz(readout / "training_schedule.npz")
    model = nn.Sequential(nn.Linear(39, 32), nn.ReLU(), nn.Linear(32, 1))
    model.load_state_dict(torch.load(readout / "direct_u.pth", map_location="cpu"), strict=True)
    model.eval(); result = {}
    with torch.no_grad():
        for name, matrix in features.items():
            x = ((matrix - schedule["mean"]) / schedule["std"]).astype(np.float32)
            rows = []
            for start in range(0, len(x), 65536):
                rows.append(torch.sigmoid(model(torch.from_numpy(x[start:start + 65536])).squeeze(-1)).numpy())
            result[name] = np.concatenate(rows).astype(np.float64)
    return result


def materialize(dataset: PackedDataset, role: str, huber: np.ndarray,
                supervised: np.ndarray) -> dict[str, np.ndarray]:
    selected = [i for i, row in enumerate(dataset.meta) if row["role"] == role]
    selected.sort(key=lambda i: (str(dataset.meta[i]["sequence"]), int(dataset.meta[i]["wait_ms"]),
                                 int(dataset.meta[i]["receiver_timestamp_us"])))
    arrays = defaultdict(list); offsets = [0]; states = {}; last = {}
    sequence = []; waits = []; timestamps = []; indices = []; trace = []
    for count, index in enumerate(selected):
        meta = dataset.meta[index]; pool = dataset.pool(index)
        key = (str(meta["sequence"]), int(meta["wait_ms"])); timestamp = int(meta["receiver_timestamp_us"])
        assert timestamp > last.get(key, -1); last[key] = timestamp
        ego, stale_id, cv_id = identities(pool)
        ego_boxes = pool["boxes"][ego].copy(); stale = pool["boxes"][stale_id].copy()
        cv = pool["boxes"][cv_id].copy(); ego_scores = pool["raw_scores"][ego].astype(np.float64)
        remote_scores = pool["raw_scores"][stale_id].astype(np.float64)
        protected_local = np.flatnonzero(pool["is_protected"][ego])
        protected_boxes = ego_boxes[protected_local]; protected_scores = ego_scores[protected_local]
        anchor_remote, anchor_ego = pairs(cv, protected_boxes, 2.5, True)
        residual = protected_boxes[anchor_ego, :2].astype(np.float64) - cv[anchor_remote, :2]
        mean, variance = states.get(key, (np.zeros(2), 1.0)); prior = variance + .15 ** 2
        if len(residual):
            gain = prior / (prior + .6 ** 2 / len(residual))
            mean = mean + gain * (residual.mean(axis=0) - mean); variance = (1.0 - gain) * prior
        else:
            variance = prior
        states[key] = (mean.copy(), float(variance))
        remote_geometry = {
            "stale": stale,
            "cv": cv,
            "huber": cv.copy(),
            "supervised_constant": cv.copy(),
            "kf": cv.copy(),
        }
        remote_geometry["huber"][:, :2] += huber.astype(np.float32)
        remote_geometry["supervised_constant"][:, :2] += supervised.astype(np.float32)
        remote_geometry["kf"][:, :2] += mean.astype(np.float32)
        source = np.r_[np.zeros(len(ego), np.int8), np.ones(len(stale_id), np.int8)]
        protected = np.r_[pool["is_protected"][ego], np.zeros(len(stale_id), np.int8)]
        anchor = np.zeros(len(source), np.int8); anchor[len(ego) + anchor_remote] = 1
        score = np.r_[ego_scores, remote_scores]
        source_id = np.r_[pool["source_detection_indices"][ego],
                          pool["source_detection_indices"][stale_id]].astype(np.int32)
        for name in GEOMETRIES:
            remote = remote_geometry[name]
            base_feature = pool["features"][stale_id] if name == "stale" else pool["features"][cv_id]
            remote_feature = update_features(base_feature, remote, protected_boxes, protected_scores)
            arrays[name + "_boxes"].append(np.concatenate((ego_boxes, remote)).astype(np.float32))
            arrays[name + "_features"].append(np.concatenate((pool["features"][ego], remote_feature)).astype(np.float32))
        arrays["raw_scores"].append(score); arrays["source"].append(source)
        arrays["protected"].append(protected); arrays["anchor"].append(anchor)
        arrays["source_id"].append(source_id)
        offsets.append(offsets[-1] + len(source)); sequence.append(str(meta["sequence"])); waits.append(int(meta["wait_ms"]))
        timestamps.append(timestamp); indices.append(index)
        trace.append([index, len(anchor_remote), variance, mean[0], mean[1], len(ego), len(stale_id)])
        if (count + 1) % 2000 == 0:
            print(json.dumps({"stage": f"materialize_{role}", "completed": count + 1,
                              "total": len(selected)}), flush=True)
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    result.update(condition_offsets=np.asarray(offsets, np.int64), dataset_index=np.asarray(indices, np.int64),
                  sequence=np.asarray(sequence, dtype="U16"), wait_ms=np.asarray(waits, np.int16),
                  receiver_timestamp_us=np.asarray(timestamps, np.int64), trace=np.asarray(trace, np.float64),
                  trace_columns=np.asarray(["dataset_index", "anchors", "kf_variance", "kf_dx", "kf_dy",
                                            "ego_candidates", "remote_candidates"], dtype="U32"))
    return result


def gaussian_soft_nms(boxes: np.ndarray, protected: np.ndarray, initial: np.ndarray) -> np.ndarray:
    fixed = np.flatnonzero(protected == 1); variable = np.flatnonzero(protected == 0)
    assigned = initial.astype(np.float64).copy()
    if not len(variable):
        return assigned
    active = initial[variable].astype(np.float64).copy()
    if len(fixed):
        overlap = iou32(boxes[variable], boxes[fixed]).max(axis=1, initial=0.0).astype(np.float64)
        active *= np.exp(-(overlap ** 2) / .5)
    remaining = list(range(len(variable)))
    while remaining:
        best = max(range(len(remaining)), key=lambda k: (active[remaining[k]], -remaining[k]))
        local = remaining.pop(best); assigned[variable[local]] = active[local]
        if remaining:
            ids = np.asarray(remaining, np.int32)
            overlap = iou32(boxes[variable[ids]], boxes[variable[[local]]])[:, 0].astype(np.float64)
            active[ids] *= np.exp(-(overlap ** 2) / .5)
    return assigned


def family(method: str) -> str:
    base = method.split("_beta", 1)[0]
    if base.endswith("direct_u"): return "direct_u"
    if base.endswith("source_u"): return "source_u"
    return "raw"


def ordered(raw: np.ndarray, source: np.ndarray, protected: np.ndarray, score: np.ndarray,
            threshold_spec: dict | None) -> tuple[np.ndarray, np.ndarray]:
    ego = np.flatnonzero(source == 0)
    if threshold_spec is not None and threshold_spec["strategy"] == "ego_fallback":
        rank = unique_ego_scores(raw[ego]); order = np.argsort(-rank, kind="stable")
        return ego[order], rank[order]
    fixed = np.flatnonzero(protected == 1); variable = np.flatnonzero(protected == 0)
    variable_score = score[variable]
    if threshold_spec is not None:
        keep = variable_score >= float(threshold_spec["threshold"])
        variable = variable[keep]; variable_score = variable_score[keep]
    fixed_rank = 2.0 + unique_ego_scores(raw[fixed], stable=True)
    variable_rank = unique_ego_scores(variable_score, stable=True)
    ids = np.r_[fixed, variable]; rank = np.r_[fixed_rank, variable_rank]
    order = np.argsort(-rank, kind="stable")
    return ids[order], rank[order]


def evaluate_role(dataset: PackedDataset, role: str, data: dict[str, np.ndarray], direct: dict[str, np.ndarray],
                  maps: dict, calibration: dict, output: Path) -> dict:
    source_u_all = source_map(data["raw_scores"], data["source"], maps["maps"]["raw_to_utility"])
    source_q_all = source_map(data["raw_scores"], data["source"], maps["maps"]["raw_to_quality"])
    direct_u_all = {name: source_map(direct[name], data["source"], maps["maps"]["direct_to_utility"])
                    for name in GEOMETRIES}
    results = defaultdict(lambda: {name: {"scores": [], "tp50": [], "tp70": [], "gt": 0,
                                                "outputs": 0, "remote": 0, "anchors": 0,
                                                "selected_geometry": np.zeros(len(GEOMETRIES), np.int64)}
                                  for name in METHODS})
    saved = {name: {"candidate": [], "geometry": [], "score": [], "tp50": [], "tp70": [], "offset": [0]}
             for name in METHODS}
    condition_outputs = np.zeros((len(METHODS), len(data["dataset_index"])), np.int32)
    condition_tp70 = np.zeros_like(condition_outputs); condition_fp70 = np.zeros_like(condition_outputs)
    optional_choice = np.full((len(data["raw_scores"]), 3), -1, np.int8)
    for condition, index in enumerate(data["dataset_index"]):
        pool = dataset.pool(int(index)); a, b = data["condition_offsets"][condition:condition + 2].astype(int)
        raw = data["raw_scores"][a:b]; source = data["source"][a:b]
        protected = data["protected"][a:b]; anchor = data["anchor"][a:b]
        boxes = {name: data[name + "_boxes"][a:b] for name in GEOMETRIES}
        direct_u = {name: direct_u_all[name][a:b] for name in GEOMETRIES}
        source_u = source_u_all[a:b]; source_q = source_q_all[a:b]
        remote = np.flatnonzero(source == 1)
        optional_codes = {
            "raw": np.full(len(raw), GEOMETRY_CODE["cv"], np.int8),
            "source_u": np.full(len(raw), GEOMETRY_CODE["cv"], np.int8),
            "direct_u": np.full(len(raw), GEOMETRY_CODE["cv"], np.int8),
        }
        if len(remote):
            tie_order = ("cv", "stale", "kf")
            matrix = np.stack([direct_u[name][remote] for name in tie_order], axis=1)
            chosen = np.argmax(matrix, axis=1)
            optional_codes["direct_u"][remote] = np.asarray([GEOMETRY_CODE[tie_order[j]] for j in chosen], np.int8)
        for column, name in enumerate(("raw", "source_u", "direct_u")):
            optional_choice[a:b, column] = optional_codes[name]

        wait = int(data["wait_ms"][condition]); gt = pool["gt_boxes"]
        for method_index, method in enumerate(METHODS):
            base, _, beta_text = method.partition("_beta")
            threshold = None
            if beta_text:
                calibration_name = {"raw": "protected_raw", "source_u": "protected_source_u",
                                    "direct_u": "protected_direct_u"}[family(base)]
                threshold = calibration["methods"][calibration_name]["working_points"][beta_text]
            if base == "ego":
                geom_code = np.full(len(raw), GEOMETRY_CODE["cv"], np.int8)
                selected_boxes = boxes["cv"]
                ids = np.flatnonzero(source == 0); rank = unique_ego_scores(raw[ids])
                order = np.argsort(-rank, kind="stable"); ids, rank = ids[order], rank[order]
            elif "softnms" in base:
                geom_code = np.full(len(raw), GEOMETRY_CODE["cv"], np.int8)
                selected_boxes = boxes["cv"]
                initial = raw if base.startswith("cv_raw") else source_q
                soft = gaussian_soft_nms(selected_boxes, protected, initial)
                ids, rank = ordered(raw, source, protected, soft, None)
            else:
                prefix = base.split("_", 1)[0]
                if prefix == "supervised": prefix = "supervised_constant"
                score_name = family(base)
                if prefix == "optional":
                    geom_code = optional_codes[score_name]
                    selected_boxes = np.empty_like(boxes["cv"])
                    score = np.empty(len(raw), np.float64)
                    for code, name in enumerate(GEOMETRIES):
                        use = geom_code == code; selected_boxes[use] = boxes[name][use]
                        if score_name == "direct_u": score[use] = direct_u[name][use]
                    if score_name == "raw": score = raw
                    elif score_name == "source_u": score = source_u
                else:
                    geom_code = np.full(len(raw), GEOMETRY_CODE[prefix], np.int8)
                    selected_boxes = boxes[prefix]
                    score = raw if score_name == "raw" else source_u if score_name == "source_u" else direct_u[prefix]
                ids, rank = ordered(raw, source, protected, score, threshold)
            eval_boxes = selected_boxes[ids]; ious = iou32(eval_boxes, gt)
            match50 = greedy(ious, np.arange(len(ids)), .5); match70 = greedy(ious, np.arange(len(ids)), .7)
            row = results[wait][method]
            row["scores"].append(rank); row["tp50"].append((match50 >= 0).astype(np.int8))
            row["tp70"].append((match70 >= 0).astype(np.int8)); row["gt"] += len(gt)
            row["outputs"] += len(ids); row["remote"] += int((source[ids] == 1).sum())
            row["anchors"] += int(((source[ids] == 1) & (anchor[ids] == 1)).sum())
            for code in range(len(GEOMETRIES)):
                row["selected_geometry"][code] += int(((source[ids] == 1) & (geom_code[ids] == code)).sum())
            record = saved[method]
            record["candidate"].append((a + ids).astype(np.int64)); record["geometry"].append(geom_code[ids])
            record["score"].append(rank); record["tp50"].append((match50 >= 0).astype(np.int8))
            record["tp70"].append((match70 >= 0).astype(np.int8)); record["offset"].append(record["offset"][-1] + len(ids))
            condition_outputs[method_index, condition] = len(ids)
            condition_tp70[method_index, condition] = int((match70 >= 0).sum())
            condition_fp70[method_index, condition] = int((match70 < 0).sum())
        if (condition + 1) % 1000 == 0:
            print(json.dumps({"stage": f"evaluate_{role}", "completed": condition + 1,
                              "total": len(data["dataset_index"])}), flush=True)

    panels = []
    for wait in WAITS:
        for method in METHODS:
            row = results[wait][method]; scores = np.concatenate(row["scores"])
            panel = {"wait_ms": wait, "method": method, "outputs": row["outputs"],
                     "remote": row["remote"], "anchors": row["anchors"],
                     "selected_geometry": {name: int(row["selected_geometry"][code])
                                           for name, code in GEOMETRY_CODE.items()}}
            for threshold in (50, 70):
                tp = np.concatenate(row[f"tp{threshold}"])
                value = voc_ap(tp.tolist(), (1 - tp).tolist(), scores.tolist(), row["gt"], True)
                panel[f"ap{threshold}"] = value["ap"] * 100.0
                if threshold == 70: panel.update(tp=value["tp"], fp=value["fp"], gt=value["gt"])
            panels.append(panel)
    archive = {"method_names": np.asarray(METHODS, dtype="U48")}
    for method, record in saved.items():
        for key in ("candidate", "geometry", "score", "tp50", "tp70"):
            archive[f"{method}__{key}"] = np.concatenate(record[key])
        archive[f"{method}__offset"] = np.asarray(record["offset"], np.int64)
    archive["condition_outputs"] = condition_outputs; archive["condition_tp70"] = condition_tp70
    archive["condition_fp70"] = condition_fp70; archive["dataset_index"] = data["dataset_index"]
    np.savez_compressed(output / f"{role}_predictions.npz", **archive)
    np.savez_compressed(output / f"{role}_optional_choices.npz", choices=optional_choice,
                        columns=np.asarray(["raw", "source_u", "direct_u"], dtype="U16"),
                        geometry_names=np.asarray(GEOMETRIES, dtype="U24"),
                        condition_offsets=data["condition_offsets"], dataset_index=data["dataset_index"],
                        source=data["source"], anchor=data["anchor"], source_id=data["source_id"])
    return {"role": role, "conditions": len(data["dataset_index"]),
            "method_conditions": len(data["dataset_index"]) * len(METHODS), "methods": list(METHODS),
            "panels": panels, "predictions_sha256": sha256(output / f"{role}_predictions.npz"),
            "optional_choices_sha256": sha256(output / f"{role}_optional_choices.npz")}


def save_material(path: Path, data: dict[str, np.ndarray]) -> str:
    np.savez_compressed(path, **data); return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); contract = json.loads(args.contract.read_text())
    assert contract["status"] == "frozen_after_qualification_before_formal_outcomes"
    review = args.workspace / contract["qualification"]["review"]
    assert sha256(review) == contract["qualification"]["review_sha256"]
    assert json.loads(review.read_text())["status"] == "passed"
    readout = args.workspace / "artifacts/research_autopilot/status_audit_20261026_1905/shared_offset_followups_v1/shared_offset_readout_task_tolerance_v2"
    strong = args.workspace / "artifacts/research_autopilot/status_audit_20261026_1905/shared_offset_followups_v1/traf_hold_strong_controls_v1"
    for filename, expected in (
        (readout / "direct_u.pth", contract["frozen_readout_artifacts"]["direct_u_weights_sha256"]),
        (readout / "training_schedule.npz", contract["frozen_readout_artifacts"]["training_schedule_sha256"]),
        (readout / "source_maps.json", contract["frozen_readout_artifacts"]["source_maps_sha256"]),
        (readout / "calibration.json", contract["frozen_readout_artifacts"]["calibration_sha256"]),
        (strong / "label_free_fit.npz", contract["frozen_readout_artifacts"]["label_free_fit_sha256"]),
    ):
        assert sha256(filename) == expected
    maps = json.loads((readout / "source_maps.json").read_text())
    calibration = json.loads((readout / "calibration.json").read_text())
    fit = load_npz(strong / "label_free_fit.npz"); huber = fit["huber"]; supervised = fit["supervised_constant"]
    train, train_summary = checked_dataset(args.workspace / "artifacts/research_autopilot/status_audit_20261021_1200/object_history_selector_v1/train_dataset")
    val, val_summary = checked_dataset(args.workspace / "artifacts/research_autopilot/status_audit_20261021_1200/object_history_selector_v1/val_dataset")
    data = {
        "mechanism_holdout14": materialize(train, "mechanism_holdout14", huber, supervised),
        "val21": materialize(val, "val21", huber, supervised),
    }
    material_hash = {role: save_material(args.output / f"{role}_materialized.npz", value)
                     for role, value in data.items()}
    direct = {role: direct_scores({name: value[name + "_features"] for name in GEOMETRIES}, readout)
              for role, value in data.items()}
    result = {
        role: evaluate_role(train if role == "mechanism_holdout14" else val, role, data[role],
                            direct[role], maps, calibration, args.output)
        for role in ("mechanism_holdout14", "val21")
    }
    summary = {
        "schema": "shared_offset_optional_traf_formal_summary_v1",
        "status": "complete_producer_review_pending",
        "task_id": contract["task_id"], "owner": contract["owner"],
        "contract": str(args.contract), "contract_sha256": sha256(args.contract),
        "input_identity": {"train_pools": train_summary["pools_sha256"],
                           "train_conditions": train_summary["conditions_sha256"],
                           "val_pools": val_summary["pools_sha256"],
                           "val_conditions": val_summary["conditions_sha256"]},
        "materialized_sha256": material_hash, "roles": result,
        "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0, "gpu_seconds": 0,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
        "boundary": contract["claim_boundary"],
    }
    atomic(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "roles"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
