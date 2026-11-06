#!/usr/bin/env python3
"""Cal8-only sourcewise monotone calibration for receiver-local controls.

This is the released score-scale correction from the 2026-10-21 receiver-local
confirmation package.  It reuses frozen candidate pools and frozen head scores,
fits one positive-slope logistic map per method and source on calibration8, and
then freezes the maps for mechanism-holdout14 and val21.  It performs no model
forward, detector update, candidate filtering, or validation-set selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from e1_receiver_value_core_v1_20261020 import (
    PackedDataset, append, atomic, error_counts, greedy, load_npz, metrics,
    pool_order, sha256,
)


METHODS = (
    "base_u_all",
    "local_u_all",
    "local_q_all",
    "source_gate_u",
    "source_gate_q",
)
BETAS = (1.0, 1.2, 1.4)
EPS = 1e-6


def progress(out: Path, stage: str, **extra: Any) -> None:
    value = {
        "stage": stage,
        "updated_at": time.time(),
        "new_detector_or_backbone_forwards": 0,
        "detector_updates": 0,
        **extra,
    }
    atomic(out / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def roles(dataset: PackedDataset) -> np.ndarray:
    result = np.empty(len(dataset.data["features"]), dtype="U24")
    for row in dataset.meta:
        result[int(row["start"]):int(row["stop"])] = row["role"]
    assert np.all(result != "")
    return result


def compose_scores(e1: dict[str, np.ndarray], token: dict[str, np.ndarray],
                   local_q: np.ndarray, remote: np.ndarray) -> dict[str, np.ndarray]:
    base_u = e1["direct_utility"].astype(np.float64)
    local_u = token["token_local_direct"].astype(np.float64)
    local_q = local_q.astype(np.float64)
    assert base_u.shape == local_u.shape == local_q.shape == remote.shape
    result = {
        "base_u_all": base_u,
        "local_u_all": local_u,
        "local_q_all": local_q,
        "source_gate_u": np.where(remote, base_u, local_u),
        "source_gate_q": np.where(remote, base_u, local_q),
    }
    assert all(np.isfinite(value).all() and np.all((value >= 0) & (value <= 1))
               for value in result.values())
    return result


def fit_positive_logistic(raw_score: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    x = np.log(np.clip(raw_score.astype(np.float64), EPS, 1 - EPS) /
               np.clip(1 - raw_score.astype(np.float64), EPS, 1 - EPS))
    y = target.astype(np.float64)
    assert len(x) and np.isfinite(x).all() and set(np.unique(y)).issubset({0.0, 1.0})

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        log_slope, intercept = float(theta[0]), float(theta[1])
        slope = np.exp(np.clip(log_slope, -12, 12))
        z = np.clip(slope * x + intercept, -40, 40)
        # BCE in its stable logaddexp form.
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z) + 1e-8 * np.dot(theta, theta))
        probability = 1.0 / (1.0 + np.exp(-z))
        residual = probability - y
        gradient = np.asarray([
            np.mean(residual * slope * x) + 2e-8 * log_slope,
            np.mean(residual) + 2e-8 * intercept,
        ], dtype=np.float64)
        return loss, gradient

    fit = minimize(lambda t: objective(t)[0], np.zeros(2, np.float64),
                   jac=lambda t: objective(t)[1], method="L-BFGS-B",
                   bounds=((-12.0, 12.0), (-40.0, 40.0)),
                   options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9})
    loss, gradient = objective(fit.x)
    slope = float(np.exp(float(fit.x[0])))
    assert fit.success and slope > 0 and np.isfinite([loss, slope, fit.x[1]]).all()
    return {
        "slope": slope,
        "intercept": float(fit.x[1]),
        "fit_rows": int(len(x)),
        "positive": int(y.sum()),
        "positive_fraction": float(y.mean()),
        "loss": loss,
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "optimizer_status": int(fit.status),
        "optimizer_message": str(fit.message),
        "iterations": int(fit.nit),
        "strictly_positive_slope": True,
    }


def apply_map(raw_score: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    x = np.log(np.clip(raw_score.astype(np.float64), EPS, 1 - EPS) /
               np.clip(1 - raw_score.astype(np.float64), EPS, 1 - EPS))
    z = np.clip(float(fit["slope"]) * x + float(fit["intercept"]), -40, 40)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64)


def fit_maps(dataset: PackedDataset, scores: dict[str, np.ndarray]) -> dict[str, Any]:
    role = roles(dataset)
    source = dataset.data["is_remote"].astype(np.int8)
    variable = dataset.data["is_protected"].astype(np.int8) == 0
    target = dataset.data["utility_y"].astype(np.int8)
    result: dict[str, Any] = {
        "schema": "traf_receiver_local_source_monotone_calibration_maps_v1",
        "status": "complete",
        "fit_population": "calibration8_variable_candidates",
        "target": "utility_y_current_IoU70_not_already_covered_by_fixed_S",
        "maps": {},
    }
    for method in METHODS:
        result["maps"][method] = {}
        for source_id, name in enumerate(("ego", "remote")):
            mask = (role == "calibration8") & variable & (source == source_id)
            result["maps"][method][name] = fit_positive_logistic(scores[method][mask], target[mask])
    return result


def apply_maps(scores: dict[str, np.ndarray], source: np.ndarray,
               maps: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for method in METHODS:
        value = np.empty_like(scores[method], dtype=np.float64)
        for source_id, name in enumerate(("ego", "remote")):
            mask = source == source_id
            value[mask] = apply_map(scores[method][mask], maps["maps"][method][name])
        result[method] = value
    return result


def calibrate_workpoints(dataset: PackedDataset, scores: dict[str, np.ndarray]) -> dict[str, Any]:
    ego_tp = ego_fp = 0
    collected = {name: {"fixed_tp": 0, "fixed_fp": 0, "scores": [], "tp": [], "remote": []}
                 for name in METHODS}
    for condition, metadata in enumerate(dataset.meta):
        if metadata["role"] != "calibration8":
            continue
        pool, score_pool = dataset.pool(condition), dataset.score_pool(condition, scores)
        ego_order, _ = pool_order(pool, score_pool, "ego")
        ego_match = greedy(pool["ious"], ego_order, 0.7)
        ego_tp += int((ego_match >= 0).sum()); ego_fp += int((ego_match < 0).sum())
        for method in METHODS:
            order, _ = pool_order(pool, score_pool, method)
            match = greedy(pool["ious"], order, 0.7)
            fixed = pool["is_protected"][order] == 1
            item = collected[method]
            item["fixed_tp"] += int(((match >= 0) & fixed).sum())
            item["fixed_fp"] += int(((match < 0) & fixed).sum())
            item["scores"].append(score_pool[method][order[~fixed]])
            item["tp"].append((match[~fixed] >= 0).astype(np.int8))
            item["remote"].append(pool["is_remote"][order[~fixed]])
    result: dict[str, Any] = {
        "status": "complete", "population": "calibration8_all_frames_four_waits",
        "ego_tp": ego_tp, "ego_fp": ego_fp, "methods": {},
    }
    for method, item in collected.items():
        score = np.concatenate(item["scores"]); tp = np.concatenate(item["tp"])
        remote = np.concatenate(item["remote"])
        order = np.argsort(-score, kind="stable")
        score, tp, remote = score[order], tp[order], remote[order]
        ends = np.r_[np.flatnonzero(np.diff(score) != 0), len(score) - 1]
        ctp, cfp, cremote = np.cumsum(tp), np.cumsum(1 - tp), np.cumsum(remote)
        method_result: dict[str, Any] = {
            "fixed_tp": item["fixed_tp"], "fixed_fp": item["fixed_fp"], "working_points": {},
        }
        for beta in BETAS:
            budget = int(np.floor(beta * ego_fp))
            chosen = {"strategy": "ego_fallback", "threshold": None, "tp": ego_tp,
                      "fp": ego_fp, "remote": 0}
            best = (ego_tp, -ego_fp, 0, 1, float("inf"))
            options = [(-1, None, item["fixed_tp"], item["fixed_fp"], 0)]
            options += [(int(j), float(score[j]), int(item["fixed_tp"] + ctp[j]),
                         int(item["fixed_fp"] + cfp[j]), int(cremote[j]))
                        for j in ends if item["fixed_fp"] + cfp[j] <= budget]
            for j, threshold, total_tp, total_fp, nremote in options:
                key = (total_tp, -total_fp, -nremote, 0,
                       float("inf") if threshold is None else threshold)
                if total_fp <= budget and key > best:
                    best = key
                    chosen = {"strategy": "empty_C" if j < 0 else "threshold",
                              "threshold": threshold, "tp": total_tp, "fp": total_fp,
                              "remote": nremote}
            chosen.update(total_fp_budget=budget,
                          variable_fp_budget=budget - item["fixed_fp"], beta=beta)
            method_result["working_points"][str(beta)] = chosen
        result["methods"][method] = method_result
    return result


def evaluate(dataset: PackedDataset, raw_scores: dict[str, np.ndarray],
             calibrated_scores: dict[str, np.ndarray], workpoints: dict[str, Any],
             out: Path, role: str) -> Path:
    destination = out / ("val_evaluation" if role == "val21" else "hold_evaluation")
    destination.mkdir(exist_ok=False)
    variants: list[tuple[str, str, dict[str, Any] | None, bool]] = []
    for method in METHODS:
        variants.append((method + "_raw", method, None, False))
        variants.append((method + "_calibrated", method, None, True))
        for beta in BETAS:
            variants.append((f"{method}_calibrated_beta{beta:.1f}", method,
                             workpoints["methods"][method]["working_points"][str(beta)], True))
    stats = defaultdict(lambda: {"scores": [], "tp50": [], "tp70": [], "gt": 0,
                                 "frames": 0, "rescued": 0, "destroyed": 0,
                                 "remote": 0, "ego": 0, "duplicate": 0,
                                 "localization": 0, "background": 0})
    records = 0; started = time.monotonic()
    with (destination / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for condition, metadata in enumerate(dataset.meta):
            if metadata["role"] != role:
                continue
            pool = dataset.pool(condition)
            raw_pool = dataset.score_pool(condition, raw_scores)
            calibrated_pool = dataset.score_pool(condition, calibrated_scores)
            ego_order, _ = pool_order(pool, raw_pool, "ego")
            ego_match = greedy(pool["ious"], ego_order, 0.7)
            ego_set = set(int(value) for value in ego_match if value >= 0)
            for label, method, working_point, calibrated in variants:
                score_pool = calibrated_pool if calibrated else raw_pool
                threshold = None; fallback = False
                if working_point:
                    fallback = working_point["strategy"] == "ego_fallback"
                    threshold = (float("inf") if working_point["strategy"] == "empty_C"
                                 else working_point["threshold"])
                order, rank = pool_order(pool, score_pool, method, threshold, fallback)
                match50 = greedy(pool["ious"], order, 0.5)
                match70 = greedy(pool["ious"], order, 0.7)
                matched = set(int(value) for value in match70 if value >= 0)
                errors = error_counts(pool["ious"], order, match70)
                for key in (("all", metadata["wait_ms"], label),
                            (metadata["sequence"], metadata["wait_ms"], label)):
                    item = stats[key]
                    item["scores"].append(rank)
                    item["tp50"].append((match50 >= 0).astype(np.int8))
                    item["tp70"].append((match70 >= 0).astype(np.int8))
                    item["gt"] += len(pool["gt_boxes"]); item["frames"] += 1
                    item["rescued"] += len(matched - ego_set)
                    item["destroyed"] += len(ego_set - matched)
                    item["remote"] += int(pool["is_remote"][order].sum())
                    item["ego"] += int((pool["is_remote"][order] == 0).sum())
                    item["duplicate"] += errors["duplicate_fp"]
                    item["localization"] += errors["localization_fp"]
                    item["background"] += errors["background_fp"]
                append(stream, {
                    "condition_index": condition, "sequence": metadata["sequence"],
                    "global_index": metadata["global_index"], "wait_ms": metadata["wait_ms"],
                    "method": label, "selected_pool_ids": order.astype(int).tolist(),
                    "scores": rank.astype(float).tolist(),
                    "matched_gt50": match50.astype(int).tolist(),
                    "matched_gt70": match70.astype(int).tolist(),
                    "rescued": len(matched - ego_set), "destroyed": len(ego_set - matched),
                    "errors": errors,
                })
                records += 1
    panels = []
    for (sequence, wait, method), item in sorted(stats.items()):
        rank = np.concatenate(item.pop("scores")); tp50 = np.concatenate(item.pop("tp50"))
        tp70 = np.concatenate(item.pop("tp70"))
        row = dict(item, sequence=sequence, wait_ms=wait, method=method,
                   ap50=metrics(tp50, rank, item["gt"])["ap"],
                   ap70=metrics(tp70, rank, item["gt"])["ap"],
                   tp70=int(tp70.sum()), fp70=int((1 - tp70).sum()))
        if sequence == "all":
            order = np.argsort(-rank, kind="stable")
            cumulative_tp = np.cumsum(tp70[order]); cumulative_fp = np.cumsum(1 - tp70[order])
            row["fixed_fp70"] = {}
            for budget in (2558, 5116, 6137):
                eligible = np.flatnonzero(cumulative_fp <= budget)
                j = int(eligible[-1]) if len(eligible) else -1
                row["fixed_fp70"][str(budget)] = {
                    "tp": int(cumulative_tp[j]) if j >= 0 else 0,
                    "fp": int(cumulative_fp[j]) if j >= 0 else 0,
                    "role": "posthoc_GT_diagnostic",
                }
        panels.append(row)
    atomic(destination / "summary.json", {
        "schema": "traf_receiver_local_source_monotone_evaluation_v1",
        "status": "complete_review_pending", "role": role,
        "records": records, "panels": panels,
        "predictions_sha256": sha256(destination / "predictions.jsonl"),
        "dataset_sha256": sha256(dataset.directory / "summary.json"),
        "wall_seconds": time.monotonic() - started,
        "boundary": "Cal8-only positive-slope sourcewise calibration of frozen scores; no candidate geometry, detector, split or val selection change.",
    })
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--equal-info", required=True, type=Path)
    parser.add_argument("--token-head", required=True, type=Path)
    parser.add_argument("--e1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, release_path, equal_info, token_head, e1, out = (
        value.resolve() for value in (args.workspace, args.release, args.equal_info,
                                      args.token_head, args.e1, args.output)
    )
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert "embodied-research.slice" in Path("/proc/self/cgroup").read_text()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        release = json.loads(release_path.read_text())
        assert release["status"] == "released_for_autonomous_execution"
        assert any(stage["id"] == "receiver_local_equal_information_controls_20261021"
                   for stage in release["stages"])
        assert "complete_independent_review_passed" in json.loads(
            (equal_info / "completion_state_v1.json").read_text())["status"]
        token_completion = json.loads((token_head / "completion_state_v1.json").read_text())
        assert "complete" in token_completion["status"]
        # E1 predates the completion-state publisher and is accepted through
        # its independently reviewed summary instead.
        e1_summary = json.loads((e1 / "summary.json").read_text())
        assert e1_summary["status"] == "complete_review_passed"
        snapshot = out / "code_snapshots" / str(time.time_ns())
        snapshot.mkdir(parents=True)
        shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
        identity = {
            "schema": "traf_receiver_local_source_monotone_identity_v1",
            "release": str(release_path.relative_to(root)), "release_sha256": sha256(release_path),
            "equal_info": str(equal_info.relative_to(root)),
            "equal_info_completion_sha256": sha256(equal_info / "completion_state_v1.json"),
            "token_head": str(token_head.relative_to(root)),
            "token_head_completion_sha256": sha256(token_head / "completion_state_v1.json"),
            "e1": str(e1.relative_to(root)), "e1_summary_sha256": sha256(e1 / "summary.json"),
            "fit_role": "calibration8", "hold_role": "mechanism_holdout14", "eval_role": "val21",
            "fit_target": "utility_y", "methods": list(METHODS),
            "selection_used_hold_or_val": False, "new_detector_or_backbone_forwards": 0,
            "detector_updates": 0, "code_snapshot": str(snapshot.relative_to(root)),
        }
        atomic(out / "execution_identity.json", identity)
        train = PackedDataset(e1 / "train_dataset"); val = PackedDataset(e1 / "val_dataset")
        train_scores = compose_scores(load_npz(e1 / "train_scores.npz"),
                                      load_npz(token_head / "train_scores.npz"),
                                      load_npz(equal_info / "local_q_scores.npz")["train"],
                                      train.data["is_remote"].astype(bool))
        val_scores = compose_scores(load_npz(e1 / "val_scores.npz"),
                                    load_npz(token_head / "val_scores.npz"),
                                    load_npz(equal_info / "local_q_scores.npz")["val"],
                                    val.data["is_remote"].astype(bool))
        progress(out, "fitting_cal8_source_maps")
        maps = fit_maps(train, train_scores)
        atomic(out / "source_monotone_maps.json", maps)
        calibrated_train = apply_maps(train_scores, train.data["is_remote"], maps)
        calibrated_val = apply_maps(val_scores, val.data["is_remote"], maps)
        np.savez_compressed(out / "calibrated_scores.npz",
                            **{f"train_{k}": v for k, v in calibrated_train.items()},
                            **{f"val_{k}": v for k, v in calibrated_val.items()})
        workpoints = calibrate_workpoints(train, calibrated_train)
        atomic(out / "working_points.json", workpoints)
        progress(out, "evaluating_hold14")
        hold = evaluate(train, train_scores, calibrated_train, workpoints, out, "mechanism_holdout14")
        progress(out, "evaluating_val21")
        val_output = evaluate(val, val_scores, calibrated_val, workpoints, out, "val21")
        val_summary = json.loads((val_output / "summary.json").read_text())
        ap70 = {method: {str(wait): 100 * next(row["ap70"] for row in val_summary["panels"]
                                               if row["sequence"] == "all" and row["wait_ms"] == wait
                                               and row["method"] == method + "_calibrated")
                         for wait in (0, 100, 200, 400)} for method in METHODS}
        raw_ap70 = {method: {str(wait): 100 * next(row["ap70"] for row in val_summary["panels"]
                                                   if row["sequence"] == "all" and row["wait_ms"] == wait
                                                   and row["method"] == method + "_raw")
                             for wait in (0, 100, 200, 400)} for method in METHODS}
        atomic(out / "summary.json", {
            "schema": "traf_receiver_local_source_monotone_calibration_v1",
            "status": "complete_review_pending", "methods": list(METHODS),
            "calibrated_val_AP70_percent": ap70, "raw_val_AP70_percent": raw_ap70,
            "delta_calibrated_minus_raw_pp": {
                method: {wait: ap70[method][wait] - raw_ap70[method][wait] for wait in ap70[method]}
                for method in METHODS},
            "maps_sha256": sha256(out / "source_monotone_maps.json"),
            "scores_sha256": sha256(out / "calibrated_scores.npz"),
            "working_points_sha256": sha256(out / "working_points.json"),
            "hold_summary_sha256": sha256(hold / "summary.json"),
            "val_summary_sha256": sha256(val_output / "summary.json"),
            "new_detector_or_backbone_forwards": 0, "detector_updates": 0,
            "wall_seconds": time.monotonic() - started,
            "boundary": "Cal8-only sourcewise positive-slope logistic correction of frozen receiver scores. System ranking effect, not module geometry, communication, strict-arrival, blind-test or novelty evidence.",
        })
        progress(out, "complete_review_pending")
    except Exception as error:
        atomic(out / "failure.json", {"status": "failed_preserved", "error": repr(error)})
        progress(out, "failed_preserved", error=repr(error))
        raise


if __name__ == "__main__":
    main()
