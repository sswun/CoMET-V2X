#!/usr/bin/env python3
"""Original ego-only source AP, restoring the original float64 tie ranking."""
import argparse
from pathlib import Path
import resource
import time

import run_peer07_timing_2x2_source_split_ap_v2_20261031 as core
from run_peer07_geometry_matched_source_ap_v1_20261031 import paired_bootstrap
from e1_receiver_value_core_v1_20261020 import unique_ego_scores


def compose(meta, pool, saved, role):
    selected = [i for i, row in enumerate(meta) if row["role"] == role]
    if len(saved["ego__offset"]) != len(selected) + 1:
        raise RuntimeError("saved ego population")
    material = {"dataset_index": core.np.asarray(selected), **{k: core.np.asarray([meta[i][k] for i in selected]) for k in ("wait_ms", "sequence", "receiver_timestamp_us")}}
    boxes, scores, original, gts, gt_ranges, tps = [], [], [], [], [], {50: [], 70: []}
    offsets, gt_offsets = [0], [0]
    for ordinal, index in enumerate(selected):
        row = meta[index]; a, b, ga, gb = [int(row[k]) for k in ("start", "stop", "gt_start", "gt_stop")]
        ego = core.np.flatnonzero(pool["hypothesis_type"][a:b] == 0)
        rank = unique_ego_scores(pool["raw_scores"][a:b][ego], stable=True)
        order = core.np.argsort(-rank, kind="stable")
        local, rank = ego[order], rank[order]
        lo, hi = map(int, saved["ego__offset"][ordinal:ordinal + 2])
        if not core.np.array_equal(local, saved["ego__order"][lo:hi]) or not core.np.array_equal(rank.astype(core.np.float32), saved["ego__score"][lo:hi]):
            raise RuntimeError("original ego candidate or float32 serialized ranking drift")
        matrix = pool["ious"][row["iou_start"]:row["iou_stop"]].reshape(b - a, gb - ga)
        for t in (50, 70):
            tps[t].append((core.greedy(matrix, local, t / 100) >= 0).astype(core.np.int8))
        original.append(a + local); boxes.append(pool["boxes"][a + local]); scores.append(rank)
        gts.append(pool["gt_boxes"][ga:gb]); gt_ranges.append((ga, gb))
        offsets.append(offsets[-1] + len(local)); gt_offsets.append(gt_offsets[-1] + gb - ga)
    geometry = core.np.concatenate(boxes)
    material.update(condition_offsets=core.np.asarray(offsets), gt_offsets=core.np.asarray(gt_offsets), gt_boxes=core.np.concatenate(gts))
    prediction = {"ego__candidate": core.np.arange(len(geometry), dtype=core.np.int64),
                  "ego__score": core.np.concatenate(scores), "ego__offset": core.np.asarray(offsets),
                  **{f"ego__tp{t}": core.np.concatenate(tps[t]) for t in (50, 70)}}
    return material, prediction, geometry, core.np.concatenate(original), gt_ranges


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); c = core.load(args.contract)
    for item in c["inputs"].values():
        if core.evidence(core.ROOT / item["path"]) != item:
            raise RuntimeError(("input drift", item["path"]))
    args.output.mkdir(exist_ok=False, parents=True)
    start, cpu = time.monotonic(), time.process_time()
    ref = core.load(core.ROOT / c["inputs"]["reference_source_summary"]["path"])
    panels = [r for r in ref["formal_panels"] if r["method"] == "stale__raw"]
    sequences = [r for r in ref["per_sequence_panels"] if r["method"] == "stale__raw"]
    core.METHODS = ("ego",)
    outputs, exact = {}, []
    for role in core.ROLES:
        meta = core.load_jsonl(core.ROOT / c["inputs"][role + "_conditions"]["path"])
        pool = core.load_npz(core.ROOT / c["inputs"][role + "_pools"]["path"])
        saved = core.load_npz(core.ROOT / c["inputs"][role + "_predictions"]["path"])
        material, prediction, geometry, original, gt_ranges = compose(meta, pool, saved, role)
        side = core.load_npz(core.ROOT / c["inputs"][role + "_source_codes"]["path"])["gt_from_side"]
        side = core.np.concatenate([side[a:b] for a, b in gt_ranges])
        rp, rs, arrays, checks = core.evaluate_role(material, prediction, {"ego": geometry}, side, role)
        exact.extend(checks.values())
        old = core.load(core.ROOT / c["inputs"][role + "_evaluation"]["path"])
        lookup = {(r["sequence"], r["wait_ms"]): r for r in old["panels"] if r["method"] == "ego"}
        for r in rp + rs:
            if r["source_group"] == "all":
                target = lookup[r.get("sequence", "all"), r["wait_ms"]]
                exact.extend(abs(r[k] - target[k]) <= 1e-10 for k in ("ap50", "ap70", "gt", "tp70", "fp70"))
        panels.extend(rp); sequences.extend(rs)
        path = args.output / (role + "_source_outcomes.npz")
        core.save_npz(path, **arrays, original_pool_candidate=original)
        outputs[role] = core.evidence(path)
    boots = paired_bootstrap(sequences, c)
    checks = {"original_float64_AP_TP_and_serialized_rank_replayed": bool(all(exact)), "48_panels_24_new_24_reference": len(panels) == 48,
              "16_bootstrap": len(boots) == 16, "zero_model_fit_forward_update_GPU": True, "no_GT_geometry_feature_copy": True}
    core.atomic_json(args.output / "summary.json", {"status": "completed_pending_independent_review", "task_id": c["task_id"], "contract": core.evidence(args.contract),
        "formal_panels": panels, "per_sequence_panels": sequences, "paired_sequence_bootstrap": boots, "checks": checks, "outputs": outputs,
        "actual_cost": {"wall_seconds": time.monotonic() - start, "cpu_process_seconds": time.process_time() - cpu,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "detector_or_backbone_forwards": 0,
            "optimizer_or_head_updates": 0, "gpu_seconds": 0}})
    if not all(checks.values()):
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
