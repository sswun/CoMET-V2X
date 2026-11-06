"""Independent sampled CPU geometry and release-NMS diagnostic for stage ledger."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
from shapely.geometry import Polygon

from evaluate_traf_v2xseq_adapted_fixed60_raw_20261006 import atomic_json, now, sha256


def poly(box):
    x, y, _, _, w, l, angle = map(float, box[:7])
    c, s = math.cos(angle), math.sin(angle)
    vertices = [(x + c*u - s*v, y + s*u + c*v) for u, v in ((-l/2,-w/2), (-l/2,w/2), (l/2,w/2), (l/2,-w/2))]
    return Polygon(vertices)


def iou(a, b):
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter)


def candidates(boxes, scores, targets, threshold=.2):
    """Independent circumscribed-circle reject, not runner AABB/area pruning."""
    radii = np.sqrt(boxes[:, 4].astype(float)**2 + boxes[:, 5].astype(float)**2) / 2
    polygons = {}
    result = []
    for target in targets:
        b = poly(target)
        gr = math.hypot(float(target[4]), float(target[5])) / 2
        allowed = np.flatnonzero(np.linalg.norm(boxes[:, :2] - target[None, :2], axis=1) <= radii + gr + 1e-5)
        strict = []
        for index in allowed:
            if index not in polygons:
                polygons[index] = poly(boxes[index])
            value = iou(polygons[index], b)
            if value >= .7:
                strict.append((int(index), value))
        winner = max(strict, key=lambda item: float(scores[item[0]])) if strict else None
        result.append({"best_anchor": winner[0] if winner else None, "has_geometry": bool(winner),
                       "has_score": bool(winner and scores[winner[0]] > threshold),
                       "iou": winner[1] if winner else None, "score": float(scores[winner[0]]) if winner else None})
    return result


def nms(boxes, scores, overlap_threshold):
    order = np.argsort(-scores, kind="stable")[:4096]
    retained = []
    polygons = {}
    radii = np.sqrt(boxes[:, 4].astype(float)**2 + boxes[:, 5].astype(float)**2) / 2
    for index in order:
        if scores[index] <= .2 or len(retained) == 100:
            break
        polygons[index] = poly(boxes[index])
        eligible = [other for other in retained if np.linalg.norm(boxes[index, :2] - boxes[other, :2]) <= radii[index] + radii[other] + 1e-5]
        if all(iou(polygons[index], polygons[other]) <= overlap_threshold for other in eligible):
            retained.append(int(index))
    return retained


def matched(boxes, scores, targets):
    remaining, pairs = list(range(len(targets))), {}
    pg, pb = [poly(g) for g in targets], [poly(b) for b in boxes]
    overlap = np.array([[iou(b, g) for g in pg] for b in pb]).reshape(len(boxes), len(targets))
    for index in np.argsort(-scores, kind="stable"):
        if not remaining:
            break
        column = max(remaining, key=lambda j: overlap[index, j])
        if overlap[index, column] >= .7:
            pairs[column] = int(index); remaining.remove(column)
    return pairs, overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    summary = json.loads((args.analysis / "summary.json").read_text())
    canonical = {}
    if summary.get("canonical_review"):
        source_review = json.loads(Path(summary["canonical_review"]).read_text())
        if source_review["status"] != "passed":
            raise RuntimeError("Canonical auxiliary review not accepted")
        canonical = {(x["global_index"], x["configured_wait_ms"]): x for x in source_review["canonical_rows"]}
    rows = [json.loads(x) for x in (args.run / "frame_manifest.jsonl").read_text().splitlines()]
    ledger = [json.loads(x) for x in (args.analysis / "target_ledger.jsonl").read_text().splitlines()]
    by_frame, counts = defaultdict(list), defaultdict(Counter)
    for target in ledger:
        by_frame[(target["global_index"], target["wait_ms"])].append(target)
        counts[str(target["wait_ms"])].update({"gt":1, target["stage"]:1,
            "aux_ego_scored":int(target["aux_ego"]["has_score"]), "aux_remote_scored":int(target["aux_remote"]["has_score"]),
            "aux_remote_only_scored":int(target["aux_remote"]["has_score"] and not target["aux_ego"]["has_score"])})
    indices = sorted({r["global_index"] for r in rows})
    selected = {indices[0], indices[len(indices)//2], indices[-1]}
    cases = []
    for row in rows:
        if row["global_index"] not in selected:
            continue
        with np.load(row["raw_path"], allow_pickle=False) as raw:
            gt = raw["gt_boxes"]
            source_boxes, source_scores = raw["source_pre_boxes"], raw["source_pre_scores"]
            if canonical:
                aux = canonical[(row["global_index"],row["configured_wait_ms"])]
                if sha256(Path(aux["canonical_path"])) != aux["canonical_sha256"]:
                    raise RuntimeError("Canonical auxiliary changed")
                with np.load(aux["canonical_path"],allow_pickle=False) as array:
                    source_boxes, source_scores = array["source_pre_boxes"], array["source_pre_scores"]
            ego = candidates(source_boxes[0], source_scores[0, :, 0], gt)
            remote = candidates(source_boxes[1], source_scores[1, :, 0], gt) if len(source_boxes) > 1 else [dict(best_anchor=None, has_geometry=False, has_score=False, iou=None, score=None) for g in gt]
            fused = candidates(raw["fused_pre_boxes"], raw["fused_pre_scores"][:, 0], gt)
            final, overlap = matched(raw["pred_boxes"], raw["pred_scores"], gt)
            items = sorted(by_frame[(row["global_index"], row["configured_wait_ms"])], key=lambda item:item["gt_index"])
            geometry = all(len(items) == len(gt) and all(a[k] == target[group][k] for k in ("best_anchor", "has_geometry", "has_score"))
                           for group, group_items in (("aux_ego",ego),("aux_remote",remote),("fused",fused)) for a,target in zip(group_items, items))
            final_check = all(target["final_prediction_index"] == final.get(j) for j,target in enumerate(items))
            kept = nms(raw["fused_pre_boxes"], raw["fused_pre_scores"][:, 0], .2)
            yaml_kept = nms(raw["fused_pre_boxes"], raw["fused_pre_scores"][:, 0], .15)
            native_match = len(kept) == len(raw["pred_boxes"]) and all(np.any(np.all(np.isclose(raw["pred_boxes"], raw["fused_pre_boxes"][k], atol=1e-5, rtol=0), 1) & np.isclose(raw["pred_scores"], raw["fused_pre_scores"][k, 0], atol=1e-6, rtol=0)) for k in kept)
            corrected_match, _ = matched(raw["fused_pre_boxes"][yaml_kept], raw["fused_pre_scores"][yaml_kept, 0], gt)
            cases.append({"global_index": row["global_index"], "wait_ms": row["configured_wait_ms"],
                          "checks":{"independent_dense_source_geometry":geometry, "independent_final_assignment":final_check,
                                    "raw_hash":sha256(Path(row["raw_path"])) == row["raw_sha256"], "CPU_native_NMS_reproduces_final":native_match},
                          "native_nms_point2":{"tp":len(final), "fp":len(raw["pred_boxes"]) - len(final)},
                          "yaml_nms_point15":{"tp":len(corrected_match), "fp":len(yaml_kept)-len(corrected_match)},
                          "nms_scope":"Diagnostic YAML-vs-release change; not a new method. FP/TP thresholds are native, not matched budget."})
    checks = {"analysis_passed":summary["status"] == "passed" and all(summary["checks"].values()),
              "all_ledger_hashes":all(sha256(args.analysis/name) == digest for name,digest in summary["ledger_sha256"].items()),
              "partition_and_source_counts":all(all(summary["counts_by_wait"][wait].get(k,0) == v for k,v in value.items()) for wait,value in counts.items()),
              "all_sampled_geometry_and_native_NMS_checks":all(all(case["checks"].values()) for case in cases),
              "deterministic_first_middle_last_each_wait":len(cases) == len(selected) * len({r["configured_wait_ms"] for r in rows})}
    value = {"status":"passed" if all(checks.values()) else "failed", "checks":checks, "cases":cases,
             "reviewed_at":now(), "reviewer_sha256":sha256(Path(__file__)), "new_detector_forward_count":0, "new_GPU_forward_count":0,
             "wall_seconds":time.monotonic()-started,
             "boundary":"Independent first/middle/last manifest frame per wait, not every raw geometry recomputed. Full ledger hash/partition checks, sampled circle-pruned exact geometry and CPU release NMS. No causal communication or raw-point/source-observability claim. YAML mismatch is engineering provenance; sampled train results are not full benchmark evidence."}
    atomic_json(args.output,value)
    print(json.dumps({k:value[k] for k in ("status","checks","cases","wall_seconds")}))
    if value["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
