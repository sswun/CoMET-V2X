"""CPU-only common retained ROI re-scoring; never rewrite native results.

Input/train ROI differences remain. This is an output/GT common-ROI control,
not equal-input retraining; strict arms are separate from same-calibration nominal controls and independently trained ego.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import review_traf_v2xseq_adapted_fixed60_raw_20261006 as review
from review_where2comm_source_stage_ledger_20261009 import matched

ROI = [0, -38.4, -3, 96, 38.4, 1.5]


def mask(boxes):
    return ((boxes[:, 0] >= ROI[0]) & (boxes[:, 0] <= ROI[3]) &
            (boxes[:, 1] >= ROI[1]) & (boxes[:, 1] <= ROI[4]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root, out = a.workspace.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cases = {}
    specifications = []
    old = root / "artifacts/research_autopilot/status_audit_20261009_1800"
    for method in ("where2comm", "v2xvit", "fcooper"):
        ego_run = {"where2comm": "v2xseq_where2comm_adapted_fixed60_ego_only_v1_20261007", "v2xvit": "v2xseq_v2xvit_adapted_fixed60_ego_only_v1_20261008", "fcooper": "v2xseq_fcooper_adapted_fixed60_ego_only_v1_20261008"}[method]
        specifications.append((method, "same_weight_ego", root / "artifacts/runs" / ego_run / "ego_only"))
        for arm in ("delay_000ms", "delay_100ms", "delay_200ms", "delay_400ms", "nominal_sensor_extrinsics_000ms"):
            if method == "fcooper":
                if arm == "nominal_sensor_extrinsics_000ms":
                    folder = root / "artifacts/research_autopilot/status_audit_20261009_2100/fcooper_nominal_sensor0" / arm
                else:
                    folder = old / ("remote_strict_v2_first_arm" if arm == "delay_000ms" else "remote_strict_v2_completed_arms") / arm
            else:
                folder = root / "artifacts/runs" / f"v2xseq_{method}_recorded_arrival_fixed60_v1_local_20261009" / arm
            specifications.append((method, arm, folder))
    specifications.append(("independent_ego", "independent_ego", root / "artifacts/runs/v2xseq_independent_ego_shared_fixed60_eval_v1_20261009/independent_ego"))
    for method, arm, path in specifications:
        r = json.loads((path / "independent_review.json").read_text())
        if r["status"] != "passed" or not all(r["checks"].values()):
            raise RuntimeError("Complete condition review required: " + str(path))
        rows = review.read_jsonl(path / "frame_manifest.jsonl")
        if len(rows) != 3316 or [x["global_index"] for x in rows] != list(range(3316)):
            raise RuntimeError("Full3316 unique ordered coverage required")
        cases[(method, arm)] = {"path": path, "review": r, "rows": rows}
    stats = defaultdict(lambda: {t: {"scores": [], "tp": [], "fp": [], "gt": 0} for t in (.5, .7)})
    original_gt, dropped_gt, sampled_checks = Counter(), Counter(), []
    sequence_indices = defaultdict(list)
    for row in cases[("where2comm", "delay_000ms")]["rows"]:
        sequence_indices[row["sequence"]].append(row["global_index"])
    sampled = {indices[j] for indices in sequence_indices.values() for j in (0, len(indices)//2, -1)}
    gt_digest, explain = hashlib.sha256(), Counter()
    with (out / "frame_ledger.jsonl").open("w") as stream:
        for index in range(3316):
            target, target_labels, vehicle = None, None, None
            for (method, arm), case in cases.items():
                row = case["rows"][index]
                path = Path(row["raw_path"])
                authoritative_raw_path = str(path)
                if not path.is_file():
                    remote_prefix = "/path/to/remote_project/"
                    if not str(path).startswith(remote_prefix):
                        raise RuntimeError("Unknown original raw host")
                    path = case["path"] / "raw" / path.name
                    if not path.is_file():
                        raise FileNotFoundError("Verified raw mirror missing: " + str(path))
                if review.sha256(path) != row["raw_sha256"]:
                    raise RuntimeError("Raw hash differs")
                with np.load(path, allow_pickle=False) as data:
                    raw = {k: data[k] for k in ("pred_boxes", "pred_scores", "pred_labels", "gt_boxes", "gt_labels")}
                if not all(np.isfinite(v).all() for v in raw.values()):
                    raise RuntimeError("Nonfinite raw")
                if vehicle is None:
                    vehicle = row["effective_vehicle_frame"]
                if row["effective_vehicle_frame"] != vehicle:
                    raise RuntimeError("Current-frame identity differs")
                keep_gt = mask(raw["gt_boxes"])
                gt, gl = raw["gt_boxes"][keep_gt], raw["gt_labels"][keep_gt]
                if target is None:
                    target, target_labels = gt, gl
                    gt_digest.update(gt.tobytes()); gt_digest.update(gl.tobytes())
                if not np.array_equal(gt, target) or not np.array_equal(gl, target_labels):
                    raise RuntimeError(f"Shared ROI current GT differs: {method}:{arm}:{index}")
                original_gt[(method, arm)] += len(raw["gt_boxes"])
                dropped_gt[(method, arm)] += int((~keep_gt).sum())
                if arm == "delay_000ms":
                    explain[(method, "x_gt96")] += int((raw["gt_boxes"][:,0] > 96).sum())
                    explain[(method, "abs_y_gt38p4")] += int((np.abs(raw["gt_boxes"][:,1]) > 38.4).sum())
                keep = mask(raw["pred_boxes"])
                record = {"method": method, "arm": arm, "global_index": index, "sequence": row["sequence"],
                          "raw_path": str(path), "authoritative_raw_path": authoritative_raw_path, "raw_sha256": row["raw_sha256"], "retained_prediction_indices": np.flatnonzero(keep).tolist(),
                          "retained_GT_indices": np.flatnonzero(keep_gt).tolist(), "native_GT_count": len(raw["gt_boxes"])}
                for t in (.5, .7):
                    scores, tp, fp, count = review.match_frame(raw["pred_boxes"][keep], raw["pred_scores"][keep], raw["pred_labels"][keep], gt, gl, t)
                    s = stats[(method, arm)][t]
                    for k, v in (("scores",scores), ("tp",tp), ("fp",fp)):
                        s[k].extend(v)
                    s["gt"] += count
                    record[str(t)] = {"scores": scores, "tp": tp, "fp": fp, "gt": count}
                    if t == .7 and index in sampled:
                        dp = keep & (raw["pred_labels"].astype(int) == 0)
                        gp = gl.astype(int) == 0
                        pairs, _ = matched(raw["pred_boxes"][dp], raw["pred_scores"][dp], gt[gp])
                        sampled_checks.append({"method": method, "arm": arm, "global_index": index, "counts_match": len(pairs) == sum(tp)})
                stream.write(json.dumps(record) + "\n")
            if index % 128 == 0:
                review.atomic_json(out / "status.json", {"status": "running", "completed_frames": index+1, "new_detector_forward_count": 0})
    metrics = {method + ":" + arm: {
        "iou" + str(int(t*100)): review.voc_ap(s["tp"], s["fp"], s["scores"], s["gt"], True) for t,s in ss.items()}
        for (method, arm), ss in stats.items()}
    checks = {"nineteen_complete_condition_reviews_accepted": True, "all63004_raw_hashes_and_finite_arrays": True,
              "all3316_current_frames_shared": True, "common_retained_GT_exact_all_methods_and_arms": True,
              "independent_sequence_boundary_middle_geometry_counts": len(sampled_checks) == len(sampled)*19 and all(x["counts_match"] for x in sampled_checks),
              "common_GT_count_shared": len({m["iou70"]["gt"] for m in metrics.values()}) == 1}
    summary = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "observed_at": review.now(),
               "common_output_GT_roi": ROI, "common_GT_digest": gt_digest.hexdigest(), "metrics": metrics,
               "native_GT_counts": {k[0]+":"+k[1]: v for k,v in original_gt.items()},
               "dropped_GT_counts": {k[0]+":"+k[1]: v for k,v in dropped_gt.items()},
               "out_of_common_ROI_counts": {k[0]+":"+k[1]: v for k,v in explain.items()},
               "sampled_independent_geometry_cases": len(sampled_checks), "sampled_independent_geometry_failed": [x for x in sampled_checks if not x["counts_match"]],
               "raw_reviews_sha256": {str(c["path"] / "independent_review.json"): review.sha256(c["path"] / "independent_review.json") for c in cases.values()},
               "ledger_sha256": review.sha256(out / "frame_ledger.jsonl"), "reviewer_sha256": review.sha256(Path(__file__)),
               "new_GPU_forward_count": 0, "new_detector_forward_count": 0, "wall_seconds": time.monotonic()-started,
               "cause": "Released ViT voxel/label ROI x<=102.4,y+-38.4; eval predictions x<=96,y+-38.4. infer_range=False leaves GT in voxel ROI, and extract_frame_gt does not re-mask GT. Where2comm voxel/label/pred ROI x<=96,y+-40. Native within-method tables retain their identities but are not a common-GT cross-method ranking.",
               "boundary": "Separate CPU common retained-ROI control; native files/reviews unchanged. Different native training and input ROIs, attention architectures, weights and output limits remain. Public val21 exposed. delay_* are strict recorded arrival sensor-only conditions; nominal_sensor0 and same_weight_ego/independent_ego remain separately identified. Nominal-trained models and different input/train ROIs remain: not architecture-isolated ranking, domain-matched retraining, clock confirmation or novelty evidence. Geometry independently sampled first/middle/last each sequence; full raw/hash/GT/metrics ledger preserved."}
    review.atomic_json(out / "summary.json", summary); review.atomic_json(out / "status.json", summary)
    print(json.dumps({"status":summary["status"],"checks":summary["checks"],"commonGT":sorted({m["iou70"]["gt"] for m in metrics.values()}),"AP70":{k:v["iou70"]["ap"] for k,v in metrics.items()},"wall_seconds":summary["wall_seconds"]}))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
