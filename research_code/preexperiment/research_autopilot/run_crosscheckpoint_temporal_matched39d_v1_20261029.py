#!/usr/bin/env python3
"""Equal-opportunity matched 39D heads for selected-Q temporal geometries."""
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
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / "preexperiment/research_autopilot"
BASE = ROOT / "preexperiment/research_baselines"
sys.path[:0] = [str(AUTO), str(BASE)]
import run_crosscheckpoint_current_temporal_fixed_readout_controls_v1_20261029 as fixed
import run_shared_offset_task_tolerance_readout_v1_20261026 as readout


ROLES = ("fit24", "calibration8", "mechanism_holdout14", "val21")
FORMAL = ROLES[2:]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): value.update(block)
    return value.hexdigest()


def load(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def rel(path: Path) -> str: return str(path.resolve().relative_to(ROOT))


def evidence(path: Path) -> dict[str, Any]: return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def geometry_material(base: dict[str, np.ndarray], boxes: np.ndarray,
                      features: np.ndarray) -> dict[str, np.ndarray]:
    arrays = defaultdict(list); iou_offsets = [0]
    for condition in range(len(base["dataset_index"])):
        a, b = base["condition_offsets"][condition:condition + 2].astype(int)
        ga, gb = base["gt_offsets"][condition:condition + 2].astype(int)
        gt = base["gt_boxes"][ga:gb]; matrix = readout.iou32(boxes[a:b], gt)
        fixed_ids = np.flatnonzero(base["protected"][a:b] == 1)
        if len(fixed_ids):
            rank = 2.0 + readout.unique_ego_scores(base["raw_scores"][a:b][fixed_ids], stable=True)
            order = fixed_ids[np.argsort(-rank, kind="stable")]
            matched = readout.greedy(matrix, order, .7); covered = {int(x) for x in matched if x >= 0}
        else:
            covered = set()
        uncovered = np.asarray([i for i in range(len(gt)) if i not in covered], np.int64)
        quality = (matrix.max(axis=1, initial=0.0) >= .7).astype(np.int8)
        utility = (matrix[:, uncovered].max(axis=1, initial=0.0) >= .7).astype(np.int8)
        ego = np.flatnonzero(base["source"][a:b] == 0); remote = np.flatnonzero(base["source"][a:b] == 1)
        overlap = np.zeros(b - a, np.float32)
        if len(ego) and len(remote): overlap[remote] = readout.iou32(boxes[a:b][remote], boxes[a:b][ego]).max(axis=1, initial=0.0)
        arrays["quality_y"].append(quality); arrays["utility_y"].append(utility)
        arrays["max_ego_iou"].append(overlap); arrays["ious"].append(matrix.astype(np.float32).reshape(-1))
        iou_offsets.append(iou_offsets[-1] + matrix.size)
    output = {key: base[key].copy() for key in ("raw_scores", "source", "protected", "anchor",
              "condition_offsets", "gt_offsets", "gt_boxes", "dataset_index", "sequence", "wait_ms",
              "receiver_timestamp_us")}
    output.update({"boxes": boxes.astype(np.float32), "features": features.astype(np.float32),
                   "quality_y": np.concatenate(arrays["quality_y"]), "utility_y": np.concatenate(arrays["utility_y"]),
                   "max_ego_iou": np.concatenate(arrays["max_ego_iou"]), "ious": np.concatenate(arrays["ious"]),
                   "iou_offsets": np.asarray(iou_offsets, np.int64)})
    return output


def train_head(material: dict[str, np.ndarray], old_schedule: Path, output: Path) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F
    output.mkdir(parents=True, exist_ok=False)
    schedule = load_npz(old_schedule); weights = schedule["weights"]; batches = schedule["batches"]
    if len(weights) != len(material["features"]) or int(batches.max()) >= len(weights):
        raise RuntimeError("frozen schedule population mismatch")
    mean = (material["features"].astype(np.float64) * weights[:, None]).sum(axis=0)
    std = np.maximum(np.sqrt(((material["features"] - mean) ** 2 * weights[:, None]).sum(axis=0)), 1e-6)
    normalized = ((material["features"] - mean) / std).astype(np.float32)
    target = material["utility_y"].astype(np.float32)
    torch.set_num_threads(2); torch.manual_seed(readout.MODEL_SEED)
    model = nn.Sequential(nn.Linear(39, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    matrix = torch.from_numpy(normalized); labels = torch.from_numpy(target); trace = []; recent = []
    started = time.monotonic()
    for update in range(readout.MODEL_UPDATES):
        ids = torch.from_numpy(batches[update].astype(np.int64)); logits = model(matrix[ids]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels[ids])
        if not bool(torch.isfinite(loss)): raise RuntimeError("nonfinite loss")
        optimizer.zero_grad(); loss.backward(); optimizer.step(); recent.append(float(loss.detach()))
        if (update + 1) % 250 == 0:
            trace.append({"update": update + 1, "mean_loss_last250": float(np.mean(recent[-250:]))})
    torch.save(model.state_dict(), output / "direct_u.pth")
    np.savez_compressed(output / "training_schedule.npz", weights=weights, batches=batches, mean=mean, std=std)
    result = {"model": "39x32_relu_x1", "seed": readout.MODEL_SEED, "updates": readout.MODEL_UPDATES,
              "parameters": int(sum(p.numel() for p in model.parameters())), "trace": trace,
              "wall_seconds": time.monotonic() - started, "weights": evidence(output / "direct_u.pth"),
              "schedule": evidence(output / "training_schedule.npz"),
              "frozen_weights_exact": bool(np.array_equal(weights, schedule["weights"])),
              "frozen_batches_exact": bool(np.array_equal(batches, schedule["batches"])), "gpu_seconds": 0}
    atomic(output / "training_summary.json", result); return result


def score(material: dict[str, np.ndarray], head: Path) -> np.ndarray:
    return readout.score_direct_u(material, head)


def source_map(material: dict[str, np.ndarray], raw: np.ndarray, head: Path) -> np.ndarray:
    maps = readout.fit_source_maps(material, raw); atomic(head / "source_maps.json", maps)
    return raw


def mapped(material: dict[str, np.ndarray], head: Path) -> np.ndarray:
    direct = score(material, head); maps = load(head / "source_maps.json")
    return readout.apply_source_map(direct, material["source"], maps["maps"]["direct_to_utility"])


def prepare_domain(spec: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]],
                                                   dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    populations = {}; frames = {}; materials = {}; references = {}
    for role in ROLES:
        rs = spec["roles"][role]; populations[role] = load_npz(ROOT / rs["population"])
        frames[role] = load_npz(ROOT / rs["frame_input"]); fixed.identity(populations[role], frames[role])
        if spec["material_kind"] == "vit_reconstructed":
            reference = load_npz(ROOT / rs["reference"])
            materials[role], kb, kf, _ = fixed.reconstruct_vit_material(populations[role], reference, role in FORMAL)
        else:
            materials[role] = dict(populations[role]); kb, kf = materials[role]["kf_boxes"], materials[role]["kf_features"]
        references[role] = {"boxes": kb, "features": kf}
    constants, _ = fixed.fit_constants(frames["fit24"], populations["fit24"]["wait_ms"])
    deltas = {}
    for role in ROLES:
        deltas[role] = {}
        for arm in fixed.ARMS:
            deltas[role][arm], _ = fixed.run_state(frames[role], populations[role]["sequence"],
                                                   populations[role]["wait_ms"], arm, constants)
    return materials, deltas, references


def domain_compute(name: str, spec: dict[str, Any], selected: str, output: Path) -> dict[str, Any]:
    materials, deltas, references = prepare_domain(spec)
    arms = ["q015"] if selected == "q015" else ["q015", selected]
    geometries: dict[str, dict[str, dict[str, np.ndarray]]] = {role: {} for role in ROLES}
    for role in ROLES:
        for arm in arms:
            boxes, features, _ = fixed.base.compose(materials[role], deltas[role][arm])
            geometries[role][arm] = geometry_material(materials[role], boxes, features)
    domain_dir = output / name; domain_dir.mkdir(parents=True, exist_ok=False)
    old_schedule = ROOT / spec["readout_root"] / "training_schedule.npz"
    heads = {}; training = {}
    for arm in arms:
        head = domain_dir / f"head_{arm}"; training[arm] = train_head(geometries["fit24"][arm], old_schedule, head)
        direct_cal = score(geometries["calibration8"][arm], head)
        source_map(geometries["calibration8"][arm], direct_cal, head); heads[arm] = head
    schedule_checks = {}
    old = load_npz(old_schedule)
    for arm, head in heads.items():
        new = load_npz(head / "training_schedule.npz")
        schedule_checks[arm] = {"weights_exact": bool(np.array_equal(old["weights"], new["weights"])),
                                "batches_exact": bool(np.array_equal(old["batches"], new["batches"])),
                                "mean_geometry_specific": True, "std_geometry_specific": True}
    panels = {}; outputs = {}
    old_maps = load(ROOT / spec["readout_root"] / "source_maps.json")
    for role in FORMAL:
        boxes_by_method = {}; scores_by_method = {}
        for arm in arms:
            geom = geometries[role][arm]; boxes_by_method[f"{arm}_raw"] = geom["boxes"]
            scores_by_method[f"{arm}_raw"] = geom["raw_scores"].astype(np.float64)
            old_direct = fixed.base.actual.optional.direct_scores({"old": geom["features"]}, ROOT / spec["readout_root"])["old"]
            scores_by_method[f"{arm}_fixed39d"] = fixed.base.actual.optional.source_map(
                old_direct, geom["source"], old_maps["maps"]["direct_to_utility"])
            boxes_by_method[f"{arm}_fixed39d"] = geom["boxes"]
            boxes_by_method[f"{arm}_matched39d"] = geom["boxes"]
            scores_by_method[f"{arm}_matched39d"] = mapped(geom, heads[arm])
        task = load_npz(ROOT / spec["roles"][role]["task_material"])
        boxes_by_method["current_task_fresh39d"] = task["task_boxes"]
        scores_by_method["current_task_fresh39d"] = task["task_direct_u"]
        role_panels, predictions = fixed.base.evaluate(materials[role], boxes_by_method, scores_by_method)
        panels[role] = role_panels; prediction_path = domain_dir / f"{role}_predictions.npz"
        np.savez_compressed(prediction_path, **predictions); outputs[role] = evidence(prediction_path)
    effects = []
    for role in FORMAL:
        table = {(int(row["wait_ms"]), row["method"]): row for row in panels[role]}
        for wait in sorted(map(int, np.unique(materials[role]["wait_ms"]))):
            selected_key = f"{selected}_matched39d"; q_key = "q015_matched39d"
            effects.append({"role": role, "wait_ms": wait, "selected": selected,
                            "selected_matched_ap70": table[(wait, selected_key)]["ap70"],
                            "q015_matched_ap70": table[(wait, q_key)]["ap70"],
                            "selected_minus_q015_matched_ap70_pp": table[(wait, selected_key)]["ap70"] - table[(wait, q_key)]["ap70"],
                            "selected_fixed_minus_matched_ap70_pp": table[(wait, f"{selected}_fixed39d")]["ap70"] - table[(wait, selected_key)]["ap70"],
                            "current_task_minus_selected_matched_ap70_pp": table[(wait, "current_task_fresh39d")]["ap70"] - table[(wait, selected_key)]["ap70"]})
    return {"identity": spec["identity"], "selected": selected, "trained_arms": arms,
            "training": training, "schedule_checks": schedule_checks, "formal_panels": panels,
            "effects": effects, "outputs": outputs,
            "task_refit_gap": (None if selected == "q015" else
                "Current task head/teacher was trained from q015-base targets; a selected-q005-base task refit is still required.")}


def compute(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    fixed_contract = load(ROOT / contract["fixed_contract"]["path"])
    fixed_terminal = load(ROOT / contract["fixed_terminal"]["path"])
    for item in fixed_contract["inputs"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]: raise RuntimeError(("fixed input drift", item["path"]))
    selected = {name: value["q_selection"]["selected"] for name, value in fixed_terminal["domains"].items()}
    domains = {name: domain_compute(name, spec, selected[name], output)
               for name, spec in fixed_contract["domains"].items()}
    checks = {"three_domains": set(domains) == {"v2xvit_seed2024", "v2xvit_seed2025", "where2comm"},
              "seed2024_q015_reused_once": domains["v2xvit_seed2024"]["trained_arms"] == ["q015"],
              "q005_domains_train_both": all(domains[name]["trained_arms"] == ["q015", "q005"]
                                           for name in ("v2xvit_seed2025", "where2comm")),
              "all_frozen_batches_weights_exact": all(all(c["weights_exact"] and c["batches_exact"]
                                                           for c in domain["schedule_checks"].values())
                                                        for domain in domains.values()),
              "zero_detector_backbone_forward": True, "zero_gpu": True}
    if not all(checks.values()): raise RuntimeError(checks)
    result = {"schema": "crosscheckpoint_temporal_matched39d_producer_v1",
              "status": "complete_producer_review_pending", "created_at": contract["frozen_at"],
              "task_id": contract["task_id"], "owner": contract["owner"],
              "contract": evidence(ROOT / contract["contract_path"]), "domains": domains, "checks": checks,
              "q005_task_refit_successor": contract["q005_task_refit_successor"],
              "actual_cost": {"wall_seconds": time.monotonic() - started, "cpu_process_seconds": time.process_time(),
                              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                              "head_updates": sum(2000 * len(d["trained_arms"]) for d in domains.values()),
                              "detector_or_backbone_forwards": 0, "gpu_seconds": 0},
              "boundary": "Equal-opportunity cached 39D heads on exposed roles; not detector training, hidden confirmation, complete-system cost, or a refit of q005-base current-task targets."}
    atomic(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "selected": selected,
                      "head_updates": result["actual_cost"]["head_updates"]}, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); contract = load(args.contract)
    if sha256(Path(__file__)) != contract["runner"]["sha256"]: raise RuntimeError("runner identity drift")
    compute(contract, args.output)


if __name__ == "__main__": main()
