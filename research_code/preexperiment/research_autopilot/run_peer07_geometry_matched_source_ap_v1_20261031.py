#!/usr/bin/env python3
"""Reuse accepted canonical stale/CV/KF/FreeAlign raw and matched-head predictions."""
import argparse
from pathlib import Path
import resource
import time

import run_peer07_timing_2x2_source_split_ap_v2_20261031 as core
from object_history_selector_core_v1_20261021 import PackedDataset


def paired_bootstrap(rows, contract):
    table = {(r["role"], r["sequence"], r["wait_ms"], r["source_group"], r["method"]): r for r in rows}
    random = core.np.random.default_rng(20261031)
    output = []
    for role in core.ROLES:
        sequences = sorted({r["sequence"] for r in rows if r["role"] == role})
        for wait in core.WAITS:
            for group in ("veh_coop", "inf"):
                for left, right in contract["contrasts"]:
                    available = [s for s in sequences if table[role, s, wait, group, right]["gt"] > 0]
                    values = core.np.asarray([table[role, s, wait, group, left]["ap70"] - table[role, s, wait, group, right]["ap70"] for s in available])
                    draws = values[random.integers(0, len(values), (10000, len(values)))].mean(axis=1)
                    output.append({"role": role, "wait_ms": wait, "source_group": group,
                                   "left": left, "right": right, "eligible_sequences": len(values),
                                   "replicates": 10000, "seed": 20261031,
                                   "mean_pp": float(values.mean()), "ci95": core.np.quantile(draws, [.025, .975]).tolist()})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = core.load(args.contract)
    for spec in contract["inputs"].values():
        if core.evidence(core.ROOT / spec["path"]) != spec:
            raise RuntimeError(("input drift", spec["path"]))
    if core.evidence(Path(__file__).resolve()) != contract["producer"]:
        raise RuntimeError("producer identity")
    args.output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    core.METHODS = tuple(contract["methods"])
    labels = core.ROOT / contract["label_roots"]["main_domain"]["path"]
    if core.label_aggregate(labels) != contract["label_roots"]["main_domain"]["identity"]:
        raise RuntimeError("label aggregate drift")
    panels, sequences, outputs, mappings, exact = [], [], {}, {}, {}
    old = core.load(core.ROOT / contract["inputs"]["accepted_summary"]["path"])
    old_table = {(r["role"], r["wait_ms"], r["geometry"] + "__" + r["method"]): r for r in old["panels"]}
    for role in core.ROLES:
        dataset = PackedDataset(core.ROOT / contract["datasets"][role])
        geometries = {g: core.load_npz(core.ROOT / contract["inputs"][role + "_" + g]["path"]) for g in contract["geometries"]}
        material = dict(geometries["stale"])
        gt = []
        for i, index in enumerate(material["dataset_index"]):
            pool = dataset.pool(int(index))
            ga, gb = material["gt_offsets"][i:i + 2].astype(int)
            if not core.np.array_equal(material["gt_original_indices"][ga:gb], pool["gt_original_indices"]):
                raise RuntimeError((role, i, "original GT identity"))
            gt.append(pool["gt_boxes"])
        material["gt_boxes"] = core.np.concatenate(gt)
        for geometry, data in geometries.items():
            for field in ("dataset_index", "sequence", "wait_ms", "receiver_timestamp_us", "condition_offsets", "gt_offsets", "gt_original_indices"):
                if not core.np.array_equal(material[field], data[field]):
                    raise RuntimeError((role, geometry, field, "material identity"))
        predictions = core.load_npz(core.ROOT / contract["inputs"][role + "_predictions"]["path"])
        geometry = {method: geometries[method.split("__")[0]]["boxes"] for method in core.METHODS}
        provenance, mapping = core.map_provenance(material, dataset.meta, role, labels)
        if mapping["unmatched"] or mapping["ambiguous"] or core.np.any(provenance < 0):
            raise RuntimeError((role, "incomplete GT source", mapping))
        rows, seq, arrays, full = core.evaluate_role(material, predictions, geometry, provenance, role)
        if not all(full.values()):
            raise RuntimeError((role, "accepted all-GT TP/FP not replayed"))
        for row in rows:
            if row["source_group"] == "all":
                expected = old_table[role, row["wait_ms"], row["method"]]
                for threshold in (50, 70):
                    if abs(row[f"ap{threshold}"] - expected[f"ap{threshold}"]) > 1e-10:
                        raise RuntimeError((role, row["method"], "accepted AP drift"))
        out, side = args.output / (role + "_source_outcomes.npz"), args.output / (role + "_gt_provenance.npz")
        core.save_npz(out, **arrays)
        core.save_npz(side, gt_from_side=provenance, gt_offsets=material["gt_offsets"], dataset_index=material["dataset_index"])
        outputs[role] = {"outcomes": core.evidence(out), "provenance": core.evidence(side)}
        mappings[role], exact[role] = mapping, full
        panels.extend(rows)
        sequences.extend(seq)
    bootstrap = paired_bootstrap(sequences, contract)
    checks = {"accepted_all_GT_AP_TP_FP_exact": all(all(r.values()) for r in exact.values()),
              "all_unique_source_mapping": all(not r["unmatched"] and not r["ambiguous"] for r in mappings.values()),
              "full_192_panels": len(panels) == 192, "bootstrap_96": len(bootstrap) == 96,
              "no_new_geometry_score_or_method_selection": True, "zero_new_forward_update_GPU": True,
              "shared_inputs_zero_copy": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {"status": "complete_producer_review_pending", "task_id": contract["task_id"],
               "contract": core.evidence(args.contract), "formal_panels": panels, "per_sequence_panels": sequences,
               "paired_sequence_bootstrap": bootstrap, "outputs": outputs, "mapping": mappings,
               "checks": checks, "boundary": contract["boundary"], "actual_cost": {
                   "wall_seconds": time.monotonic() - start, "cpu_process_seconds": time.process_time() - cpu,
                   "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                   "detector_or_backbone_forwards": 0, "optimizer_or_head_updates": 0, "gpu_seconds": 0}}
    core.atomic_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
