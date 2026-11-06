"""Check comet/estimator.py against the accepted H1 recipe states used in the paper.

Run inside the original research environment (dataset, cached detector
materials and research_code on the path). For each detector domain and split it
rebuilds the per-message offsets with the clean implementation and compares
them with the accepted ``{split}_state.npz`` of the H1 recipe.
Missing splits, unequal anchors or numerical differences of 1e-9 or more
make the verification fail with a nonzero exit status.

    python tools/verify_against_recipe.py --project /path/to/project --out verify.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from comet.estimator import estimate_offsets  # noqa: E402

# q selected on calibration8 only (R1), r = 0.6, gate = 2.5 m for every domain.
Q = {"author": 0.05, "strict60": 0.025, "v2xvit_seed2024": 0.1, "v2xvit_seed2025": 0.05, "where2comm": 0.05}
SPLITS = ("fit24", "calibration8", "val21", "mechanism_holdout14")
STATE = "artifacts/research_autopilot/status_audit_20261104_peer19/recipe_v3_interface_recovery_v2/{domain}/{split}_state.npz"


def messages(m):
    for k, (a0, b0) in enumerate(zip(m["condition_offsets"][:-1], m["condition_offsets"][1:])):
        a, b = int(a0), int(b0)
        src = m["source"][a:b]
        remote = m["cv_boxes"][a:b][src == 1]
        ego = m["cv_boxes"][a:b][src == 0]
        yield (str(m["sequence"][k]), int(m["wait_ms"][k])), remote, ego


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, type=Path)
    p.add_argument("--domains", nargs="+", choices=list(Q), default=list(Q))
    p.add_argument("--out", type=Path, default=Path("verify_against_recipe.json"))
    a = p.parse_args()
    a.project = a.project.resolve()
    os.chdir(a.project)
    for sub in ("preexperiment/research_autopilot", "preexperiment/research_baselines", "preexperiment"):
        sys.path.insert(0, str(a.project / sub))
    from run_peer16_cached_methods_v1_20261103 import materials

    report = {}
    errors = {}
    for domain in a.domains:
        data = materials(domain)[0]
        for split in SPLITS:
            key = f"{domain}/{split}"
            path = a.project / STATE.format(domain=domain, split=split)
            if split not in data or not path.exists():
                errors[key] = "Missing detector materials or saved state"
                continue
            m = data[split][0]
            est = estimate_offsets(messages(m), q=Q[domain], r=0.6, gate=2.5)
            with np.load(path) as z:
                ref = {k: z[k] for k in z.files}
            if len(est["delta"]) == 0 or any(k not in ref or ref[k].shape != v.shape for k, v in est.items()):
                errors[key] = "Empty replay or missing or differently shaped state arrays"
                continue
            row = dict(
                messages=int(len(est["delta"])),
                max_abs_delta_diff=float(np.abs(est["delta"] - ref["delta"]).max()),
                max_abs_variance_diff=float(np.abs(est["variance"] - ref["variance"]).max()),
                anchors_equal=bool(np.array_equal(est["anchors"], ref["anchors"])),
                max_abs_observation_diff=float(np.abs(est["observation"] - ref["observation"]).max()),
            )
            report[key] = row
            print(domain, split, row, flush=True)
    diff_fields = ("max_abs_delta_diff", "max_abs_variance_diff", "max_abs_observation_diff")
    ok = not errors and all(
        r["anchors_equal"] and all(r[field] < 1e-9 for field in diff_fields)
        for r in report.values()
    )
    if errors:
        report["_errors"] = errors
    report["_all_equal"] = ok
    a.out.write_text(json.dumps(report, indent=1))
    print("ALL EQUAL" if ok else "MISMATCH", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
