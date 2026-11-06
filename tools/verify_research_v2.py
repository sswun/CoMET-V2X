"""Check comet/acquisition.py, comet/stress.py and comet/fusion.py against accepted R8d/R8e outputs.

Run inside the original research environment (dataset, cached detector
materials and research_code on the path), like ``verify_against_recipe.py``:

    python tools/verify_research_v2.py --project /path/to/project --out verify_research_v2.json

Checks
* CoMET-A on real data: five detectors x calibration8/val21/mechanism_holdout14
  against the accepted R8d states (max |delta| difference < 1e-9).
* Injection + CoMET-A: every accepted injected case (T1 3/5 m) - the injected
  shift equals the saved harness and the CoMET-A state equals the saved state.
* Two-tier fusion: the selected R8e arm (IoU 0.3, no score calibration) for
  every accepted val21/mechanism_holdout14 seed - fused scores, origins and
  candidate indices equal the saved predictions frame by frame.
Any mismatch or missing accepted file makes the exit status nonzero.

The parts can run as separate processes and be merged afterwards:

    python tools/verify_research_v2.py --project P --skip-fusion --out a.json
    python tools/verify_research_v2.py --project P --skip-acquisition --fusion val21:20261101 --out f1.json
    python tools/verify_research_v2.py --merge a.json f1.json ... --out verify_research_v2.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from comet.acquisition import estimate_offsets_with_acquisition  # noqa: E402
from comet.fusion import combine_two_tiers  # noqa: E402
from comet.stress import injection_shifts  # noqa: E402

Q = {"author": 0.05, "strict60": 0.025, "v2xvit_seed2024": 0.1, "v2xvit_seed2025": 0.05, "where2comm": 0.05}
ROLES = ("calibration8", "val21", "mechanism_holdout14")
P26 = "artifacts/research_autopilot/status_audit_20261106_peer26"
SEEDS = (20261101, 20261102, 20261103)


def messages(m):
    for k, (a0, b0) in enumerate(zip(m["condition_offsets"][:-1], m["condition_offsets"][1:])):
        a, b = int(a0), int(b0)
        src = m["source"][a:b]
        yield (str(m["sequence"][k]), int(m["wait_ms"][k])), m["cv_boxes"][a:b][src == 1], m["cv_boxes"][a:b][src == 0]


def inject(m, shift):
    out = dict(m)
    remote = m["source"] == 1
    rep = np.repeat(shift, np.diff(m["condition_offsets"]).astype(int), axis=0)
    for key in ("stale_boxes", "cv_boxes", "huber_boxes", "supervised_constant_boxes", "kf_boxes"):
        if key in m:
            out[key] = m[key].copy()
            out[key][remote, :2] += rep[remote].astype(out[key].dtype)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path)
    p.add_argument("--out", type=Path, default=Path("verify_research_v2.json"))
    p.add_argument("--skip-fusion", action="store_true")
    p.add_argument("--skip-acquisition", action="store_true")
    p.add_argument("--fusion", nargs="+", default=None, help="role:seed subset, e.g. val21:20261101")
    p.add_argument("--merge", nargs="+", type=Path, default=None, help="merge partial reports instead of running")
    a = p.parse_args()
    out_path = a.out.resolve()
    if a.merge:
        report, errors = {}, {}
        for part in a.merge:
            d = json.loads(part.read_text())
            errors.update(d.pop("_errors", {})); d.pop("_all_equal", None); d.pop("_parts", None)
            report.update(d)
        return finish(report, errors, out_path, parts=[x.name for x in a.merge])
    if a.project is None:
        p.error("--project is required unless --merge is given")
    a.project = a.project.resolve()
    os.chdir(a.project)
    for sub in ("preexperiment/research_autopilot", "preexperiment/research_baselines", "preexperiment"):
        sys.path.insert(0, str(a.project / sub))
    # Accepted R8 loader: recipe materials, with the four-wait strict60 calibration8 material.
    from peer20_population_core_v1_20261105 import data as materials

    report, errors = {}, {}
    cache = {}
    for domain in ([] if a.skip_acquisition else Q):
        data = materials(domain)[0]
        cache[domain] = data
        for role in ROLES:
            key = f"CoMET_A/real/{domain}/{role}"
            path = a.project / P26 / "R8d" / domain / f"{role}_state.npz"
            if role not in data or not path.exists():
                errors[key] = "missing materials or accepted state"; continue
            m = data[role][0]
            with np.load(path) as z:
                ref = z["delta"]
            if len(ref) != len(m["sequence"]):
                errors[key] = f"accepted state has {len(ref)} rows, materials {len(m['sequence'])}"; continue
            est = estimate_offsets_with_acquisition(messages(m), q=Q[domain])
            report[key] = dict(messages=int(len(ref)), triggers=est["triggers"],
                               max_abs_delta_diff=float(np.abs(est["delta"] - ref).max()))
            print(key, report[key], flush=True)

    # Accepted injected cases: material v2 where it exists (v1 strict60 cases stopped on a missing array).
    cases = {}
    for folder in ("R8d", "R8d_material_v2"):
        for case in sorted((a.project / P26 / folder / "cases").glob("*_T1_*")):
            if (case / "states.npz").exists():
                cases[case.name] = case
    for case in ([] if a.skip_acquisition else cases.values()):
        domain, rest = case.name.split("_calibration8_") if "_calibration8_" in case.name else case.name.split("_val21_")
        role = "calibration8" if "_calibration8_" in case.name else "val21"
        amp = float(rest.split("_")[1])
        key = f"CoMET_A/injected/{case.parent.parent.name}/{case.name}"
        if not (case / "states.npz").exists() or role not in cache.get(domain, {}):
            errors[key] = "missing accepted states or materials"; continue
        m = cache[domain][role][0]
        shift = injection_shifts(m["sequence"], m["receiver_timestamp_us"], "T1", amp)
        with np.load(case / "states.npz") as z:
            ref = z["CoMET_A"]
        row = dict(messages=int(len(ref)))
        if (case / "injection_harness.npz").exists():
            with np.load(case / "injection_harness.npz") as z:
                row["max_abs_shift_diff"] = float(np.abs(z["shift"] - shift).max())
        if len(ref) != len(m["sequence"]):
            errors[key] = "row count differs from materials"; continue
        est = estimate_offsets_with_acquisition(messages(inject(m, shift)), q=Q[domain])
        row.update(triggers=est["triggers"], max_abs_delta_diff=float(np.abs(est["delta"] - ref).max()))
        report[key] = row
        print(key, row, flush=True)

    if not a.skip_fusion:
        import run_peer26_R8e_v1_20261107 as r8e
        models = json.loads((a.project / P26 / "R8e" / "score_calibration.json").read_text())["models"]
        wanted = [(r, s) for r in ("val21", "mechanism_holdout14") for s in SEEDS]
        if a.fusion:
            wanted = [(x.split(":")[0], int(x.split(":")[1])) for x in a.fusion]
        for role, seed in wanted:
            key = f"fusion/{role}/{seed}"
            path = None
            for d in (a.project / P26 / "R8e_recovery_v2", a.project / P26 / "R8e"):
                if (d / f"{role}_{seed}_iou0.3_cal0" / "predictions.npz").exists():
                    path = d / f"{role}_{seed}_iou0.3_cal0" / "predictions.npz"; break
            if path is None:
                errors[key] = "missing accepted predictions"; continue
            coef = [float(t["coeff"]["False"]) for t in models[str(seed)]]
            _, _, frames, _, _ = r8e.sources(role, seed)
            with np.load(path) as z:
                ref = {k: z[k] for k in z.files}
            bad = 0; n = 0
            for f, (b1, s1, b2, s2, _gt, ai, bi) in enumerate(frames):
                boxes, scores, origin = combine_two_tiers(b1, s1, b2, s2, coef[0], coef[1])
                lo, hi = int(ref["combo__offset"][f]), int(ref["combo__offset"][f + 1])
                cand = np.asarray([ai[j] if t == 0 else bi[j] for t, j in origin], np.int64)
                same = (hi - lo == len(scores) and np.array_equal(ref["combo__origin"][lo:hi], origin[:, 0])
                        and np.array_equal(ref["combo__candidate"][lo:hi], cand)
                        and np.allclose(ref["combo__score"][lo:hi], scores, rtol=0, atol=1e-12))
                bad += int(not same); n += 1
            report[key] = dict(frames=n, mismatched_frames=bad, accepted=str(path.relative_to(a.project)))
            print(key, report[key], flush=True)

    return finish(report, errors, out_path)


def finish(report, errors, out_path, parts=None):
    ok = bool(report) and not errors and all(
        r.get("max_abs_delta_diff", 0.0) < 1e-9 and r.get("max_abs_shift_diff", 0.0) < 1e-9 and r.get("mismatched_frames", 0) == 0
        for r in report.values())
    report = dict(sorted(report.items()))
    if errors:
        report["_errors"] = errors
    if parts:
        report["_parts"] = parts
    report["_all_equal"] = ok
    out_path.write_text(json.dumps(report, indent=1))
    print("ALL EQUAL" if ok else "MISMATCH", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
