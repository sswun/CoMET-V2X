"""Exercise the verifier CLI against a one-message synthetic research workspace."""
import json
from pathlib import Path
from runpy import run_path
import subprocess
import sys

import numpy as np

def test_verifier_checks_coverage_and_every_state(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/verify_against_recipe.py"
    verifier = run_path(str(script))
    splits, state_pattern = verifier["SPLITS"], verifier["STATE"]
    module = tmp_path / "preexperiment/research_autopilot/run_peer16_cached_methods_v1_20261103.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import numpy as np\n"
        "def materials(domain):\n"
        "    m = dict(condition_offsets=[0, 1], source=np.array([1]), "
        "cv_boxes=np.zeros((1, 2)), sequence=['s'], wait_ms=[0])\n"
        f"    return ({{s: (m,) for s in {splits!r}}},)\n"
    )
    args = [sys.executable, str(script), "--project", ".", "--domains", "author", "--out", "report.json"]

    def run():
        result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
        assert (tmp_path / "report.json").exists(), result.stderr
        return result.returncode, json.loads((tmp_path / "report.json").read_text())

    code, report = run()
    assert code != 0 and report["_all_equal"] is False and len(report["_errors"]) == len(splits)

    states = dict(delta=np.zeros((1, 2)), observation=np.zeros((1, 2)),
                  anchors=np.array([0]), variance=np.array([1.0025]))
    for split in splits:
        path = tmp_path / state_pattern.format(domain="author", split=split)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **states)
    code, report = run()
    assert code == 0 and report["_all_equal"] is True

    target = tmp_path / state_pattern.format(domain="author", split="val21")
    for field in ("variance", "observation"):
        np.savez(target, **{**states, field: states[field] + 1})
        code, report = run()
        assert code != 0 and report["_all_equal"] is False
        assert np.isclose(report["author/val21"][f"max_abs_{field}_diff"], 1.0)
