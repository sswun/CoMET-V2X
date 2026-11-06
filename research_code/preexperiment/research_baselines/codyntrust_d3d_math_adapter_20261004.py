"""Disclosed legacy-name adapter for the qualified upstream d3d math binary."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


BINARY_SHA256 = "fbe24985fb8f618acb1d4ee014196987799072773360d99954f21cf8a2931ecb"
REVIEW_SHA256 = "36cd6a4f3e7c8f8a26c64333a6bfc5732666ef1b63819f5b118d394df653ff97"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_legacy_mathh(binary, review):
    """Expose upstream d3d.math raw calls under the method's stale mathh name."""
    binary = Path(binary).resolve()
    review = Path(review).resolve()
    assert digest(binary) == BINARY_SHA256
    assert digest(review) == REVIEW_SHA256
    accepted = json.loads(review.read_text())
    assert accepted["status"] == "passed"
    assert accepted["binary_sha256"] == BINARY_SHA256
    assert "reject a plain differentiable torch.special replacement" in accepted["decision"]

    name = "codyntrust_d3d_math_v030"
    spec = importlib.util.spec_from_file_location(name, binary)
    native = importlib.util.module_from_spec(spec)
    sys.modules[name] = native
    spec.loader.exec_module(native)
    assert native.cuda_available

    package = types.ModuleType("d3d")
    package.__path__ = []
    mathh = types.ModuleType("d3d.mathh")
    for function in ("i0e", "i1e", "i0e_cuda", "i1e_cuda"):
        setattr(mathh, function, getattr(native, function))
    package.mathh = mathh
    sys.modules["d3d"] = package
    sys.modules["d3d.mathh"] = mathh
    return {
        "status": "qualified_legacy_module_alias_installed",
        "legacy_import": "d3d.mathh",
        "upstream_module": "d3d.math at v0.3.0",
        "binary": str(binary),
        "binary_sha256": BINARY_SHA256,
        "offline_review": str(review),
        "offline_review_sha256": REVIEW_SHA256,
        "gradient_semantics": "raw upstream i0e_cuda is detached; no differentiable torch.special substitution",
        "official_method_sources_modified": False,
    }
