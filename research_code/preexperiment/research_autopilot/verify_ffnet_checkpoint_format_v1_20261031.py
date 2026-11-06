#!/usr/bin/env python3
"""CPU-only structural verifier for the acquired author FFNet checkpoint."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any, Mapping


ROOT = Path("/path/to/project")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def container_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    record: dict[str, Any] = {
        "schema": "ffnet_checkpoint_cpu_format_verification_v1",
        "status": "running",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "boundary": ("CPU deserialization and state-dict structure only; not source/license clearance, "
                     "model compatibility, inference, AP, or redistribution authorization."),
    }
    exit_code = 1
    try:
        if not checkpoint.is_file() or checkpoint.stat().st_size < 100_000:
            raise RuntimeError("checkpoint missing or unexpectedly small")
        with checkpoint.open("rb") as stream:
            prefix = stream.read(4096).lstrip().lower()
        if prefix.startswith((b"<!doctype html", b"<html", b"<?xml", b"quota exceeded")):
            raise RuntimeError("checkpoint prefix is textual HTML/XML/quota evidence")
        record["identity"] = {"bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint)}

        import torch

        value = torch.load(str(checkpoint), map_location="cpu")
        record["torch"] = {"version": torch.__version__, "container_type": container_name(value)}
        if not isinstance(value, Mapping):
            raise RuntimeError(f"top-level checkpoint is not a mapping: {container_name(value)}")
        state = value.get("state_dict", value)
        if not isinstance(state, Mapping):
            raise RuntimeError("checkpoint state_dict is absent or not a mapping")
        keys = [str(key) for key in state.keys()]
        tensors = [(str(key), tensor) for key, tensor in state.items() if torch.is_tensor(tensor)]
        if not tensors or len(tensors) < max(1, len(state) // 2):
            raise RuntimeError("checkpoint mapping does not contain a plausible tensor state_dict")
        nonfinite = []
        numel = 0
        dtypes: dict[str, int] = {}
        for key, tensor in tensors:
            numel += int(tensor.numel())
            dtype = str(tensor.dtype)
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
                nonfinite.append(key)
                if len(nonfinite) >= 20:
                    break
        if nonfinite:
            raise RuntimeError(f"non-finite tensors in checkpoint: {nonfinite}")
        record["structure"] = {
            "top_level_keys": sorted(str(key) for key in value.keys()),
            "state_dict_entries": len(state),
            "tensor_count": len(tensors),
            "tensor_numel": numel,
            "dtype_tensor_counts": dtypes,
            "first_state_keys": sorted(keys)[:40],
            "has_meta_mapping": isinstance(value.get("meta"), Mapping),
        }
        record["status"] = "passed_cpu_deserialization_and_tensor_structure"
        exit_code = 0
    except Exception as exc:
        record["status"] = "failed_closed"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    atomic_json(output, record)
    print(json.dumps({"status": record["status"], "output": str(output)}, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
