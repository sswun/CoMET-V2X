"""Native dependency and train-only cost probe, never a CoDynTrust result.

Use an isolated output directory for every attempt. Official sources are read
only. The OPV2V recipe does not use database sampling; its unrelated OpenPCDet
dependency is deferred, not replaced by a fake implementation.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "third_party/research_baselines_20260930"
NATIVE = ROOT / ".conda/codyntrust-native-20261004"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def extension(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare_runtime(build=False):
    import torch
    from torch.utils.cpp_extension import load

    manifests = []
    for op, name in [("roiaware_pool3d", "roiaware_pool3d_cuda"),
                     ("iou3d_nms", "iou3d_nms_cuda")]:
        src = SOURCES / "INSTINCT/opencood/pcdet_utils" / op / "src"
        sibling = SOURCES / "CoDynTrust/opencood/pcdet_utils" / op / "src"
        source_hashes = {p.name: digest(p) for p in sorted(src.iterdir()) if p.is_file()}
        assert all(digest(sibling / name) == sha for name, sha in source_hashes.items())
        dest = NATIVE / op
        dest.mkdir(parents=True, exist_ok=True)
        if build:
            mod = load(name=name, sources=[str(p) for p in sorted(src.iterdir())
                       if p.suffix in (".cpp", ".cu")], build_directory=str(dest),
                       extra_cflags=["-O2"], extra_cuda_cflags=["-O2"],
                       extra_ldflags=["-L" + str(ROOT / ".conda/cuda-11.8-build/lib"),
                                      "-Wl,-rpath," + str(ROOT / ".conda/cuda-11.8-build/lib")],
                       verbose=True)
            binary = Path(mod.__file__)
        else:
            binary = dest / (name + ".so")
        package = "opencood.pcdet_utils." + op
        importlib.import_module(package)
        extension(package + "." + name, binary)
        manifests.append({"operation": op, "source_hashes": source_hashes,
                          "binary": str(binary), "binary_sha256": digest(binary)})

    bbox = SOURCES / "CoST/opencood/utils/box_overlaps.cpython-39-x86_64-linux-gnu.so"
    source_sha = digest(SOURCES / "CoST/opencood/utils/box_overlaps.pyx")
    assert source_sha == digest(SOURCES / "INSTINCT/opencood/utils/box_overlaps.pyx")
    assert source_sha == digest(SOURCES / "CoDynTrust/opencood/utils/box_overlaps.pyx")
    importlib.import_module("opencood.utils")
    extension("opencood.utils.box_overlaps", bbox)
    manifests.append({"operation": "box_overlaps", "source_sha256": source_sha,
                      "binary": str(bbox), "binary_sha256": digest(bbox)})

    package = importlib.import_module("opencood.data_utils.augmentor")
    name = "opencood.data_utils.augmentor.database_sampler"
    spec = importlib.util.spec_from_file_location(
        name, SOURCES / "INSTINCT/opencood/data_utils/augmentor/database_sampler.py")
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(package, "database_sampler", module)
    return manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--train-probe", action="store_true")
    parser.add_argument("--forward-probe", action="store_true")
    parser.add_argument("--d3d-binary")
    parser.add_argument("--d3d-review")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(),
              "purpose": "native dependency and train-only resource qualification",
              "scientific_performance": False, "full_training_released": False,
              "official_sources_modified": False, "steps": [], "status": "running"}
    try:
        import numpy as np
        import torch
        random.seed(20261004)
        np.random.seed(20261004)
        torch.manual_seed(20261004)
        torch.set_num_threads(2)
        result["native_dependencies"] = prepare_runtime(build=args.build)
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.data_utils.datasets import build_dataset
        from opencood.tools import train_utils
        config = SOURCES / "INSTINCT/opencood/hypes_yaml/opv2v/lidar_only_with_noise/coalign/pointpillar_uncertainty.yaml"
        cfg = load_yaml(str(config))
        assert not any(x["NAME"] == "gt_sampling" for x in cfg["data_augment"])
        root = ROOT / "artifacts/data/opv2v/full"
        cfg.update(root_dir=str(root / "train"), validate_dir=str(root / "validate"),
                   test_dir=str(root / "test"), data_dir=str(root / "train"))
        result.update(config=str(config), config_sha256=digest(config),
                      config_changes="local paths and missing data_dir only; diagnostic batch=2 instead of published16",
                      optional_dependency="OpenPCDet database_sampler import is lazy; no gt_sampling in this recipe")
        started = time.monotonic()
        dataset = build_dataset(cfg, visualize=False, train=True)
        result.update(train_frames=len(dataset), dataset_build_seconds=time.monotonic()-started)
        assert len(dataset) > 500
        indices = [0, 1, 418, 419, 492, 493, len(dataset)-2, len(dataset)-1]
        result["frozen_train_indices"] = indices
        save(output / "status.json", result)
        model = None
        if args.train_probe or args.forward_probe:
            assert torch.cuda.is_available()
            model = train_utils.create_model(cfg).cuda().train()
            result["gpu"] = torch.cuda.get_device_name(0)
        if args.train_probe:
            assert args.d3d_binary and args.d3d_review
            from codyntrust_d3d_math_adapter_20261004 import install_legacy_mathh
            result["d3d_compatibility"] = install_legacy_mathh(
                args.d3d_binary, args.d3d_review)
            criterion = train_utils.create_loss(cfg).cuda()
            optimizer = train_utils.setup_optimizer(cfg, model)
        for step in range(4):
            start = time.monotonic()
            batch = dataset.collate_batch_train([dataset[i] for i in indices[2*step:2*step+2]])
            assert batch is not None
            row = {"step": step, "indices": indices[2*step:2*step+2],
                   "loader_seconds": time.monotonic()-start}
            if args.train_probe:
                optimizer.zero_grad(set_to_none=True)
                batch = train_utils.to_device(batch, torch.device("cuda:0"))
                torch.cuda.synchronize()
                start = time.monotonic()
                prediction = model(batch["ego"])
                loss = criterion(prediction, batch["ego"]["label_dict"])
                assert torch.isfinite(loss), "nonfinite native training loss"
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(x).all().item() for x in grads)
                optimizer.step()
                torch.cuda.synchronize()
                row.update(training_seconds=time.monotonic()-start,
                           loss_diagnostic_only=float(loss.detach().cpu()),
                           max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           finite_gradients=True)
            elif args.forward_probe:
                batch = train_utils.to_device(batch, torch.device("cuda:0"))
                torch.cuda.synchronize()
                start = time.monotonic()
                with torch.no_grad():
                    prediction = model(batch["ego"])
                torch.cuda.synchronize()
                tensors = {k: v for k, v in prediction.items() if torch.is_tensor(v)}
                assert tensors and all(torch.isfinite(v).all().item() for v in tensors.values())
                row.update(forward_seconds=time.monotonic()-start,
                           output_shapes={k: list(v.shape) for k, v in tensors.items()},
                           max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           training_cost_measured=False)
            result["steps"].append(row)
            save(output / "status.json", result)
        result.update(status="passed", checkpoints_saved=False,
                      limitation="Eight train examples, random initialization, batch2; no AP or validation/test. Not an epoch estimate or batch16 qualification.")
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["finished_at"] = datetime.now().astimezone().isoformat()
        save(output / "status.json", result)


if __name__ == "__main__":
    main()
