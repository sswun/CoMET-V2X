"""E1 fixed-reference candidate utilities. No detector training or inference."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from audit_v2xseq_strict_common_roi_20261010 import mask as common_mask
from fit_evaluate_candidate_d_heads_v1_20261019 import feature_matrix
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import polygons_hwl, voc_ap

WAITS = (0, 100, 200, 400)
ROLES = ("fit24", "calibration8", "mechanism_holdout14", "val21")
LEARNED = ("quality", "direct_utility", "factorized", "source_logistic")
BASES = ("ego", "all_ego_source", "all_ego_conflict015")
POLICIES = LEARNED + ("factorized_all_ego_readout",)
BETAS = (1.0, 1.2, 1.4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def append(stream: Any, value: Any) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def iou32(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """The accepted evaluator's polygon construction and float32 IoU semantics."""
    bp, gp = polygons_hwl(boxes), polygons_hwl(gt)
    out = np.zeros((len(bp), len(gp)), dtype=np.float32)
    if not len(bp) or not len(gp):
        return out
    bounds = np.asarray([p.bounds for p in gp])
    for i, a in enumerate(bp):
        ax0, ay0, ax1, ay1 = a.bounds
        possible = np.flatnonzero((bounds[:, 0] <= ax1) & (bounds[:, 2] >= ax0) &
                                  (bounds[:, 1] <= ay1) & (bounds[:, 3] >= ay0))
        for j in possible:
            b = gp[j]
            union = a.union(b).area
            out[i, j] = a.intersection(b).area / union if union else 0.0
    return out


def greedy(matrix: np.ndarray, order: np.ndarray, threshold: float) -> np.ndarray:
    """Return matched current GT index for each sorted prediction, else -1."""
    remaining = list(range(matrix.shape[1]))
    matches = np.full(len(order), -1, dtype=np.int32)
    for out_i, row in enumerate(order):
        if not remaining:
            break
        values = matrix[int(row), remaining]
        best = int(np.argmax(values))
        if float(values[best]) >= threshold:
            matches[out_i] = remaining.pop(best)
    return matches


def unique_ego_scores(raw: np.ndarray, stable: bool = False) -> np.ndarray:
    """Resolve exact local score ties in the original evaluator's ego order.

    float32 detector scores have enough float64 space for this deterministic
    tie encoding. Non-tied scores and all geometries are unchanged.
    """
    result = raw.astype(np.float64).copy()
    order = np.argsort(-raw, kind="stable") if stable else np.argsort(-raw)
    for k in range(1, len(order)):
        prev, cur = int(order[k - 1]), int(order[k])
        if result[cur] >= result[prev]:
            result[cur] = np.nextafter(result[prev], -np.inf)
    return result


def remote_indices(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    keep = np.flatnonzero(common_mask(boxes) & (labels.astype(int) == 0))
    order = np.argsort(-scores[keep], kind="stable")
    matrix = iou32(boxes[keep], boxes[keep])
    chosen: list[int] = []
    for j in order:
        if chosen and float(matrix[int(j), chosen].max(initial=0.0)) > 0.15:
            continue
        chosen.append(int(j))
    return keep[np.asarray(chosen, dtype=np.int32)], {
        "raw_remote": len(boxes), "common_roi_remote": len(keep),
        "remote_self_nms_removed": len(keep) - len(chosen),
    }


def make_condition(ego: dict[str, np.ndarray], source: dict[str, np.ndarray] | None,
                   transformed: np.ndarray | None, transform: np.ndarray | None,
                   age: float, protect_threshold: float) -> dict[str, np.ndarray | dict[str, int]]:
    em = common_mask(ego["pred_boxes"]) & (ego["pred_labels"].astype(int) == 0)
    ei = np.flatnonzero(em).astype(np.int32)
    eb = ego["pred_boxes"][em].astype(np.float32)
    es = ego["pred_scores"][em].astype(np.float64)
    gm = common_mask(ego["gt_boxes"]) & (ego["gt_labels"].astype(int) == 0)
    gi = np.flatnonzero(gm).astype(np.int32)
    gb = ego["gt_boxes"][gm].astype(np.float32)
    protected = es >= protect_threshold
    sb, ss = eb[protected], es[protected]
    ef = feature_matrix(eb, eb, es, 0.0, np.eye(4), sb, ss)
    if source is None:
        ri = np.empty(0, dtype=np.int32)
        rb, rf = np.empty((0, 7), np.float32), np.empty((0, 30), np.float64)
        rs = np.empty(0, np.float64)
        diag = {"raw_remote": 0, "common_roi_remote": 0, "remote_self_nms_removed": 0}
    else:
        assert transformed is not None and transform is not None
        ri, diag = remote_indices(transformed, source["pred_scores"], source["pred_labels"])
        rb = transformed[ri].astype(np.float32)
        rs = source["pred_scores"][ri].astype(np.float64)
        rf = feature_matrix(rb, source["pred_boxes"][ri], rs, age, transform, sb, ss)
    boxes = np.concatenate((eb, rb))
    raw = np.r_[es, rs]
    is_remote = np.r_[np.zeros(len(eb), np.int8), np.ones(len(rb), np.int8)]
    features = np.column_stack((np.concatenate((ef, rf)), is_remote)).astype(np.float32)
    sm = np.r_[protected, np.zeros(len(rb), bool)]
    matrix = iou32(boxes, gb)
    # Encode ties AFTER putting the prefix in its final numeric interval.
    # Adding 2 after nextafter(raw) would round those tiny differences away.
    es_unique = unique_ego_scores(2.0 + es)
    sids = np.flatnonzero(sm)
    sorder = sids[np.argsort(-es_unique[sids])]
    smatches = greedy(matrix, sorder, 0.7)
    covered = set(int(x) for x in smatches if x >= 0)
    q = (matrix.max(axis=1, initial=0.0) >= 0.7).astype(np.int8)
    uncovered = [j for j in range(len(gb)) if j not in covered]
    u = (matrix[:, uncovered].max(axis=1, initial=0.0) >= 0.7).astype(np.int8)
    assert np.all(u <= q)
    return {
        "boxes": boxes, "raw_scores": raw, "ego_order_scores": np.r_[es_unique, rs],
        "is_remote": is_remote, "is_protected": sm.astype(np.int8),
        "source_detection_indices": np.r_[ei, ri].astype(np.int32),
        "features": features, "gt_boxes": gb, "gt_original_indices": gi,
        "ious": matrix, "quality_y": q, "utility_y": u,
        "diagnostic": diag,
    }


def pool_order(pool: dict[str, np.ndarray], scores: dict[str, np.ndarray], name: str,
               threshold: float | None = None, fallback: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Candidate row IDs and actual scalar rank scores; no post-score filtering."""
    e = np.flatnonzero(pool["is_remote"] == 0)
    r = np.flatnonzero(pool["is_remote"] == 1)
    s = np.flatnonzero(pool["is_protected"] == 1)
    if fallback or name == "ego":
        # Preserve original raw scores and global tie semantics for the anchor.
        selected, rank = e, pool["raw_scores"][e].astype(np.float64)
    elif name in ("all_ego_source", "all_ego_conflict015"):
        if name.endswith("conflict015") and len(r) and len(e):
            r = r[iou32(pool["boxes"][r], pool["boxes"][e]).max(axis=1) <= 0.15]
        selected = np.r_[e, r]
        rank = np.r_[pool["ego_order_scores"][e], unique_ego_scores(pool["raw_scores"][r], stable=True)]
    elif name == "factorized_all_ego_readout":
        variable_rank = unique_ego_scores(scores["factorized"][r], stable=True)
        if threshold is not None:
            keep = scores["factorized"][r] >= threshold
            r, variable_rank = r[keep], variable_rank[keep]
        selected = np.r_[e, r]
        rank = np.r_[pool["ego_order_scores"][e], variable_rank]
    else:
        c = np.flatnonzero(pool["is_protected"] == 0)
        variable_rank = unique_ego_scores(scores[name][c], stable=True)
        if threshold is not None:
            keep = scores[name][c] >= threshold
            c, variable_rank = c[keep], variable_rank[keep]
        selected = np.r_[s, c]
        rank = np.r_[pool["ego_order_scores"][s], variable_rank]
    order = np.argsort(-rank)
    return selected[order], rank[order].astype(np.float64)


class PackedDataset:
    def __init__(self, directory: Path):
        self.directory = directory
        self.meta = list(read_rows(directory / "conditions.jsonl"))
        self.data = load_npz(directory / "pools.npz")

    def pool(self, index: int) -> dict[str, np.ndarray]:
        m = self.meta[index]
        a, b, ga, gb = m["start"], m["stop"], m["gt_start"], m["gt_stop"]
        out = {k: self.data[k][a:b] for k in (
            "boxes", "raw_scores", "ego_order_scores", "is_remote", "is_protected",
            "source_detection_indices", "features", "quality_y", "utility_y",
        )}
        out["gt_boxes"] = self.data["gt_boxes"][ga:gb]
        out["gt_original_indices"] = self.data["gt_original_indices"][ga:gb]
        out["ious"] = self.data["ious"][m["iou_start"]:m["iou_stop"]].reshape(b-a, gb-ga)
        return out

    def score_pool(self, index: int, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        m = self.meta[index]
        return {k: v[m["start"]:m["stop"]] for k, v in scores.items()}


class PackWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.stream = (directory / "conditions.jsonl").open("w")
        self.arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        self.cursor = self.gt_cursor = self.iou_cursor = self.conditions = 0
        self.started = time.monotonic()

    def add(self, metadata: dict[str, Any], value: dict[str, Any]) -> None:
        n, g = len(value["boxes"]), len(value["gt_boxes"])
        meta = dict(metadata, condition_index=self.conditions, start=self.cursor, stop=self.cursor+n,
                    gt_start=self.gt_cursor, gt_stop=self.gt_cursor+g,
                    iou_start=self.iou_cursor, iou_stop=self.iou_cursor+n*g,
                    S_ids=np.flatnonzero(value["is_protected"]).astype(int).tolist(),
                    C_ids=np.flatnonzero(value["is_protected"] == 0).astype(int).tolist(),
                    diagnostic=value["diagnostic"])
        append(self.stream, meta)
        for k, v in value.items():
            if k != "diagnostic":
                self.arrays[k].append(v.reshape(-1) if k == "ious" else v)
        self.cursor += n; self.gt_cursor += g; self.iou_cursor += n*g; self.conditions += 1

    def finish(self, provenance: dict[str, Any]) -> dict[str, Any]:
        self.stream.close()
        arrays = {k: np.concatenate(v, axis=0) for k, v in self.arrays.items()}
        np.savez_compressed(self.directory / "pools.npz", **arrays)
        result = dict(provenance, status="complete", conditions=self.conditions, candidates=self.cursor,
                      gt_occurrences=self.gt_cursor, iou_values=self.iou_cursor,
                      wall_seconds=time.monotonic()-self.started,
                      pools_sha256=sha256(self.directory / "pools.npz"),
                      conditions_sha256=sha256(self.directory / "conditions.jsonl"))
        atomic(self.directory / "summary.json", result)
        return result


def metrics(tp: np.ndarray, scores: np.ndarray, gt: int) -> dict[str, Any]:
    return voc_ap(tp.astype(int).tolist(), (1-tp).astype(int).tolist(), scores.astype(float).tolist(), gt, True)


def error_counts(matrix: np.ndarray, order: np.ndarray, matches: np.ndarray) -> dict[str, int]:
    maxima = matrix[order].max(axis=1, initial=0.0)
    wrong = matches < 0
    return {"tp": int((~wrong).sum()), "duplicate_fp": int((wrong & (maxima >= .7)).sum()),
            "localization_fp": int((wrong & (maxima >= .1) & (maxima < .7)).sum()),
            "background_fp": int((wrong & (maxima < .1)).sum())}
