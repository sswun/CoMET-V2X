"""Two-tier box-level fusion of CoMET late fusion (tier 1) and intermediate fusion (tier 2).

Tier 1 (late fusion with the CoMET-corrected remote boxes) localises well;
tier 2 (an intermediate-fusion detector run with the CoMET-corrected pose)
recalls more objects. The combination, fixed on calibration8 only (R8e):

1. pair tier-1 and tier-2 boxes one-to-one by BEV IoU >= ``iou_threshold``
   (0.3), greedily in descending IoU, ties broken by tier-1 then tier-2 index;
2. a pair keeps the tier-1 box and the larger of the two scores;
3. an unpaired box keeps its own geometry, with its score multiplied by its
   tier's coefficient (clipped least squares of TP70 on score, calibration8);
4. BEV NMS at IoU 0.15, in descending score order (ties by index).

Boxes are ``(x, y, z, h, w, l, yaw)`` in the ego frame. No ground truth is
used at inference; the two coefficients are the only fitted quantities.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["bev_corners", "bev_iou", "fit_unmatched_coefficient", "combine_two_tiers"]


def bev_corners(boxes: np.ndarray) -> np.ndarray:
    """(N, 4, 2) BEV corners of ``(x, y, z, h, w, l, yaw)`` boxes (counter-clockwise)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    template = np.asarray([[1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=np.float64) / 2.0
    corners = template[None] * np.stack([b[:, 5], b[:, 4]], -1)[:, None, :]
    c, s = np.cos(b[:, 6]), np.sin(b[:, 6])
    x = corners[..., 0] * c[:, None] - corners[..., 1] * s[:, None]
    y = corners[..., 0] * s[:, None] + corners[..., 1] * c[:, None]
    return np.stack([x + b[:, None, 0], y + b[:, None, 1]], -1)


def _area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman intersection of two convex counter-clockwise polygons."""
    out = list(subject)
    for k in range(len(clip)):
        a, b = clip[k], clip[(k + 1) % len(clip)]
        if not out:
            break
        inp, out = out, []
        edge = b - a

        def inside(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= 0

        for i in range(len(inp)):
            p, q = inp[i], inp[(i + 1) % len(inp)]
            pin, qin = inside(p), inside(q)
            if pin:
                out.append(p)
            if pin != qin:
                d = q - p
                den = edge[0] * d[1] - edge[1] * d[0]
                if den != 0:
                    t = (edge[1] * (p[0] - a[0]) - edge[0] * (p[1] - a[1])) / den
                    out.append(p + t * d)
    return np.asarray(out).reshape(-1, 2)


def bev_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotated BEV IoU matrix, computed in float64 and returned as float32 (accepted evaluator semantics)."""
    ca, cb = bev_corners(a), bev_corners(b)
    out = np.zeros((len(ca), len(cb)), dtype=np.float32)
    if not len(ca) or not len(cb):
        return out
    amin, amax = ca.min(1), ca.max(1); bmin, bmax = cb.min(1), cb.max(1)
    area_a = np.asarray([_area(p) for p in ca]); area_b = np.asarray([_area(p) for p in cb])
    for i in range(len(ca)):
        possible = np.flatnonzero((bmin[:, 0] <= amax[i, 0]) & (bmax[:, 0] >= amin[i, 0]) &
                                  (bmin[:, 1] <= amax[i, 1]) & (bmax[:, 1] >= amin[i, 1]))
        for j in possible:
            inter = _area(_clip(ca[i], cb[j]))
            union = area_a[i] + area_b[j] - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return out


def fit_unmatched_coefficient(scores: np.ndarray, tp70: np.ndarray) -> float:
    """Clipped least-squares scale ``min(1, max(0, <s, y> / <s, s>))`` on calibration detections."""
    s = np.asarray(scores, float); y = np.asarray(tp70, float)
    return float(np.clip(np.dot(s, y) / max(np.dot(s, s), 1e-30), 0.0, 1.0))


def combine_two_tiers(boxes1: np.ndarray, scores1: np.ndarray, boxes2: np.ndarray, scores2: np.ndarray,
                      coef1: float, coef2: float, iou_threshold: float = 0.3,
                      nms_threshold: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse one frame. Returns ``(boxes, scores, origin)``; ``origin[k] = (tier, index)`` with tier 0 or 1."""
    a = np.asarray(boxes1, float).reshape(-1, 7); b = np.asarray(boxes2, float).reshape(-1, 7)
    sa = np.asarray(scores1, float); sb = np.asarray(scores2, float)
    overlap = bev_iou(a, b)
    ii, jj = np.nonzero(overlap >= iou_threshold)
    order = np.lexsort((jj, ii, -overlap[ii, jj]))
    used_a, used_b, pairs = set(), set(), []
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); pairs.append((i, j))
    boxes, scores, origin = [], [], []
    for i, j in pairs:
        boxes.append(a[i]); scores.append(max(sa[i], sb[j])); origin.append((0, i))
    for tier, (bb, ss, used, coef) in enumerate(((a, sa, used_a, coef1), (b, sb, used_b, coef2))):
        for i in range(len(bb)):
            if i not in used:
                boxes.append(bb[i]); scores.append(ss[i] * coef); origin.append((tier, i))
    boxes = np.asarray(boxes, float).reshape(-1, 7); scores = np.asarray(scores, float)
    origin = np.asarray(origin, np.int64).reshape(-1, 2)
    order = np.lexsort((np.arange(len(scores)), -scores)); keep = []
    while len(order):
        i = int(order[0]); keep.append(i); rest = order[1:]
        order = rest[bev_iou(boxes[i:i + 1], boxes[rest])[0] <= nms_threshold] if len(rest) else rest
    return boxes[keep], scores[keep], origin[keep]
