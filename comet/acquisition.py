"""CoMET-A: lock acquisition for offsets beyond the association gate.

CoMET (``comet.estimator``) associates remote and ego boxes on the uncorrected
geometry with a 2.5 m gate. If the true offset of a stream is larger than the
gate, no anchor is ever matched and the filter stays at zero. CoMET-A adds a
lost-lock detector and a coarse offset search; when it never triggers it is
identical to CoMET.

* lost lock: ``K = 5`` consecutive messages in which both agents have at least
  three boxes but no mutual-nearest pair lies within the gate;
* acquisition: all ego-minus-remote displacements with norm <= 10 m over the
  last ``W = 5`` messages (current one included) are voted in a 0.5 m 2-D
  histogram; the coarse offset is the mean of the displacements within 1.0 m
  of the highest bin. It is accepted only with >= 6 supporting pairs from
  >= 3 messages, norm >= 2.0 m, and a peak at least 1.5 times the highest bin
  that is not adjacent to it;
* after acquisition: the filter is reset to mean ``x_acq`` and variance 0.25,
  and association uses the remote boxes shifted by the *fixed* ``x_acq`` (not
  the running estimate, so the filter cannot confirm itself); the observation
  is ``x_acq`` plus the mean residual. A later loss of lock re-acquires.

All parameters were fixed before any result was read (pre-registered R8d) and
are not tuned. No ground truth, future data or per-pair offsets are used.
``tools/verify_research_v2.py`` checks this module against the accepted states.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Hashable, Iterable, Optional, Tuple

import numpy as np

from .estimator import mutual_hungarian

__all__ = ["AcquisitionParams", "acquire_coarse_offset", "CoMETAEstimator", "estimate_offsets_with_acquisition"]


@dataclass(frozen=True)
class AcquisitionParams:
    lost_messages: int = 5      # K
    window: int = 5             # W
    radius: float = 10.0        # displacement search radius (m)
    bin_size: float = 0.5       # histogram bin (m)
    near: float = 1.0           # support radius around the peak bin centre (m)
    min_pairs: int = 6
    min_messages: int = 3
    min_norm: float = 2.0       # only offsets the gate cannot reach
    peak_ratio: float = 1.5
    reset_variance: float = 0.25
    min_boxes: int = 3          # both agents need this many boxes for a message to count as lost


def acquire_coarse_offset(window: Iterable[Tuple[np.ndarray, np.ndarray]],
                          params: AcquisitionParams = AcquisitionParams()) -> Optional[np.ndarray]:
    """Coarse offset from a window of ``(remote_boxes, ego_boxes)`` messages, or ``None``."""
    values, messages = [], []
    for k, (remote, ego) in enumerate(window):
        remote = np.asarray(remote); ego = np.asarray(ego)
        if not len(remote) or not len(ego):
            continue
        d = (ego[None, :, :2].astype(float) - remote[:, None, :2].astype(float)).reshape(-1, 2)
        use = np.linalg.norm(d, axis=1) <= params.radius
        values.extend(d[use]); messages.extend([k] * int(use.sum()))
    if len(values) < params.min_pairs:
        return None
    d = np.asarray(values); messages = np.asarray(messages)
    edges = np.arange(-params.radius, params.radius + params.bin_size, params.bin_size)
    hist = np.histogram2d(d[:, 0], d[:, 1], bins=(edges, edges))[0]
    order = np.argsort(hist.ravel())[::-1]
    a, b = np.unravel_index(order[0], hist.shape)
    centre = np.asarray((edges[0] + params.bin_size * (a + 0.5), edges[0] + params.bin_size * (b + 0.5)))
    mask = np.linalg.norm(d - centre, axis=1) <= params.near
    if mask.sum() < params.min_pairs or len(set(messages[mask].tolist())) < params.min_messages:
        return None
    second = 0.0
    for flat in order[1:]:
        i, j = np.unravel_index(flat, hist.shape)
        if max(abs(i - a), abs(j - b)) > 1:
            second = hist[i, j]
            break
    value = d[mask].mean(0)
    if np.linalg.norm(value) < params.min_norm or (second > 0 and hist[a, b] / second < params.peak_ratio):
        return None
    return value


@dataclass
class _Stream:
    x: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p: float = 1.0
    x_acq: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lost: int = 0
    history: Deque = field(default_factory=deque)
    triggers: int = 0


class CoMETAEstimator:
    """CoMET with lost-lock detection and coarse re-acquisition (one filter per stream)."""

    def __init__(self, q: float = 0.05, r: float = 0.6, gate: float = 2.5, p0: float = 1.0,
                 params: AcquisitionParams = AcquisitionParams()):
        self.q, self.r, self.gate, self.p0, self.params = q, r, gate, p0, params
        self.streams: dict[Hashable, _Stream] = {}

    def _match(self, remote: np.ndarray, ego: np.ndarray, shift: np.ndarray) -> np.ndarray:
        if not len(remote) or not len(ego):
            return np.empty((0, 2))
        moved = np.array(remote, copy=True)
        if np.any(shift):  # same dtype semantics as the accepted run (float32 boxes stay float32)
            moved[:, :2] += np.asarray(shift, dtype=moved.dtype)
        i, j = mutual_hungarian(moved[:, :2], np.asarray(ego)[:, :2], self.gate)
        return np.asarray(ego)[j, :2].astype(float) - moved[i, :2].astype(float)

    def update(self, stream: Hashable, remote_boxes: np.ndarray, ego_boxes: np.ndarray) -> dict:
        """Advance ``stream`` by one message. Boxes are (N, >=2) arrays in the ego frame."""
        s = self.streams.get(stream)
        if s is None:
            s = self.streams[stream] = _Stream(p=float(self.p0), history=deque(maxlen=self.params.window))
        remote_boxes = np.asarray(remote_boxes); ego_boxes = np.asarray(ego_boxes)
        s.history.append((remote_boxes, ego_boxes))
        res = self._match(remote_boxes, ego_boxes, s.x_acq)
        prior = s.p + self.q * self.q
        triggered = False
        if len(res):
            z = s.x_acq + res.mean(0)
            gain = prior / (prior + self.r * self.r / len(res))
            s.x = s.x + gain * (z - s.x); s.p = (1.0 - gain) * prior; s.lost = 0
        else:
            s.p = prior
            enough = min(len(remote_boxes), len(ego_boxes)) >= self.params.min_boxes
            s.lost = s.lost + 1 if enough else 0
            if s.lost >= self.params.lost_messages:
                value = acquire_coarse_offset(list(s.history), self.params)
                if value is not None:
                    s.x_acq = value; s.x = value.copy(); s.p = self.params.reset_variance
                    s.lost = 0; s.triggers += 1; triggered = True
        return dict(delta=s.x.copy(), variance=float(s.p), anchors=int(len(res)), triggered=triggered,
                    coarse_offset=s.x_acq.copy())


def estimate_offsets_with_acquisition(messages: Iterable[Tuple[Hashable, np.ndarray, np.ndarray]],
                                      q: float = 0.05, r: float = 0.6, gate: float = 2.5,
                                      params: AcquisitionParams = AcquisitionParams()) -> dict:
    """Batch helper: ``messages`` yields ``(stream, remote_boxes, ego_boxes)`` in arrival order."""
    est = CoMETAEstimator(q=q, r=r, gate=gate, params=params)
    rows = [est.update(s, a, b) for s, a, b in messages]
    out = {k: np.asarray([row[k] for row in rows]) for k in ("delta", "variance", "anchors", "triggered")}
    out["triggers"] = int(out["triggered"].sum())
    return out
