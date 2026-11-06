"""CoMET: causal common-mode error tracking for asynchronous V2X late fusion.

A remote agent's boxes, after projection into the ego frame with the public
calibration and online poses, carry a common translation error shared by all
objects of one message. CoMET treats every ego detection as a potential anchor,
associates it with the remote boxes, averages the residuals into one
observation per message, and tracks the offset with a scalar Kalman filter per
sender stream. No training, ground truth, future frames, or per-pair official
offsets are used.

This module is a clean re-implementation of the recipe used in the paper
(the "H1" configuration in ``research_code``):

* anchors: all ego boxes (``source == 0``), no score threshold;
* association: Hungarian on BEV centre distance, kept only if the pair is
  within ``gate`` metres and the two boxes are mutual nearest neighbours;
* observation: mean residual ``ego_xy - remote_xy`` over the matched pairs;
* filter: random-walk state, isotropic variance shared by x and y,
  measurement variance ``r**2 / n`` for ``n`` matched anchors, no innovation
  gating.

``tools/verify_against_recipe.py`` checks this module against the states that
produced the paper numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "mutual_hungarian",
    "common_mode_observation",
    "CommonModeKF",
    "CoMETEstimator",
    "apply_offset",
    "estimate_offsets",
]


def mutual_hungarian(a_xy: np.ndarray, b_xy: np.ndarray, gate: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    """Associate two box sets by BEV centre distance.

    Returns index arrays ``(i, j)`` such that ``a_xy[i]`` matches ``b_xy[j]``.
    A Hungarian pair is kept only if its distance is at most ``gate`` and the
    two boxes are each other's nearest neighbour.
    """
    a = np.asarray(a_xy, dtype=float)
    b = np.asarray(b_xy, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.empty(0, int), np.empty(0, int)
    d = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=-1)
    i, j = linear_sum_assignment(d)
    good = (d[i, j] <= gate) & (d.argmin(1)[i] == j) & (d.argmin(0)[j] == i)
    return i[good], j[good]


def common_mode_observation(remote_xy: np.ndarray, ego_xy: np.ndarray, gate: float = 2.5) -> Tuple[int, np.ndarray]:
    """One common-mode observation from a single message.

    ``remote_xy``: remote boxes already projected into the ego frame (and, in
    the paper, propagated to the decision time with a constant-velocity model).
    ``ego_xy``: the ego agent's own detections at the decision time.
    Returns ``(n, z)``: the number of matched anchors and the mean residual
    ``ego - remote`` (zeros when nothing matched).
    """
    i, j = mutual_hungarian(remote_xy, ego_xy, gate)
    if len(i) == 0:
        return 0, np.zeros(2)
    residual = np.asarray(ego_xy, float)[j, :2] - np.asarray(remote_xy, float)[i, :2]
    return int(len(i)), residual.mean(0)


@dataclass
class CommonModeKF:
    """Random-walk Kalman filter on a 2-D translation with isotropic variance.

    ``q``: process noise standard deviation per message (metres).
    ``r``: single-anchor measurement noise standard deviation (metres);
    an observation averaged over ``n`` anchors has variance ``r**2 / n``.
    """

    q: float = 0.05
    r: float = 0.6
    p0: float = 1.0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(2))
    var: float = 1.0
    steps: int = 0

    def __post_init__(self) -> None:
        self.var = float(self.p0)

    def step(self, n: int, z: np.ndarray) -> Tuple[np.ndarray, float]:
        """Advance one message; update with ``z`` when ``n > 0``."""
        prior = self.var + self.q * self.q
        if n:
            gain = prior / (prior + self.r * self.r / n)
            self.mean = self.mean + gain * (np.asarray(z, float) - self.mean)
            self.var = (1.0 - gain) * prior
        else:
            self.var = prior
        self.steps += 1
        return self.mean.copy(), float(self.var)


class CoMETEstimator:
    """Streaming estimator keeping one filter per sender stream.

    In deployment a stream is one sender-receiver pair. In the paper's offline
    replay each arrival-delay setting is replayed independently, so the stream
    key is ``(sequence, wait_ms)``.
    """

    def __init__(self, q: float = 0.05, r: float = 0.6, gate: float = 2.5, p0: float = 1.0):
        self.q, self.r, self.gate, self.p0 = q, r, gate, p0
        self.filters: dict[Hashable, CommonModeKF] = {}

    def update(self, stream: Hashable, remote_xy: np.ndarray, ego_xy: np.ndarray) -> dict:
        kf = self.filters.setdefault(stream, CommonModeKF(self.q, self.r, self.p0))
        n, z = common_mode_observation(remote_xy, ego_xy, self.gate)
        delta, var = kf.step(n, z)
        return dict(delta=delta, variance=var, anchors=n, observation=z)


def apply_offset(remote_boxes: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Translate remote boxes (x, y, ...) in the ego frame by the estimated offset."""
    out = np.array(remote_boxes, dtype=float, copy=True)
    out[:, :2] += np.asarray(delta, float)
    return out


def estimate_offsets(
    messages: Iterable[Tuple[Hashable, np.ndarray, np.ndarray]],
    q: float = 0.05,
    r: float = 0.6,
    gate: float = 2.5,
) -> dict:
    """Batch helper: ``messages`` yields ``(stream, remote_xy, ego_xy)`` in arrival order.

    Returns arrays ``delta`` (N, 2), ``variance`` (N,), ``anchors`` (N,) and
    ``observation`` (N, 2), one row per message.
    """
    est = CoMETEstimator(q=q, r=r, gate=gate)
    rows = [est.update(s, a, b) for s, a, b in messages]
    return {k: np.asarray([row[k] for row in rows]) for k in ("delta", "variance", "anchors", "observation")}
