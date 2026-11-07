"""Unit tests for comet/acquisition.py, comet/fusion.py and comet/stress.py (no dataset needed)."""
import numpy as np

from comet.acquisition import CoMETAEstimator, acquire_coarse_offset, estimate_offsets_with_acquisition
from comet.estimator import estimate_offsets
from comet.fusion import bev_iou, combine_two_tiers, fit_unmatched_coefficient
from comet.stress import injection_shifts, sequence_direction


def _scene(rng, n=10, spread=40.0):
    return rng.uniform(-spread, spread, size=(n, 2))


def test_identical_to_comet_when_offset_within_gate():
    rng = np.random.default_rng(1)
    truth = np.array([0.8, -0.5])
    msgs = []
    for _ in range(40):
        remote = _scene(rng)
        msgs.append((("s", 0), remote, remote + truth + rng.normal(0, 0.1, remote.shape)))
    a = estimate_offsets_with_acquisition(msgs)
    b = estimate_offsets(msgs)
    assert a["triggers"] == 0
    assert np.array_equal(a["delta"], b["delta"]) and np.array_equal(a["anchors"], b["anchors"])


def test_reacquires_offset_beyond_gate():
    rng = np.random.default_rng(2)
    truth = np.array([4.0, -3.0])  # 5 m, beyond the 2.5 m gate
    est = CoMETAEstimator(q=0.05)
    out = None
    for _ in range(30):
        remote = _scene(rng, n=12, spread=60.0)
        out = est.update(("s", 0), remote, remote + truth + rng.normal(0, 0.1, remote.shape))
    assert est.streams[("s", 0)].triggers >= 1
    assert np.linalg.norm(out["delta"] - truth) < 0.15 and out["anchors"] > 0


def test_plain_comet_never_locks_beyond_gate():
    rng = np.random.default_rng(2)
    truth = np.array([4.0, -3.0])
    msgs = []
    for _ in range(30):
        remote = _scene(rng, n=12, spread=60.0)
        msgs.append((("s", 0), remote, remote + truth + rng.normal(0, 0.1, remote.shape)))
    b = estimate_offsets(msgs)
    assert np.linalg.norm(b["delta"][-1] - truth) > 2.0


def test_acquisition_rejects_small_or_ambiguous_peaks():
    rng = np.random.default_rng(3)
    remote = _scene(rng, n=12, spread=60.0)
    small = [(remote, remote + np.array([1.0, 0.0]))] * 5
    assert acquire_coarse_offset(small) is None  # |m| < 2 m: the gate already reaches it
    assert acquire_coarse_offset([(remote[:2], remote[:2] + 4.0)]) is None  # too little support


def test_bev_iou_axis_aligned_and_rotated():
    a = np.array([[0, 0, 0, 1.5, 2.0, 4.0, 0.0]])
    b = np.array([[2.0, 0, 0, 1.5, 2.0, 4.0, 0.0], [0, 0, 0, 1.5, 2.0, 4.0, np.pi / 2], [30, 30, 0, 1.5, 2, 4, 0]])
    iou = bev_iou(a, b)[0]
    assert np.isclose(iou[0], 4.0 / 12.0, atol=1e-6)        # half overlap of two 2x4 boxes
    assert np.isclose(iou[1], 4.0 / 12.0, atol=1e-6)        # cross: 2x2 overlap
    assert iou[2] == 0.0 and iou.dtype == np.float32


def test_two_tier_combination_rules():
    t1 = np.array([[0, 0, 0, 1.5, 2, 4, 0], [20, 0, 0, 1.5, 2, 4, 0]], float)
    t2 = np.array([[0.2, 0, 0, 1.5, 2, 4, 0], [-20, 5, 0, 1.5, 2, 4, 0]], float)
    boxes, scores, origin = combine_two_tiers(t1, np.array([0.6, 0.9]), t2, np.array([0.8, 0.5]), 0.5, 0.4)
    assert origin.tolist() == [[0, 0], [0, 1], [1, 1]]  # scores 0.8, 0.45, 0.2
    paired = np.flatnonzero((origin[:, 0] == 0) & (origin[:, 1] == 0))[0]
    assert np.allclose(boxes[paired], t1[0]) and np.isclose(scores[paired], 0.8)   # tier-1 box, max score
    lone1 = np.flatnonzero((origin[:, 0] == 0) & (origin[:, 1] == 1))[0]
    lone2 = np.flatnonzero((origin[:, 0] == 1) & (origin[:, 1] == 1))[0]
    assert np.isclose(scores[lone1], 0.9 * 0.5) and np.isclose(scores[lone2], 0.5 * 0.4)
    assert np.all(np.diff(scores) <= 0)


def test_nms_removes_overlapping_unpaired_boxes():
    t1 = np.array([[0, 0, 0, 1.5, 2, 4, 0]], float)
    t2 = np.array([[0.0, 1.7, 0, 1.5, 2, 4, 0]], float)  # IoU ~0.08 < 0.3: not paired, and < 0.15: kept
    _, _, origin = combine_two_tiers(t1, np.array([0.9]), t2, np.array([0.9]), 1.0, 1.0)
    assert len(origin) == 2
    t2 = np.array([[0.0, 1.2, 0, 1.5, 2, 4, 0]], float)  # IoU ~0.25: not paired, suppressed by NMS
    _, _, origin = combine_two_tiers(t1, np.array([0.9]), t2, np.array([0.8]), 1.0, 1.0)
    assert origin.tolist() == [[0, 0]]


def test_unmatched_coefficient_is_clipped_least_squares():
    s = np.array([0.9, 0.8, 0.6, 0.5]); y = np.array([1, 0, 0, 0])
    assert np.isclose(fit_unmatched_coefficient(s, y), 0.9 / np.dot(s, s))
    assert fit_unmatched_coefficient(np.array([0.1]), np.array([1])) == 1.0


def test_injection_profiles():
    seqs = ["0001"] * 5 + ["0002"] * 3
    ts = [0, 1, 2, 3, 4, 10, 20, 30]
    t1 = injection_shifts(seqs, ts, "T1", 3.0)
    assert np.allclose(np.linalg.norm(t1, axis=1), 3.0)
    assert np.allclose(t1[0], 3.0 * sequence_direction("0001")) and np.allclose(t1[5], t1[7])
    t2 = injection_shifts(seqs, ts, "T2")
    assert np.allclose(np.linalg.norm(t2[:5], axis=1), [0, 0.5, 1.0, 1.5, 2.0])
    t3 = injection_shifts(seqs, ts, "T3")
    assert np.allclose(np.linalg.norm(t3[:5], axis=1), [0, 0, 2, 2, 2])
