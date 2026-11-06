"""Unit tests for comet/estimator.py on hand-made inputs (no dataset needed)."""
import numpy as np

from comet.estimator import (
    CoMETEstimator,
    CommonModeKF,
    apply_offset,
    common_mode_observation,
    mutual_hungarian,
)


def test_mutual_hungarian_gate_and_mutual_nn():
    a = np.array([[0.0, 0.0], [10.0, 0.0]])
    b = np.array([[0.5, 0.0], [30.0, 0.0]])
    i, j = mutual_hungarian(a, b, gate=2.5)
    assert i.tolist() == [0] and j.tolist() == [0]


def test_mutual_hungarian_empty():
    i, j = mutual_hungarian(np.zeros((0, 2)), np.ones((3, 2)))
    assert len(i) == 0 and len(j) == 0


def test_observation_is_mean_residual():
    remote = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, -3.0]])
    shift = np.array([-1.0, 0.4])
    n, z = common_mode_observation(remote, remote + shift)
    assert n == 3 and np.allclose(z, shift)


def test_kf_matches_closed_form():
    kf = CommonModeKF(q=0.05, r=0.6)
    prior = 1.0 + 0.05 ** 2
    gain = prior / (prior + 0.36 / 4)
    mean, var = kf.step(4, np.array([1.0, -2.0]))
    assert np.allclose(mean, gain * np.array([1.0, -2.0]))
    assert np.isclose(var, (1 - gain) * prior)
    mean2, var2 = kf.step(0, np.zeros(2))
    assert np.allclose(mean2, mean) and np.isclose(var2, var + 0.05 ** 2)


def test_streams_are_independent_and_converge():
    est = CoMETEstimator(q=0.05)
    rng = np.random.default_rng(0)
    truth = np.array([-1.1, 0.3])
    for _ in range(50):
        remote = rng.uniform(-40, 40, size=(8, 2))
        ego = remote + truth + rng.normal(0, 0.2, size=remote.shape)
        out = est.update(("seq", 0), remote, ego)
    assert np.linalg.norm(out["delta"] - truth) < 0.1
    other = est.update(("seq", 400), np.zeros((0, 2)), np.zeros((0, 2)))
    assert np.allclose(other["delta"], 0) and other["anchors"] == 0


def test_apply_offset_only_moves_xy():
    boxes = np.array([[1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.1]])
    out = apply_offset(boxes, np.array([0.5, -1.0]))
    assert np.allclose(out[0, :2], [1.5, 1.0]) and np.allclose(out[0, 2:], boxes[0, 2:])
