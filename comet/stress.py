"""Controlled pose-error injection used by the robustness experiment (R8c/R8d).

A known translation error is added to every remote box of a message (in the
ego frame), the same for all boxes of the message; ground truth is unchanged.
The direction is drawn once per sequence from a PCG64 stream seeded with
``seed`` and the SHA-256 of the sequence name, so it does not depend on the
detector, split, wait setting or processing order. Profiles:

* ``"T1"``: constant ``amplitude`` metres;
* ``"T2"``: linear drift from 0 at the first message of the sequence to 2 m at the last;
* ``"T3"``: a 2 m step at the temporal midpoint of the sequence.

The injected values are an experimental perturbation: no estimator reads them.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

__all__ = ["sequence_direction", "injection_shifts", "inject_remote_boxes"]

SEED = 2026111401


def sequence_direction(sequence: str, seed: int = SEED) -> np.ndarray:
    words = np.frombuffer(hashlib.sha256(str(sequence).encode()).digest()[:16], dtype="<u4")
    rng = np.random.default_rng(np.random.SeedSequence([seed, *map(int, words)]))
    t = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(t), np.sin(t)])


def injection_shifts(sequences: Sequence[str], timestamps_us: Sequence[int], kind: str,
                     amplitude: float = 0.0, seed: int = SEED) -> np.ndarray:
    """(N, 2) translation injected into each message (one row per message)."""
    seqs = np.asarray([str(s) for s in sequences]); ts = np.asarray(timestamps_us)
    directions = {s: sequence_direction(s, seed) for s in sorted(set(seqs.tolist()))}
    span = {s: (float(ts[seqs == s].min()), float(ts[seqs == s].max())) for s in directions}
    out = np.zeros((len(seqs), 2))
    for k, (s, t) in enumerate(zip(seqs, ts)):
        lo, hi = span[s]
        frac = (float(t) - lo) / (hi - lo) if hi > lo else 0.0
        if kind == "T1":
            a = amplitude
        elif kind == "T2":
            a = 2.0 * frac
        elif kind == "T3":
            a = 2.0 * (float(t) >= (lo + hi) / 2)
        else:
            raise ValueError(kind)
        out[k] = directions[s] * a
    return out


def inject_remote_boxes(boxes: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Translate the xy of one message's remote boxes by ``shift`` (dtype preserved)."""
    out = np.array(boxes, copy=True)
    out[:, :2] += np.asarray(shift, dtype=out.dtype)
    return out
