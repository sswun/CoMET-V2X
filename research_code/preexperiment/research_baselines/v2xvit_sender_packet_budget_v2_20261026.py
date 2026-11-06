#!/usr/bin/env python3
"""Frozen byte-budget wrapper around the published V2XBPKT1 wire codec.

The wire payload is *exactly* ``v2xseq_real_motion_packets.encode_packet``:
a 16-byte prefix, canonical ASCII JSON carrying sequence/frame/time/full 4x4
pose, followed by contiguous boxes, scores, and labels.  This module only
freezes sender-local ordering and chooses how many complete 36-byte logical
detection records fit.

If even the canonical zero-detection V2XBPKT1 packet exceeds a finite budget,
the legal result is the existing recorded-arrival ``no packet delivered``
state, represented at transport level by zero bytes.  An oversized codec packet
is never emitted.  Both current and previous sender frames must independently
pass through this wrapper before constant-velocity history is constructed.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from v2xseq_real_motion_packets import (
    PREFIX, decode_packet as decode_v1_packet, encode_packet as encode_v1_packet,
)


LOGICAL_RECORD_BYTES = 36  # 7*f32 box + f32 score + i32 label
AUTHORIZED_BUDGETS = (256, 512, 1024, None)


def packet_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sender_order(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Freeze sender-only ranking before any receiver/GT computation.

    Primary order is score descending.  Exact ties use label, all seven box
    fields, then the detector's original row index, ascending.  Original index
    is a deterministic tie-breaker only; it is not an extra wire field.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if original_indices is None:
        original_indices = np.arange(len(boxes), dtype=np.uint32)
    original_indices = np.asarray(original_indices).reshape(-1)
    if not (len(boxes) == len(scores) == len(labels) == len(original_indices)):
        raise ValueError("box, score, label, and original-index counts differ")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("box or score is non-finite")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must lie in [0,1]")
    if np.any(labels < np.iinfo(np.int32).min) or np.any(labels > np.iinfo(np.int32).max):
        raise ValueError("labels do not fit the V2XBPKT1 int32 field")
    rows = range(len(boxes))
    return np.asarray(sorted(
        rows,
        key=lambda index: (
            -float(scores[index]), int(labels[index]),
            *(float(value) for value in boxes[index]),
            int(original_indices[index]),
        ),
    ), dtype=np.int64)


def _encode_prefix(
    *, sequence: str, source_frame: str, source_timestamp_us: int,
    source_pose: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
    labels: np.ndarray, order: np.ndarray, count: int,
) -> bytes:
    selected = order[:count]
    return encode_v1_packet(
        sequence=sequence, source_frame=source_frame,
        source_timestamp_us=int(source_timestamp_us), source_pose=source_pose,
        boxes=boxes[selected], scores=scores[selected], labels=labels[selected],
    )


def encode_budgeted_packet(
    *,
    sequence: str,
    source_frame: str,
    source_timestamp_us: int,
    source_pose: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    budget_bytes: int | None,
    original_indices: np.ndarray | None = None,
) -> bytes:
    """Return the largest score-prefix V2XBPKT1 payload that fits the budget.

    ``b''`` is the legal no-delivery state when the required V2XBPKT1 metadata
    alone cannot fit.  It is not passed to the V2XBPKT1 decoder.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    order = sender_order(boxes, scores, labels, original_indices)
    if budget_bytes is None:
        return _encode_prefix(
            sequence=sequence, source_frame=source_frame,
            source_timestamp_us=source_timestamp_us, source_pose=source_pose,
            boxes=boxes, scores=scores, labels=labels, order=order, count=len(order),
        )
    budget_bytes = int(budget_bytes)
    if budget_bytes < 0:
        raise ValueError("budget must be non-negative or None")
    empty = _encode_prefix(
        sequence=sequence, source_frame=source_frame,
        source_timestamp_us=source_timestamp_us, source_pose=source_pose,
        boxes=boxes, scores=scores, labels=labels, order=order, count=0,
    )
    if len(empty) > budget_bytes:
        return b""
    capacity = min(len(order), (budget_bytes - len(empty)) // LOGICAL_RECORD_BYTES)
    payload = _encode_prefix(
        sequence=sequence, source_frame=source_frame,
        source_timestamp_us=source_timestamp_us, source_pose=source_pose,
        boxes=boxes, scores=scores, labels=labels, order=order, count=capacity,
    )
    if len(payload) > budget_bytes:
        raise AssertionError("complete-record capacity calculation exceeded budget")
    return payload


def decode_budgeted_packet(payload: bytes) -> dict[str, Any] | None:
    """Decode V2XBPKT1, or return ``None`` for the legal 0-byte no-packet state."""
    if not payload:
        return None
    decoded = decode_v1_packet(payload)
    metadata = decoded["metadata"]
    return {
        "schema": metadata["schema"],
        "sequence": metadata["sequence"],
        "source_frame": metadata["source_frame"],
        "source_timestamp_us": int(metadata["source_timestamp_us"]),
        "source_pose_lidar_to_world": np.asarray(
            metadata["source_pose_lidar_to_world"], dtype=np.float64),
        "packet_bytes": len(payload),
        "boxes": decoded["boxes"],
        "scores": decoded["scores"],
        "labels": decoded["labels"],
    }


def decode_current_and_history(
    current_payload: bytes,
    previous_payload: bytes | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Decode already same-budget current/history payloads for CV construction."""
    current = decode_budgeted_packet(current_payload)
    previous = None if previous_payload is None else decode_budgeted_packet(previous_payload)
    if current is None:
        # No current packet means there is no stale or CV remote arm, regardless
        # of whether an older packet existed at its own sender time.
        return None, previous
    if previous is not None:
        if current["sequence"] != previous["sequence"]:
            raise ValueError("history packet is from a different sequence")
        if previous["source_timestamp_us"] >= current["source_timestamp_us"]:
            raise ValueError("history packet is not strictly earlier")
    return current, previous


def empty_packet_bytes(
    *, sequence: str, source_frame: str, source_timestamp_us: int,
    source_pose: np.ndarray,
) -> int:
    payload = encode_v1_packet(
        sequence=sequence, source_frame=source_frame,
        source_timestamp_us=int(source_timestamp_us), source_pose=source_pose,
        boxes=np.empty((0, 7), np.float32), scores=np.empty(0, np.float32),
        labels=np.empty(0, np.int32),
    )
    return len(payload)


def layout_identity() -> dict[str, Any]:
    return {
        "schema": "v2xvit_sender_packet_budget_layout_v2",
        "wire_codec": "V2XBPKT1 unchanged",
        "wire_codec_source": "preexperiment/research_baselines/v2xseq_real_motion_packets.py",
        "endianness": "little",
        "prefix_bytes": PREFIX.size,
        "prefix_fields": ["magic:8s", "metadata_bytes:u32", "record_count:u32"],
        "metadata_encoding": "canonical sort_keys compact ASCII JSON, variable length",
        "required_metadata": [
            "schema", "sequence", "source_frame", "source_timestamp_us",
            "source_pose_lidar_to_world[4][4]", "box_order", "array_encoding",
        ],
        "array_layout": [
            "boxes:f32[count,7] contiguous", "scores:f32[count] contiguous",
            "labels:i32[count] contiguous",
        ],
        "logical_record_bytes": LOGICAL_RECORD_BYTES,
        "alignment": "packed little-endian prefix; packed 4-byte array scalars; no budget padding",
        "authorized_budgets_bytes": [256, 512, 1024, None],
        "finite_packets_are_padded_to_budget": False,
        "selection": (
            "score descending; ties label,x,y,z,h,w,l,yaw,original detector row "
            "ascending; sender-local detector outputs only"
        ),
        "infeasible_metadata_rule": (
            "if the canonical zero-detection V2XBPKT1 payload exceeds the budget, "
            "emit 0 bytes and use the existing no-packet-delivered receiver state"
        ),
    }


assert PREFIX.size == 16  # 8-byte magic plus two uint32 fields.

