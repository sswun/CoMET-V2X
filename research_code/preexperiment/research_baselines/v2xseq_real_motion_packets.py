#!/usr/bin/env python3
"""Packet codec and legal-history motion controls for V2X-Seq box messages."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


MAGIC = b"V2XBPKT1"
PREFIX = struct.Struct("<8sII")


def encode_packet(
    *,
    sequence: str,
    source_frame: str,
    source_timestamp_us: int,
    source_pose: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> bytes:
    """Serialize one delivered application payload and return its actual bytes."""
    boxes = np.asarray(boxes, dtype="<f4").reshape(-1, 7)
    scores = np.asarray(scores, dtype="<f4").reshape(-1)
    labels = np.asarray(labels, dtype="<i4").reshape(-1)
    pose = np.asarray(source_pose, dtype=np.float64).reshape(4, 4)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("box, score, and label counts differ")
    if not (
        np.isfinite(boxes).all()
        and np.isfinite(scores).all()
        and np.isfinite(pose).all()
        and np.all((scores >= 0.0) & (scores <= 1.0))
    ):
        raise ValueError("packet contains non-finite values or invalid scores")
    metadata = json.dumps(
        {
            "schema": "v2xseq_box_packet_v1",
            "sequence": str(sequence),
            "source_frame": str(source_frame),
            "source_timestamp_us": int(source_timestamp_us),
            "source_pose_lidar_to_world": pose.tolist(),
            "box_order": "x_y_z_h_w_l_yaw",
            "array_encoding": "little_endian_float32_boxes_scores_int32_labels",
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return b"".join(
        (
            PREFIX.pack(MAGIC, len(metadata), len(boxes)),
            metadata,
            boxes.tobytes(order="C"),
            scores.tobytes(order="C"),
            labels.tobytes(order="C"),
        )
    )


def decode_packet(payload: bytes) -> dict[str, Any]:
    if len(payload) < PREFIX.size:
        raise ValueError("truncated packet prefix")
    magic, metadata_bytes, count = PREFIX.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("packet magic mismatch")
    expected = PREFIX.size + metadata_bytes + int(count) * 36
    if len(payload) != expected:
        raise ValueError(f"packet length mismatch: {len(payload)} != {expected}")
    start = PREFIX.size
    metadata = json.loads(payload[start : start + metadata_bytes].decode("ascii"))
    if metadata.get("schema") != "v2xseq_box_packet_v1":
        raise ValueError("packet schema mismatch")
    start += metadata_bytes
    boxes = np.frombuffer(payload, dtype="<f4", count=int(count) * 7, offset=start).copy().reshape(-1, 7)
    start += int(count) * 28
    scores = np.frombuffer(payload, dtype="<f4", count=int(count), offset=start).copy()
    start += int(count) * 4
    labels = np.frombuffer(payload, dtype="<i4", count=int(count), offset=start).copy()
    return {"metadata": metadata, "boxes": boxes, "scores": scores, "labels": labels}


def packet_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def transform_boxes7d(boxes: np.ndarray, source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Rigidly transform lidar-frame boxes; dimensions remain h,w,l."""
    result = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 7)
    if not len(result):
        return result
    source = np.asarray(source_pose, dtype=np.float64).reshape(4, 4)
    target = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
    transform = np.linalg.inv(target) @ source
    homogeneous = np.column_stack((result[:, :3].astype(np.float64), np.ones(len(result))))
    result[:, :3] = (transform @ homogeneous.T).T[:, :3]
    direction = np.column_stack(
        (np.cos(result[:, 6]), np.sin(result[:, 6]), np.zeros(len(result)))
    )
    rotated = (transform[:3, :3] @ direction.T).T
    result[:, 6] = np.arctan2(rotated[:, 1], rotated[:, 0])
    return result


def associate_motion(
    current_boxes: np.ndarray,
    current_pose: np.ndarray,
    current_timestamp_us: int,
    previous: dict[str, Any] | None,
    horizon_s: float,
    *,
    max_history_s: float = 0.5,
    max_speed_mps: float = 50.0,
    association_margin_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use only the current and latest previously delivered sender packets."""
    boxes = np.asarray(current_boxes, dtype=np.float32).copy().reshape(-1, 7)
    confirmed = np.zeros(len(boxes), dtype=bool)
    diagnostic: dict[str, Any] = {
        "remote_boxes": int(len(boxes)),
        "matched_boxes": 0,
        "history_dt_s": None,
        "horizon_s": float(horizon_s),
        "mean_shift_m": 0.0,
        "max_shift_m": 0.0,
    }
    if horizon_s <= 0.0:
        confirmed[:] = True
        return boxes, confirmed, diagnostic
    if previous is None or not len(boxes):
        return boxes, confirmed, diagnostic
    previous_boxes = np.asarray(previous["boxes"], dtype=np.float32).reshape(-1, 7)
    if not len(previous_boxes):
        return boxes, confirmed, diagnostic
    history_dt = (int(current_timestamp_us) - int(previous["source_timestamp_us"])) / 1_000_000.0
    diagnostic["history_dt_s"] = float(history_dt)
    if history_dt <= 0.0 or history_dt > float(max_history_s):
        return boxes, confirmed, diagnostic
    aligned = transform_boxes7d(previous_boxes, previous["source_pose"], current_pose)
    distance = np.linalg.norm(boxes[:, None, :2] - aligned[None, :, :2], axis=2)
    rows, columns = linear_sum_assignment(distance)
    gate = float(association_margin_m) + float(max_speed_mps) * history_dt
    shifts: list[float] = []
    for row, column in zip(rows, columns):
        if float(distance[row, column]) > gate:
            continue
        velocity = (boxes[row, :2] - aligned[column, :2]) / history_dt
        speed = float(np.linalg.norm(velocity))
        if speed > float(max_speed_mps):
            velocity *= float(max_speed_mps) / speed
        shift = velocity * float(horizon_s)
        boxes[row, :2] += shift.astype(np.float32)
        confirmed[int(row)] = True
        shifts.append(float(np.linalg.norm(shift)))
    diagnostic["matched_boxes"] = int(confirmed.sum())
    if shifts:
        diagnostic["mean_shift_m"] = float(np.mean(shifts))
        diagnostic["max_shift_m"] = float(np.max(shifts))
    return boxes, confirmed, diagnostic


def stale_and_motion_arms(
    current_packet: dict[str, Any],
    previous_packet: dict[str, Any] | None,
    receiver_pose: np.ndarray,
    receiver_timestamp_us: int,
) -> dict[str, Any]:
    """Construct stale, CV, and two-hit predictions before local-box merging."""
    metadata = current_packet["metadata"]
    source_pose = np.asarray(metadata["source_pose_lidar_to_world"], dtype=np.float64)
    source_timestamp = int(metadata["source_timestamp_us"])
    horizon = (int(receiver_timestamp_us) - source_timestamp) / 1_000_000.0
    if horizon < -1e-9:
        raise ValueError("source packet is from the receiver's future")
    boxes = np.asarray(current_packet["boxes"], dtype=np.float32)
    motion, confirmed, diagnostic = associate_motion(
        boxes, source_pose, source_timestamp, previous_packet, max(0.0, horizon)
    )
    return {
        "horizon_s": max(0.0, horizon),
        "stale_boxes": transform_boxes7d(boxes, source_pose, receiver_pose),
        "constant_velocity_boxes": transform_boxes7d(motion, source_pose, receiver_pose),
        "two_hit_constant_velocity_boxes": transform_boxes7d(motion[confirmed], source_pose, receiver_pose),
        "two_hit_mask": confirmed,
        "scores": np.asarray(current_packet["scores"], dtype=np.float32),
        "labels": np.asarray(current_packet["labels"], dtype=np.int32),
        "diagnostic": diagnostic,
    }

