#!/usr/bin/env python3
"""Attach exact vehicle LiDAR timestamps to frozen V2X-ViT packed rows.

The packed history cache intentionally stored ``None`` for conditions without
a roadside packet.  Causal filters still need the receiver clock on those
rows.  The vehicle-side dataset manifest provides that clock independently of
packet availability, and its values exactly equal every timestamp already
stored on rows that do have a packet.
"""

from __future__ import annotations

import json
from pathlib import Path

from e1_receiver_value_core_v1_20261020 import sha256
from object_history_selector_core_v1_20261021 import PackedDataset


DEFAULT_VEHICLE_DATA_INFO = Path(
    "artifacts/data/stage1/v2x_seq_spd/trainval_metadata/"
    "V2X-Seq-SPD/vehicle-side/data_info.json"
)


class TimestampedPackedDataset(PackedDataset):
    """PackedDataset view with receiver timestamps filled in memory."""

    def __init__(self, directory: Path, vehicle_data_info: Path = DEFAULT_VEHICLE_DATA_INFO):
        super().__init__(Path(directory))
        vehicle_data_info = Path(vehicle_data_info)
        rows = json.loads(vehicle_data_info.read_text())
        mapping = {
            str(row["frame_id"]): (
                int(row["pointcloud_timestamp"]), str(row["sequence_id"])
            )
            for row in rows
        }
        assert len(mapping) == len(rows)
        injected = verified_existing = 0
        for meta in self.meta:
            frame = str(meta["current_vehicle_frame"])
            timestamp, sequence = mapping[frame]
            assert sequence == str(meta["sequence"])
            if meta.get("receiver_timestamp_us") is None:
                meta["receiver_timestamp_us"] = timestamp
                injected += 1
            else:
                assert int(meta["receiver_timestamp_us"]) == timestamp
                verified_existing += 1
        self.receiver_timestamp_identity = {
            "vehicle_data_info": str(vehicle_data_info),
            "vehicle_data_info_sha256": sha256(vehicle_data_info),
            "rule": "exact vehicle-side pointcloud_timestamp indexed by current_vehicle_frame",
            "rows": len(self.meta),
            "injected_rows": injected,
            "verified_existing_rows": verified_existing,
            "persistent_cache_modified": False,
        }

