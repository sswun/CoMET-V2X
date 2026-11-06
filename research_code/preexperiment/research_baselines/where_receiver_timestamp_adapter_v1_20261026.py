#!/usr/bin/env python3
"""In-memory receiver-clock adapter for the reviewed Where prepared cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from object_history_selector_core_v1_20261021 import PackedDataset


class TimestampedPackedDataset(PackedDataset):
    """Attach a qualified complete clock without changing the packed cache."""

    def __init__(self, directory: Path, timestamp_archive: Path, split: str):
        super().__init__(Path(directory))
        assert split in ("train", "val")
        key = split + "_receiver_timestamp_us"
        with np.load(timestamp_archive, allow_pickle=False) as archive:
            timestamps = archive[key].astype(np.int64)
        assert len(timestamps) == len(self.meta)
        for index, (meta, timestamp) in enumerate(zip(self.meta, timestamps)):
            assert int(meta["condition_index"]) == index
            existing = meta.get("receiver_timestamp_us")
            if existing is not None:
                assert int(existing) == int(timestamp)
            self.meta[index] = dict(meta, receiver_timestamp_us=int(timestamp))

