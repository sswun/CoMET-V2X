"""Single-message recorded-arrival adapter for the existing shared loader.

Sensor extrinsics only: pair-dependent system_error_offset is excluded because
its online provenance is unestablished. This is a separate calibrated condition,
not a relabeling of the accepted nominal benchmark. No historical source labels
or paired future ego pose are used to construct the remote input.
"""
from __future__ import annotations

import bisect
from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path

PROTOCOL_ID = "v2xseq_recorded_arrival_sensor_extrinsics_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Catalogue:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.paths = {side: self.root / side / "data_info.json"
                      for side in ("vehicle-side", "infrastructure-side")}
        self.info = {side: {x["frame_id"]: x for x in json.loads(p.read_text())}
                     for side, p in self.paths.items()}
        self.available = defaultdict(list)
        for x in self.info["infrastructure-side"].values():
            if (self.root / "infrastructure-side" / x["pointcloud_path"]).is_file():
                self.available[x["sequence_id"]].append(
                    (int(x["pointcloud_timestamp"]), x["frame_id"]))
        for values in self.available.values():
            values.sort()

    def select(self, sequence, vehicle_frame, delay_ms):
        vehicle = self.info["vehicle-side"][vehicle_frame]
        if vehicle["sequence_id"] != sequence:
            raise RuntimeError("Vehicle sequence identity changed")
        decision = int(vehicle["pointcloud_timestamp"])
        choices = self.available[sequence]
        pos = bisect.bisect_right(choices, (decision - delay_ms * 1000, "\uffff")) - 1
        selected = choices[pos] if pos >= 0 else None
        return {"protocol_id": PROTOCOL_ID, "sequence": sequence,
                "effective_label_frame": vehicle_frame,
                "effective_vehicle_frame": vehicle_frame,
                "effective_infrastructure_frame": selected[1] if selected else None,
                "vehicle_pointcloud_timestamp_us": decision,
                "infrastructure_pointcloud_timestamp_us": selected[0] if selected else None,
                "configured_wait_ms": delay_ms,
                "actual_age_ms": (decision - selected[0]) / 1000 if selected else None,
                "no_packet_fallback": selected is None,
                "calibration_policy": "published_sensor_extrinsics_only_no_pair_offset"}


def pose_from_json(path, np, nested=False):
    value = json.loads(Path(path).read_text())
    if nested:
        value = value["transform"]
    pose = np.eye(4)
    pose[:3, :3] = value["rotation"]
    pose[:3, 3] = np.asarray(value["translation"]).reshape(3)
    if not np.isfinite(pose).all():
        raise RuntimeError("Nonfinite sensor extrinsics")
    return pose


def install(dataset, np, corrected_index_installer, delay_ms, root):
    p = dataset.params
    if not (p["fusion"]["dataset"] == "v2xseq" and dataset.ego_agent == "vehicle"
            and p["dataset"]["frame_his"] == p["dataset"]["cav_frame_his"] == 1
            and dataset.proj_first and not dataset.train):
        raise RuntimeError("Adapter requires eval loader, current ego and one remote message")
    catalogue = Catalogue(root)
    corrected_index_installer(dataset, np)
    ego_retrieval = dataset.retrieve_base_data
    dataset.arrival_catalogue = catalogue
    dataset.arrival_delay_ms = delay_ms
    dataset.arrival_input_audit = None

    def retrieve(idx, cur_ego_pose_flag=False):
        scene_id = int(np.searchsorted(dataset.len_record, idx, side="right"))
        previous = int(dataset.len_record[scene_id - 1]) if scene_id else 0
        scene = dataset.scenario_database[scene_id]
        ego_id = next(k for k, x in scene.items() if isinstance(k, int) and x["ego"])
        remote_id = next(k for k, x in scene.items() if isinstance(k, int) and not x["ego"])
        ego_content = scene[ego_id]
        keys = [k for k in ego_content if k not in {"agent", "ego"}]
        current = ego_content[keys[idx - previous]]
        sequence = str(current["sequence"])
        audit = catalogue.select(sequence, str(current["timestamp"]), delay_ms)
        # Retrieve current ego and its original cooperative GT only. The
        # released retrieval cannot touch a nominal/future remote sensor.
        dataset.scenario_database[scene_id] = OrderedDict([(ego_id, ego_content)])
        old_delay = dataset.agent_i_delay
        dataset.agent_i_delay = 0
        try:
            base = ego_retrieval(idx, cur_ego_pose_flag=cur_ego_pose_flag)
        finally:
            dataset.scenario_database[scene_id] = scene
            dataset.agent_i_delay = old_delay
        dataset.COM_RANGE = 10000.0
        ego_pose = base[ego_id]["params"]["lidar_pose"]
        if audit["no_packet_fallback"]:
            # Exact existing ego intervention layout: retain a padded remote
            # slot, exclude it by COM_RANGE=0, and retain current cooperative GT.
            remote_pose = ego_pose.copy()
            remote_pose[0, 3] += 1
            transform = np.eye(4)
            paths = [""]
            lidars = [np.zeros((0, 4), dtype=np.float32)]
            age_frames = 0.0
            dataset.COM_RANGE = 0.0
            calibration_paths = []
        else:
            info = catalogue.info["infrastructure-side"][audit["effective_infrastructure_frame"]]
            if info["sequence_id"] != sequence:
                raise RuntimeError("Remote source sequence mismatch")
            path = catalogue.root / "infrastructure-side" / info["pointcloud_path"]
            calibration = catalogue.root / "infrastructure-side" / info["calib_virtuallidar_to_world_path"]
            remote_pose = pose_from_json(calibration, np)
            transform = np.linalg.inv(ego_pose) @ remote_pose
            paths = [str(path)]
            age_frames = audit["actual_age_ms"] * dataset.lidar_freq / 1000
            lidars = dataset.get_his_lidar(paths, [remote_pose], cav_one_hot=remote_id, delay=age_frames)
            calibration_paths = [str(calibration)]
        remote = {"ego": False, "time_delay": age_frames,
                  "params": {"transformation_matrix": transform,
                             "gt_transformation_matrix": np.eye(4),
                             "spatial_correction_matrix": np.eye(4),
                             "lidar": lidars, "lidar_path": paths,
                             "lidar_pose": remote_pose,
                             "labels": [np.zeros((0, 12))],
                             "his_lidar_pos": [ego_pose]},
                  "lidar_np": lidars, "gt_his_label": []}
        data = OrderedDict([(ego_id, base[ego_id]), (remote_id, remote)])
        data.update((k, v) for k, v in base.items() if not isinstance(k, int))
        ego_info = catalogue.info["vehicle-side"][audit["effective_vehicle_frame"]]
        ego_calibs = [catalogue.root / "vehicle-side" / ego_info[k]
                      for k in ("calib_lidar_to_novatel_path", "calib_novatel_to_world_path")]
        audit.update(global_index=int(idx), source_pose=remote_pose.tolist() if paths[0] else None,
                     current_ego_pose=ego_pose.tolist(), remote_to_current_ego=transform.tolist(),
                     calibration_sha256={str(x): sha(x) for x in calibration_paths + ego_calibs},
                     actual_time_delay_frames=[0.0, age_frames],
                     system_error_offset_used=False,
                     remote_labels_loaded=False,
                     physical_clock_independently_verified=False)
        projected = lidars[0][:, :3] @ transform[:3, :3].T + transform[:3, 3]
        bounds = np.asarray(p["voxelization"]["lidar_range"])
        in_range = ((projected >= bounds[:3]) & (projected < bounds[3:])).all(axis=1)
        ego_body = ((projected[:, 0] >= -1.95) & (projected[:, 0] <= 2.95)
                    & (projected[:, 1] >= -1.1) & (projected[:, 1] <= 1.1))
        audit.update(remote_raw_points=int(len(projected)),
                     remote_points_in_eval_range=int((in_range & ~ego_body).sum()))
        dataset.arrival_input_audit = audit
        return data

    dataset.retrieve_base_data = retrieve


def path_audit(dataset, batch, row, delay_ms, data_root, frame_his, cav_frame_his, lidar_frequency_hz):
    audit = dict(dataset.arrival_input_audit)
    paths = [str(x) for x in batch["lidar_path"]]
    stems = [Path(x).stem if x else "" for x in paths]
    expected = [audit["effective_vehicle_frame"], audit["effective_infrastructure_frame"] or ""]
    delays = [float(x) for x in batch["time_delays"][0]]
    if audit["global_index"] != row["global_index"] or stems != expected:
        raise RuntimeError("Actual model path/source identity differs from arrival selection")
    if not all(abs(a - b) < 1e-6 for a, b in zip(delays, audit["actual_time_delay_frames"])):
        raise RuntimeError("Actual age encoding differs from selected sensor timestamps")
    if any(not Path(x).is_file() or not Path(x).is_relative_to(Path(data_root).resolve()) for x in paths if x):
        raise RuntimeError("Selected packet path missing or outside shared data view")
    expected_len = [1 if audit["no_packet_fallback"] or audit["remote_points_in_eval_range"] == 0 else 2]
    if batch["record_len"].tolist() != expected_len:
        raise RuntimeError(f"Actual agent cardinality differs at {row['global_index']}: {batch['record_len'].tolist()} != {expected_len}; sources={stems}")
    if not audit["no_packet_fallback"] and audit["actual_age_ms"] < delay_ms:
        raise RuntimeError("An ineligible or future packet reached model input")
    audit.update(input_path_stems=stems, expected_path_stems=expected,
                 record_len=batch["record_len"].tolist(), expected_remote_delay_frames=delays[1],
                 received_packet_without_in_range_points=not audit["no_packet_fallback"] and audit["remote_points_in_eval_range"] == 0)
    return audit
