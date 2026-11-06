"""Native TraF 4+4 eval input adapter, never an author-algorithm repair.

Complete metadata determines the latest eligible packet before resolving files.
Insufficient histories repeat the latest legal source (released loader's path
padding convention), with that source's own pose. No invented past measurement.
Released rank-based temporal embedding remains unchanged; actual timestamps and
its mismatch to rank encoding are explicitly retained. No historical annotation
is retrieved. Only current cooperative GT is used in evaluation bookkeeping.
"""
import bisect
from collections import OrderedDict, defaultdict
import copy
import hashlib
import json
from pathlib import Path

from v2xseq_recorded_arrival_20261009 import pose_from_json
from evaluate_traf_v2xseq_adapted_fixed60_raw_20261006 import sha256

PROTOCOL_ID = 'traf_author_native4_recorded_arrival_sensor_extrinsics_v1'


class Catalogue:
    def __init__(self, roots):
        self.roots = [Path(x).resolve() for x in roots]
        self.paths = {side:self.roots[0]/side/'data_info.json' for side in ('vehicle-side','infrastructure-side')}
        self.info = {side:{x['frame_id']:x for x in json.loads(path.read_text())} for side,path in self.paths.items()}
        self.grouped = {side:defaultdict(list) for side in self.info}
        for side,info in self.info.items():
            for frame,x in info.items():
                self.grouped[side][x['sequence_id']].append((int(x['pointcloud_timestamp']),frame))
            for entries in self.grouped[side].values():
                entries.sort()

    def history(self, side, sequence, decision, wait):
        choices = self.grouped[side][sequence]
        cutoff = decision-(wait*1000 if side=='infrastructure-side' else 0)
        end = bisect.bisect_right(choices,(cutoff,'\uffff'))
        return [dict(frame_id=frame,sensor_time_us=stamp,age_ms=(decision-stamp)/1000)
                for stamp,frame in reversed(choices[max(0,end-4):end])]

    def locate(self, side, info, field):
        relative = side+'/'+info[field]
        for root in self.roots:
            path = root/relative
            if path.is_file():
                return path
        raise FileNotFoundError('Latest legal source is not staged: '+relative)

    def load(self, side, entries, np):
        padded = [dict(x,padding_replica=False) for x in entries]
        if entries:
            padded += [dict(entries[0],padding_replica=True) for _ in range(4-len(entries))]
        for rank,x in enumerate(padded):
            info = self.info[side][x['frame_id']]
            path = self.locate(side,info,'pointcloud_path')
            if side=='vehicle-side':
                calibs = [self.locate(side,info,k) for k in ('calib_lidar_to_novatel_path','calib_novatel_to_world_path')]
                pose = pose_from_json(calibs[1],np) @ pose_from_json(calibs[0],np,nested=True)
            else:
                calibs = [self.locate(side,info,'calib_virtuallidar_to_world_path')]
                pose = pose_from_json(calibs[0],np)
            x.update(path=str(path),source_pose=pose.tolist(),history_rank=rank,
                     calibration_sha256={str(p):sha256(p) for p in calibs},pointcloud_sha256=sha256(path),
                     author_encoded_age_ms=entries[0]['age_ms']+rank*100.0,
                     actual_minus_author_age_ms=x['age_ms']-(entries[0]['age_ms']+rank*100.0))
        return padded


def install(ds, np, roots, wait_ms, mode='strict'):
    p = ds.params
    assert not ds.train and ds.ego_agent=='vehicle' and ds.proj_first
    assert p['dataset']['frame_his']==p['dataset']['cav_frame_his']==4
    assert not p['model']['reader']['timestamp'] and not p['wild_setting'].get('one_hot_cav_id',False)
    assert ds.max_cav==2 and not p['dataset']['merge_pcd_first'] and ds.lidar_freq==10
    cat = Catalogue(roots)
    ds.native4_catalogue = cat
    ds.native4_audit = None
    ds.native4_projected = {}
    original_single = ds.get_item_single_car

    def capture_single(value):
        processed,void = original_single(value)
        side = 'vehicle-side' if value['ego'] else 'infrastructure-side'
        ds.native4_projected[side] = [x.copy() for x in processed['projected_lidar']]
        return processed,void
    ds.get_item_single_car = capture_single

    def retrieve(idx, cur_ego_pose_flag=False):
        ds.native4_projected = {}
        scene_id = int(np.searchsorted(ds.len_record,idx,side='right'))
        prev = int(ds.len_record[scene_id-1]) if scene_id else 0
        scene = ds.scenario_database[scene_id]
        ego = next(k for k,x in scene.items() if isinstance(k,int) and x['ego'])
        remote = next(k for k,x in scene.items() if isinstance(k,int) and not x['ego'])
        keys = [k for k in scene[ego] if k not in ('ego','agent')]
        current = scene[ego][keys[idx-prev]]
        frame,sequence = str(current['timestamp']),str(current['sequence'])
        decision = int(cat.info['vehicle-side'][frame]['pointcloud_timestamp'])
        original_ego = cat.history('vehicle-side',sequence,decision,0)
        assert original_ego[0]['frame_id']==frame
        original_remote = cat.history('infrastructure-side',sequence,decision,wait_ms)
        if mode=='ego':
            original_remote = []
        elif mode=='nominal':
            paired = str(scene[remote][keys[idx-prev]]['timestamp'])
            info = cat.info['infrastructure-side'][paired]
            assert info['sequence_id']==sequence
            original_remote = cat.history('infrastructure-side',sequence,int(info['pointcloud_timestamp']),0)
            original_remote = [dict(x,age_ms=(decision-x['sensor_time_us'])/1000) for x in original_remote]
            assert original_remote[0]['frame_id']==paired
        else:
            assert mode=='strict'
        histories = {side:cat.load(side,entries,np) for side,entries in
                     [('vehicle-side',original_ego),('infrastructure-side',original_remote)]}
        ego_pose = np.asarray(histories['vehicle-side'][0]['source_pose'])
        result = OrderedDict()
        current_gt = ds.generate_object_center([ds.coop_label_dict[frame]],time=0)
        from utils.pcd_utils import mask_ego_points
        masked_gt = mask_ego_points(current_gt.copy())
        for cav_id,side in [(ego,'vehicle-side'),(remote,'infrastructure-side')]:
            items = histories[side]
            if items:
                poses = [np.asarray(x['source_pose']) for x in items]
                paths = [x['path'] for x in items]
                age_frames = items[0]['age_ms']/100.0
                # Original source-local history projection; subsequent original
                # get_item_single_car projects latest source to current ego.
                lidars = ds.get_his_lidar(paths,poses,cav_one_hot=cav_id,delay=age_frames)
                pose = poses[0]
                transform = np.linalg.inv(ego_pose) @ pose
            else:
                paths = ['']*4
                lidars = [np.zeros((0,4),dtype=np.float32) for _ in range(4)]
                pose = ego_pose.copy(); pose[0,3] += 1
                transform = np.eye(4); age_frames = 0.0
            if side=='vehicle-side':
                # Inputs already aligned into current ego. Original eval loader
                # does not externally project ego histories (delay_ego=0).
                assert np.allclose(transform,np.eye(4),atol=1e-8)
            result[cav_id] = dict(ego=cav_id==ego,time_delay=age_frames,
                params=dict(transformation_matrix=transform,gt_transformation_matrix=np.eye(4),
                            spatial_correction_matrix=np.eye(4),lidar=lidars,lidar_path=paths,
                            lidar_pose=pose,his_lidar_pos=[ego_pose]*4,
                            labels=[np.zeros((0,12))]),lidar_np=lidars,
                gt_his_label=[current_gt] if cav_id==ego else [])
            for x in items:
                x['source_to_current_ego'] = (np.linalg.inv(ego_pose) @ np.asarray(x['source_pose'])).tolist()
        ds.COM_RANGE = 10000.0 if original_remote else 0.0
        result.update(token_dict=copy.deepcopy(ds.vehicle_infos[sequence][frame]),
                      timestamp=frame,label=ds.coop_label_dict[frame],
                      gt_boxes=masked_gt,gt_boxes_ego=masked_gt.copy())
        ds.native4_audit = dict(protocol_id=PROTOCOL_ID,global_index=int(idx),sequence=sequence,
            current_vehicle_frame=frame,effective_vehicle_frame=frame,effective_label_frame=frame,
            decision_us=decision,wait_ms=wait_ms,selection_mode=mode,
            available_ego_history=len(original_ego),available_remote_history=len(original_remote),
            no_packet_fallback=not original_remote,history=histories,current_ego_pose=ego_pose.tolist(),
            system_error_offset_used=False,historical_GT_retrieved=False,remote_GT_retrieved=False,
            physical_clock_independently_verified=False,
            temporal_encoding_policy='unmodified_author_latest_age_plus_rank_100ms; true age and padding recorded',
            padding_policy='missing-history slots repeat newest legal measurement and its own pose, not synthetic older observations')
        return result
    ds.retrieve_base_data = retrieve


def batch_audit(ds, batch, np):
    audit = copy.deepcopy(ds.native4_audit)
    entries = [x for side in ('vehicle-side','infrastructure-side') for x in audit['history'][side]]
    expected = [x['path'] for x in entries[:4]]+([x['path'] for x in entries[4:]] if not audit['no_packet_fallback'] else ['']*4)
    assert list(batch['lidar_path'])==expected and len(expected)==8
    for side,values in audit['history'].items():
        for x in values:
            info = ds.native4_catalogue.info[side][x['frame_id']]
            assert info['sequence_id']==audit['sequence']
            if side=='vehicle-side' or audit['selection_mode']=='strict':
                assert x['sensor_time_us']<=audit['decision_us']-(audit['wait_ms']*1000 if side=='infrastructure-side' else 0)
    actual = [float(x) for x in batch['time_delays'][0]]
    wanted = [audit['history']['vehicle-side'][0]['age_ms']/100.0,
              audit['history']['infrastructure-side'][0]['age_ms']/100.0 if not audit['no_packet_fallback'] else 0.0]
    assert np.allclose(actual,wanted,rtol=0,atol=2e-5)
    slots = sorted(set(int(x) for x in batch['processed_lidar']['voxel_coords'][:,0].tolist()))
    n = int(batch['record_len'][0])
    assert slots==list(range(n)) and 4<=n<=8
    # First four occupied slots must remain ego history; otherwise released
    # regroup would mistake a remote slot for an ego slot. Qualification gate.
    for raw in ds.native4_projected['vehicle-side']:
        assert len(raw)>0, 'native ego4 rank layout cannot compact an empty ego slot'
    if audit['no_packet_fallback']:
        assert n==4
    audit.update(model_input_paths=expected,model_time_delays=actual,record_len=[n],voxel_slots=slots,
                 projected_point_counts={side:[len(x) for x in values] for side,values in ds.native4_projected.items()},
                 processed_tensor_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in batch['processed_lidar'].items()})
    return audit


def inference_input(batch):
    # Author model consumes only these fields. Supervision/token/field/GT and
    # raw labels cannot enter inference, even though native loader constructs
    # annotation-derived loss targets as part of its ordinary collation.
    return {k:batch[k] for k in ('processed_lidar','record_len','time_delays')}
