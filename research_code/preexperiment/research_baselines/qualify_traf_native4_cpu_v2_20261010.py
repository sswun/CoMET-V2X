"""Real native4 loader qualification; reuse complete source metadata preparation."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import sys
import time

import traf_native4_recorded_arrival_20261010 as adapter
from evaluate_traf_v2xseq_adapted_fixed60_raw_20261006 import atomic_json, append_jsonl, dataset_rows, now, sha256


def build(root):
    source = root/'third_party/third_party/related_works/TraF-Align'
    sys.path.insert(0,str(source))
    from datasets import build_dataset
    from hypes_yaml import yaml_utils
    config = root/'artifacts/checkpoints/traf_align_v2xseq_0f6f98de/config.yaml'
    split = source/'datasets/Basedataset/V2XSeq_dataset_split_official.yaml'
    author = root/'artifacts/data/v2x_seq_spd_traf_align_author21_v1/V2X-Seq-SPD'
    shared = root/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD'
    hypes = yaml_utils.load_yaml(str(config))
    hypes.update(root_dir=str(author)+os.sep,split_dir=str(split),delay_ego=0)
    hypes['train_params'].update(train_batch_size=1,val_batch_size=1)
    hypes['dataset']['augment'] = False
    hypes = yaml_utils.check_pillar_params(hypes)
    assert hypes['dataset']['frame_his']==hypes['dataset']['cav_frame_his']==4
    return hypes,build_dataset,[author,shared],config,split


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--metadata-summary',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a = p.parse_args(); root=a.workspace.resolve(); out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    prep=json.loads(a.metadata_summary.read_text())
    assert prep['status']=='passed' and prep['missing_count']==0 and all(prep['checks'].values())
    assert all(sha256(Path(x))==h for x,h in prep['frozen_inputs'].items())
    plan=a.metadata_summary.parent/'source_history_plan.jsonl'
    assert sha256(plan)==prep['source_plan_sha256']
    planned={(x['current_vehicle_frame'],x['wait_ms']):x for x in (json.loads(s) for s in plan.read_text().splitlines())}
    hypes,builder,roots,config,split=build(root)
    reference=builder(copy.deepcopy(hypes),set='val')
    rows=dataset_rows(reference)
    assert len(rows)==3316 and len({x['sequence'] for x in rows})==21
    boundaries=set()
    for scene in range(21):
        selected=[x['global_index'] for x in rows if x['scene_id']==scene]
        boundaries.update((selected[0],selected[len(selected)//2],selected[-1]))
    checks={}; arms={}; counts=Counter(); real_count=0
    for wait in (0,100,200,400):
        ds=builder(copy.deepcopy(hypes),set='val')
        adapter.install(ds,np,roots,wait)
        ego=builder(copy.deepcopy(hypes),set='val'); adapter.install(ego,np,roots,wait,mode='ego')
        cache=json.dumps(ds.coop_label_dict,sort_keys=True)
        qualify=set(boundaries)
        for row in rows:
            frame=str(ds.scenario_database[row['scene_id']][0][row['timestamps'][row['local_index']]]['timestamp'])
            expected=planned[(frame,wait)]
            for side in ('vehicle-side','infrastructure-side'):
                actual=ds.native4_catalogue.history(side,row['sequence'],expected['decision_us'],wait)
                assert [(x['frame_id'],x['sensor_time_us']) for x in actual]==[(x['frame_id'],x['sensor_time_us']) for x in expected[side+'_history']]
            if len(expected['vehicle-side_history'])<4 or len(expected['infrastructure-side_history'])<4:
                qualify.add(row['global_index'])
        for index in sorted(qualify):
            np.random.seed(2023+index)
            batch=ds.collate_batch([ds[index]])
            audit=adapter.batch_audit(ds,batch,np)
            frame=audit['current_vehicle_frame']
            # Independent current annotation geometry, no historical sources.
            from utils.pcd_utils import mask_ego_points
            expected=ds.mask_gt_box_out_of_range(mask_ego_points(ds.generate_object_center([ds.coop_label_dict[frame]])))
            label=batch['label_dict'].numpy()
            if expected.size:
                assert label.shape==(len(expected),expected.shape[1]+1)
                assert np.allclose(label[:,1:],expected,rtol=0,atol=2e-5)
            else:
                assert label.size==0, 'Empty current GT must remain empty'
            for side,entries in audit['history'].items():
                for x in entries:
                    direct=np.linalg.solve(np.asarray(audit['current_ego_pose']),np.asarray(x['source_pose']))
                    assert np.allclose(direct,x['source_to_current_ego'],rtol=0,atol=1e-7)
                    # Independently verify sampled consumed coordinates before
                    # voxelization after two native projection stages.
                    from utils.pcd_utils import pcd_to_np
                    points=pcd_to_np(x['path'])[:,:3]
                    projected=points @ direct[:3,:3].T+direct[:3,3]
                    body=(projected[:,0]>=-1.95)&(projected[:,0]<=2.95)&(projected[:,1]>=-1.1)&(projected[:,1]<=1.1)
                    projected=projected[~body]
                    raw=ds.native4_projected[side][x['history_rank']][:,:3]
                    # Upstream mask_ego_points happens before voxel range cut.
                    assert raw.shape==projected.shape and np.allclose(raw,projected,rtol=2e-5,atol=5e-4)
            if audit['no_packet_fallback']:
                np.random.seed(2023+index); ego_batch=ego.collate_batch([ego[index]])
                for k in ('record_len','prior_encoding','pairwise_t_matrix','spatial_correction_matrix','label_dict'):
                    assert torch.equal(batch[k],ego_batch[k]),k
                assert all(torch.equal(batch['processed_lidar'][k],ego_batch['processed_lidar'][k]) for k in batch['processed_lidar'])
            audit.update(current_GT_exact=True,independent_history_projection_exact=True,
                         fallback_tensors_exact=True if audit['no_packet_fallback'] else None)
            append_jsonl(out/'real_batches.jsonl',audit)
            counts[str(wait)+'_ego_available_'+str(audit['available_ego_history'])]+=1
            counts[str(wait)+'_remote_available_'+str(audit['available_remote_history'])]+=1
            real_count+=1
        assert cache==json.dumps(ds.coop_label_dict,sort_keys=True)
        arms[str(wait)]=dict(real_batches=len(qualify),metadata_frames=3316)
        checks[str(wait)+'_metadata_matches_reused_plan']=True
        checks[str(wait)+'_boundaries_insufficient_no_packet_currentGT_projection_fallback']=True
        print(json.dumps(dict(wait_ms=wait,real_batches=len(qualify))),flush=True)
        del ds,ego
    # Import actual author model, but deliberately no CUDA/model forward.
    from models.traf_align import TraFAlign
    checks.update(full_author_model_import=True,no_CUDA_initialized=not torch.cuda.is_initialized(),
                  original_config_split_plan_immutable=sha256(plan)==prep['source_plan_sha256'],
                  no_historical_or_remote_GT_in_inference=True)
    frozen={str(x):sha256(x) for x in (Path(__file__),Path(adapter.__file__),config,split,plan,a.metadata_summary.resolve())}
    summary=dict(status='passed',checks=checks,observed_at=now(),real_batches=real_count,arms=arms,
        history_case_counts=dict(counts),frozen_inputs=frozen,protocol_id=adapter.PROTOCOL_ID,
        new_detector_forwards=0,wall_seconds=time.monotonic()-tick,
        boundary='Native4 legal-source loader/geometry qualification only. Latest available history rank embedding unchanged, not exact irregular-age embedding, strict1 information-budget match or completed effect reproduction.')
    atomic_json(out/'summary.json',summary)
    print(json.dumps(dict(status='passed',checks=len(checks),real_batches=real_count,wall_seconds=summary['wall_seconds'])))


if __name__=='__main__':main()

