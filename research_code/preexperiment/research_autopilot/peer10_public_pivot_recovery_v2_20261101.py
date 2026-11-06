"""Complete missing public poses in four-wait ledgers. No GT or detection reads."""
import copy,json,sys
from functools import lru_cache
import numpy as np
from peer10_tournament_v1_20261101 import ROOT
sys.path.insert(0,str(ROOT/'preexperiment/research_baselines'))
from v2xseq_recorded_arrival_20261009 import pose_from_json
import run_peer10_se2_expanded_anchor_v1_20261101 as science
_state=science.state
@lru_cache(None)
def infrastructure_infos():
    root=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD';return root,{str(r['frame_id']):r for r in json.loads((root/'infrastructure-side/data_info.json').read_text())}
@lru_cache(None)
def infrastructure_pose(frame):
    root,infos=infrastructure_infos();r=infos[frame];return pose_from_json(root/'infrastructure-side'/r['calib_virtuallidar_to_world_path'],np)
def rows_with_public_pivot(rows):
    output=[]
    for r in rows:
        if r.get('source') is not None and 'source_to_current_ego' not in r['source']:
            row=dict(r);row['source']=dict(r['source']);frame=str(row['source']['source_frame']);T=np.linalg.solve(np.asarray(row['current_ego_pose_lidar_to_world'],np.float64),infrastructure_pose(frame));row['source']['source_to_current_ego']=T.tolist();output.append(row)
        else:output.append(r)
    return output
def state(m,rows,*args,**kwargs):return _state(m,rows_with_public_pivot(rows),*args,**kwargs)
def install():science.state=state
