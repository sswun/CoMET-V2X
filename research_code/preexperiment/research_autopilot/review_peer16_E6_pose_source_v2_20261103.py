"""Audit 12-column loader GT identity, then independently score common-ROI GT."""
import argparse, hashlib, inspect, os, runpy, sys
from types import SimpleNamespace
import numpy as np
from peer16_v1_20261103 import ROOT, BASE, atomic, ev, load, now, register
import review_peer16_E6_pose_source_v1_20261103 as old
OUT=BASE/'E6_independent_v2'
LABEL_CONTEXT={}
def current_gt(method,base,record):
    if method not in LABEL_CONTEXT:
        runpy.run_path(str(ROOT/'preexperiment/research_autopilot/compat_traf_numpy124_thermal_v4/sitecustomize.py'),run_name='approved_label_compat')
        src=ROOT/'third_party/third_party/related_works/TraF-Align'
        sys.path.insert(0,str(src))
        from hypes_yaml import yaml_utils
        from datasets.Basedataset.v2xseq_basedataset import V2XSEQBaseDataset
        from utils.pcd_utils import mask_ego_points
        c=load(old.PROD/'contract.json');h=yaml_utils.load_yaml(str(ROOT/c['methods'][method]['config_identity']['path']))
        LABEL_CONTEXT[method]=(h,V2XSEQBaseDataset,mask_ego_points)
    h,cls,remove_ego=LABEL_CONTEXT[method]
    frame=base['effective_label_frame']
    p=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD'/h['dataset']['label']/(frame+'.json')
    labels=[x for x in load(p) if x['type'] not in h['dataset']['ignore_cls']]
    full=cls.generate_object_center(SimpleNamespace(params=h),[labels])
    full=remove_ego(full)
    bounds=h['dataset'].get('eval_range') if h['dataset'].get('infer_range',False) else h['voxelization']['lidar_range']
    if len(full) and full.shape[1]:
        keep=(full[:,0]>=bounds[0])&(full[:,0]<=bounds[3])&(full[:,1]>=bounds[1])&(full[:,1]<=bounds[4]);full=full[keep].astype(np.float32)
        full[:,7]=np.asarray(h['dataset']['cls_map'])[full[:,7].astype(int)]
    else:full=np.empty((0,8),np.float32)
    assert list(full.shape)==record['current_GT_shape'] and hashlib.sha256(full.tobytes()).hexdigest()==record['current_GT_sha256'],('reconstructed_12field_GT',method,frame,full.shape,record['current_GT_shape'])
    return full[:,:7],full[:,7]
ns=dict(old.__dict__,OUT=OUT,__file__=__file__,current_gt=current_gt)
source=inspect.getsource(old.review)
begin=source.index('            whole=np.column_stack((gt,gl))')
end=source.index("            if last!=r['shard']:",begin)
source=source[:begin]+'''            rebuilt,labels=current_gt(method,base,r)
            # Native architectures crop labels differently before storage;
            # official common-ROI current population must still be exact.
            assert np.array_equal(rebuilt[mask(rebuilt)],gt[mask(gt)]) and np.array_equal(labels[mask(rebuilt)],gl[mask(gt)]),('common_current_GT',method,w,i)
'''+source[end:]
exec(compile(source,__file__+'::GT_schema_adjudication','exec'),ns)
source=inspect.getsource(old.run).replace("'E6_independent'","'E6_independent_v2'")
exec(compile(source,__file__+'::finite_ready_first','exec'),ns)
def prepare():
    register('E6_independent_v2',__file__,
        'v1 compared an actual 12-field GT hash with an 8-field raw projection. Rebuild all12 fields from original current labels and per-architecture crop, require exact digest; separately require exact accepted common-ROI GT. No new inference or GT model input.',
        dict(GPU_forwards=0,CPU_threads=1,wait_h=20,disk_MiB='<30 compact outcomes'),
        producer=ev(old.PROD/'contract.json'),predecessor=ev(old.OUT/'where/terminal.json'),references=old.REF)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();prepare() if a.mode=='prepare' else ns['run']()
