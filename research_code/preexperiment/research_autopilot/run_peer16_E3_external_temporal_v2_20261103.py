"""Centroid displacement decomposition is exact even with original 3D tilt."""
import argparse
import inspect
import numpy as np
import run_peer16_E3_external_temporal_v1_20261103 as old
from peer16_v1_20261103 import BASE,ev,load,register
OUT=BASE/'E3_external_temporal_v2'

def translations(m,b,valid):
    z=[];error=0.
    for a,e in zip(m['condition_offsets'][:-1],m['condition_offsets'][1:]):
        a,e=int(a),int(e);ids=a+np.flatnonzero(m['source'][a:e]==1)
        shift=b[ids,:2].astype(float)-m['stale_boxes'][ids,:2].astype(float)
        mean=shift.mean(0) if len(ids) else np.zeros(2)
        if len(ids):error=max(error,float(np.max(abs((shift-mean)+mean-shift))))
        z.append(mean)
    return np.asarray(z),error

def prepare():
    c=load(old.OUT/'contract.json')
    extra={k:v for k,v in c.items() if k not in ('created_at','script','instructions','prediction_before_result','cost','no_model_wakeup','official_main_unchanged')}
    extra['geometry']='Exact mean XY displacement over same message remote candidates, relative to stale. Smooth that centroid translation; leave instantaneous centered displacement field, yaw, dimensions and z unchanged. This is a translation-component control, not a full SE3 temporal estimator.'
    extra['repair']='Original full 3D source transform has tilt; yaw-only SE2 reconstruction is not exact (4mm at condition23). No scientific output completed. Replace invalid planar assertion with exact centroid decomposition; no tolerance relaxation, no rerun of completed scores.'
    extra['predecessor']=ev(old.OUT/'terminal.json')
    register('E3_external_temporal_v2',__file__,c['prediction_before_result'],c['cost'],**extra)

def run():
    s=inspect.getsource(old.run).replace("'E3_external_temporal'","'E3_external_temporal_v2'")
    ns=dict(old.__dict__,OUT=OUT,translations=translations,__file__=__file__)
    exec(compile(s,__file__+'::centroid_temporal_control','exec'),ns);ns['run']()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
