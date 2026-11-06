"""Finite independent geometric AP checks for filter and readout paper assets."""
import argparse
import time
import traceback
from collections import defaultdict
import numpy as np
from peer16_v1_20261103 import ROOT,BASE,atomic,ev,load,now,register,record
from run_peer16_cached_methods_v1_20261103 import materials
from review_crosscheckpoint_current_q005_heading_geometry_peer06_v1_20261030 import heading_boxes
import review_traf_v2xseq_adapted_fixed60_raw_20261006 as ind
from object_history_selector_core_v1_20261021 import common_mask
OUT=BASE/'E4_E7_independent_AP'

def prepare():
    register('E4_E7_independent_AP',__file__,
        'Rebuild heading-corrected boxes from saved state and independently recompute IoU/TP/VOC for E4 factors and all8 E7 heads, both roles and four waits. No retraining/tuning, discrepancies retained.',
        dict(GPU_forwards=0,CPU_threads=1,wall_h='2-10',disk_MiB='<6 metrics only; no copied pools/GT'),
        producers=[ev(BASE/x/'terminal.json') for x in ('E4_filter_design_v2','E7_readout_v3')],
        scope='Independent box heading projection/float32 translation, polygon3D IoU, class-zero one-to-one matching and pooled VOC. Shares saved states, candidate selection and saved head scores. Not independent head optimization or neural scoring; source/noanchor/distance and threshold sweep remain separate.')

def geometry(m,d):
    b=m['cv_boxes'].copy()
    for i,(a,e) in enumerate(zip(m['condition_offsets'][:-1],m['condition_offsets'][1:])):
        a,e=int(a),int(e);ids=a+np.flatnonzero(m['source'][a:e]==1)
        b[ids]=heading_boxes(m['stale_boxes'][ids],m['cv_boxes'][ids]);b[ids,:2]+=d[i].astype(np.float32)
    return b

def check(m,b,p,name,gold):
    stats=defaultdict(lambda:dict(score=[],tp50=[],tp70=[],GT=0,mis50=0,mis70=0))
    for i,w in enumerate(m['wait_ms']):
        a,e=map(int,p[name+'__offset'][i:i+2]);ids=p[name+'__candidate'][a:e].astype(int);score=p[name+'__score'][a:e]
        ga,ge=map(int,m['gt_offsets'][i:i+2]);gt=m['gt_boxes'][ga:ge];gt=gt[common_mask(gt)];s=stats[int(w)];s['GT']+=len(gt);s['score'].extend(score)
        assert (ids>=m['condition_offsets'][i]).all() and (ids<m['condition_offsets'][i+1]).all()
        for th in (50,70):
            _,tp,_,ng=ind.match_frame(b[ids],score,np.zeros(len(ids)),gt,np.zeros(len(gt)),th/100);assert ng==len(gt)
            s['tp'+str(th)].extend(tp);s['mis'+str(th)]+=int(np.sum(np.asarray(tp)!=p[name+'__tp'+str(th)][a:e]))
    result=[]
    for w,s in sorted(stats.items()):
        row=dict(method=name,wait_ms=w,GT=s['GT']);expected=next(x for x in gold if x['method']==name and x['wait_ms']==w)
        for th in (50,70):
            value=100*ind.voc_ap(s['tp'+str(th)],(1-np.asarray(s['tp'+str(th)])).tolist(),s['score'],s['GT'],True)['ap'];row['AP'+str(th)]=value;row['error'+str(th)]=value-expected['ap'+str(th)];row['TP_mismatch'+str(th)]=s['mis'+str(th)]
        result.append(row)
    return result

def run():
    start=time.monotonic();record('E4_E7_independent_AP',dict(status='running_saved_state_independent_geometry_AP'))
    try:
        data,_,_=materials('author');results=[]
        for role in ('mechanism_holdout14','val21'):
            m,_=data[role];src=BASE/'E4_filter_design_v2';states=np.load(src/(role+'_states.npz'));gold=load(src/(role+'_panels.json'))
            for arm in sorted({r['method'] for r in gold if not r['method'].endswith('_raw')}):
                b=geometry(m,states[arm+'__delta']);p=np.load(src/(role+'_'+arm+'_predictions.npz'))
                for name in (arm,arm+'_raw'):results.extend(dict(r,family='E4',role=role) for r in check(m,b,p,name,gold))
                atomic(OUT/'results.json',results);atomic(OUT/'status.json',dict(created_at=now(),family='E4',role=role,arm=arm,reviewed_panels=len(results)))
            with np.load(BASE/'E3_temporal_baselines'/(role+'_states.npz')) as z:d=z['KF'].copy()
            b=geometry(m,d)
            for head in sorted((BASE/'E7_readout_v3').glob('*_seed*')):
                if not (head/'panels.json').exists():continue
                pp=[r for r in load(head/'panels.json') if r['role']==role];p=np.load(head/(role+'_predictions.npz'))
                results.extend(dict(r,family='E7',role=role,head=head.name) for r in check(m,b,p,head.name,pp))
                atomic(OUT/'results.json',results);atomic(OUT/'status.json',dict(created_at=now(),family='E7',role=role,head=head.name,reviewed_panels=len(results)))
        exact=all(all(abs(r['error'+str(th)])<1e-8 and r['TP_mismatch'+str(th)]==0 for th in (50,70)) for r in results)
        status='complete_independently_reviewed' if exact else 'completed_discrepancy_retained'
        atomic(OUT/'terminal.json',dict(created_at=now(),status=status,results=ev(OUT/'results.json'),GPU_forwards=0,wall_s=time.monotonic()-start,scope=load(OUT/'contract.json')['scope']));record('E4_E7_independent_AP',dict(status=status,terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record('E4_E7_independent_AP',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
