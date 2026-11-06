"""Existing FFNet strict outputs on actual full official common GT, not intersection."""
import argparse
import json
import time
import traceback
from collections import defaultdict
import numpy as np
from peer16_v1_20261103 import ROOT,BASE,atomic,ev,load,now,register,record
from run_peer16_cached_methods_v1_20261103 import load_author
from audit_v2xseq_strict_common_roi_20261010 import mask
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import match_frame,voc_ap
OUT=BASE/'E11_full_common_GT'

def prepare():
    register('E11_full_common_GT',__file__,
        'Prior FFNet common-intersection drops GT by native support first, '
        'so it is not the full31123 common-GT baseline. Actually rematch '
        'immutable val13264 saved fused outputs against full unchanged '
        'current cooperative common GT. Both frozen serializer conventions '
        'reported without AP selection. Existing native-intersection '
        'evidence retained, not renamed. This supports same-population '
        'comparison with the new standalone-ego control once both complete.',
        dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='0.2-3',disk_MiB='<15 compact matches; source predictions/GT referenced'),
        old=ev(BASE/'E11_strict_author_serializer/contract.json'),
        old_independent=ev(BASE/'E11_common_official_independent/terminal.json'),
        scope='Full main common GT, retained native model predictions/support; BEV only, source/aux independent separate, no automatic manuscript row replacement')

def run():
    start=time.monotonic();record('E11_full_common_GT',dict(status='running_full_GT_actual_AP'))
    try:
        c=load(BASE/'E11_strict_author_serializer/contract.json');original=load(ROOT/c['original_contract']['path']);manifest=load(ROOT/c['completed_manifest']['path'])
        meta=[json.loads(s) for s in (ROOT/original['inputs']['strict_conditions']['val']['path']).read_text().splitlines()]
        m,rows=load_author('val21');lookup={(str(r['sequence']),int(r['wait_ms']),str(r['current_vehicle_frame'])):i for i,r in enumerate(rows)}
        assert len(lookup)==13264
        ledgers=defaultdict(lambda:dict(score=[],tp=[],fp=[],GT=0,offset=[0]));seen=set();count=0
        for shard in manifest['shards']:
            assert ev(ROOT/shard['path'])=={k:shard[k] for k in ('path','sha256','bytes')}
            with np.load(ROOT/shard['path']) as z:
                for j,index in enumerate(z['condition_index']):
                    if str(z['split'][j])!='val':continue
                    s=meta[int(index)];key=str(z['sequence'][j]),int(z['wait_ms'][j]),str(s['vehicle']['frame'])
                    assert key not in seen;seen.add(key);i=lookup[key]
                    ga,gb=map(int,m['gt_offsets'][i:i+2]);gt=m['gt_boxes'][ga:gb];gt=gt[mask(gt)]
                    a,b=map(int,z['pred_offsets'][j:j+2]);native=z['pred_boxes_hwl'][a:b].copy();native[:,6]*=-1;ss=z['pred_scores'][a:b]
                    for kind in ('native_canonical','author_serializer'):
                        boxes=native.copy()
                        if kind=='author_serializer':boxes[:,[4,5]]=boxes[:,[5,4]]
                        keep=mask(boxes)
                        for th in (50,70):
                            score,tp,fp,n=match_frame(boxes[keep],ss[keep],np.zeros(keep.sum()),gt,np.zeros(len(gt)),th/100)
                            d=ledgers[kind,key[1],th];d['score'].extend(score);d['tp'].extend(tp);d['fp'].extend(fp);d['GT']+=n;d['offset'].append(len(d['score']))
                    count+=1
                    if count%256==0:atomic(OUT/'status.json',dict(created_at=now(),conditions=count,total=13264,wall_s=time.monotonic()-start))
        assert len(seen)==13264
        panels=[]
        for (kind,wait,th),d in sorted(ledgers.items()):
            assert d['GT']==31123
            value=voc_ap(d['tp'],d['fp'],d['score'],d['GT'],True)
            panels.append(dict(convention=kind,wait_ms=wait,threshold=th,BEV_AP=100*value['ap'],GT=d['GT'],frames=3316))
            np.savez_compressed(OUT/(kind+'_wait'+str(wait)+'_AP'+str(th)+'_matches.npz'),**{k:np.asarray(v) for k,v in d.items()})
        atomic(OUT/'panels.json',panels)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',panels=ev(OUT/'panels.json'),GPU_forwards=0,wall_s=time.monotonic()-start,full_common_GT=True,old_intersection_retained=True,not_3D_AP=True))
        record('E11_full_common_GT',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json')))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
        record('E11_full_common_GT',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
