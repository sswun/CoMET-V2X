"""Shared zero-forward native adapters: current-frame/GT joins precede scoring.

All canonical boxes are HWL (verified against shared current official GT).
Common ROI is applied to ORIGINAL GT/predictions only. No shifted-GT ROI.
Only compact scored outcomes are persisted by consumers; raw boxes referenced.
"""
import json
import numpy as np
from peer20_population_core_v1_20261105 import *
from review_peer15_external_raw_AP_v1_20261103 import raw_path
from e1_receiver_value_core_v1_20261020 import iou32,greedy
import review_traf_v2xseq_adapted_fixed60_raw_20261006 as polygon

def canon(x):
    x=np.asarray(x,float).reshape(-1,7)
    return x[np.lexsort(tuple(x[:,j] for j in reversed(range(7))))] if len(x) else x

def assemble(domain,role,cases,label,review=False):
    material,rows=data(domain)[0][role]
    by={(str(r['sequence']),int(r['wait_ms']),str(r['current_vehicle_frame'])):i for i,r in enumerate(rows)}
    assert len(by)==len(rows)
    boxes=[];scores=[];truth=[];rr=[];go=[0];co=[0];hits={50:[],70:[]};seen=set()
    for r,b,s,cl,g,gl in cases:
        w=int(r['wait_ms']);frame=str(r.get('current_vehicle_frame',r.get('effective_vehicle_frame',r.get('effective_label_frame'))))
        key=str(r['sequence']),w,frame;assert key in by and key not in seen,key;seen.add(key);i=by[key]
        expected=rows[i]
        if 'global_index' in r:assert int(r['global_index'])==int(expected['global_index'])
        gm=common_mask(g)&(gl==0);gg=g[gm,:7]
        a,z=map(int,material['gt_offsets'][i:i+2]);ref=material['gt_boxes'][a:z];ref=ref[common_mask(ref)]
        assert gg.shape==ref.shape and np.allclose(canon(gg),canon(ref),rtol=0,atol=2e-4),('native_GT_identity',domain,role,key)
        pm=common_mask(b)&(cl==0);bb=b[pm,:7];ss=s[pm].astype(np.float64);order=np.argsort(-ss);bb=bb[order];ss=ss[order]
        for th in (50,70):
            if review:
                # match_frame sorts again; equal scores may reorder! Restore its
                # explicit order in both adapters rather than comparing ties to
                # another sorting pass.
                # Independent matcher below is run at already-ordered ranks by
                # strictly decreasing rank-only scores; matching is rank-based.
                _,tp,_,ng=polygon.match_frame(bb,-np.arange(len(bb),dtype=float),np.zeros(len(bb)),gg,np.zeros(len(gg)),th/100)
                assert ng==len(gg)
            else:tp=(greedy(iou32(bb,gg),np.arange(len(bb)),th/100)>=0).astype(np.int8)
            hits[th].extend(tp)
        boxes.append(bb);scores.append(ss);truth.append(gg);rr.append(dict(expected));co.append(co[-1]+len(bb));go.append(go[-1]+len(gg))
    assert len(seen)==len(rows),(domain,role,len(seen),len(rows))
    m={k:np.asarray([r[k] for r in rr]) for k in ('sequence','wait_ms','receiver_timestamp_us')}
    m.update(condition_offsets=np.asarray(co),gt_offsets=np.asarray(go),gt_boxes=np.concatenate(truth),dataset_index=np.arange(len(rr)))
    p={label+'__offset':np.asarray(co),label+'__candidate':np.arange(co[-1]),label+'__score':np.concatenate(scores),
       **{label+'__tp'+str(th):np.asarray(v,np.int8) for th,v in hits.items()}}
    return m,rr,p,np.concatenate(boxes)

def shard_cases(root,arm):
    records=load(root/'records.json')
    for w in WAITS:
        selected=[r for r in records if int(r['wait_ms'])==w]
        assert len({r['shard'] for r in selected})==1
        with np.load(root/selected[0]['shard']) as z:
            for r in selected:
                k=str(r['shard_case']);g=z['c'+k+'_gt']
                yield r,z['c'+k+'_'+arm+'_box3d_lidar'],z['c'+k+'_'+arm+'_scores'],z['c'+k+'_'+arm+'_label_preds'],g[:,:7],g[:,7]

def manifest_cases(folders):
    for w,folder in zip(WAITS,folders):
        rows=[json.loads(l) for l in (folder/'frame_manifest.jsonl').read_text().splitlines() if l.strip()]
        for r in rows:
            path=raw_path(r,folder);assert polygon.sha256(path)==r['raw_sha256']
            with np.load(path) as z:yield dict(r,wait_ms=w),z['pred_boxes'],z['pred_scores'],z['pred_labels'],z['gt_boxes'],z['gt_labels']
