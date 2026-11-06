"""Post-hoc label populations, shared-input identities, and paired sequence AP.

Zero and co-labelled masks deliberately overlap in cal8 (35 frames); nonzero
is reported separately and is never relabelled co-labelled. Neither is an input.
"""
from collections import defaultdict
import numpy as np
from peer20_v1_20261105 import *
import run_peer19_H1_assets_v1_20261104 as h1
from object_history_selector_core_v1_20261021 import common_mask
import review_peer18_final_native_statistics_v1_20261104 as independent
from analyze_v2xvit_current_geometry_sequence_bootstrap_v1_20261025 import weighted_ap_batch,voc_ap
from run_peer12_cached_ceiling_v1_20261101 import reference_deltas

WAITS=(0,100,200,400);ROLES=('calibration8','val21','mechanism_holdout14');POPS=('all','zero','colabelled','nonzero')
REC=P19/'recipe_v3_interface_recovery_v2'

def data(domain):
    result,head,q=h1.recipe.materials(domain)
    if domain=='strict60':
        oldm,rows=result['calibration8']
        by={(str(r['sequence']),int(r['receiver_timestamp_us'])):r for r in rows}
        with np.load(P17/'strict_cal4/calibration8_four_wait_actual_material.npz') as z:m={k:z[k] for k in z.files}
        selected=[]
        for s,t,w in zip(m['sequence'],m['receiver_timestamp_us'],m['wait_ms']):
            r=dict(by[str(s),int(t)]);r['wait_ms']=int(w);selected.append(r)
        result['calibration8']=(m,selected)
    return result,head,q

def annotations(rows):
    lookup={r['frame']:r for r in load(BASE/'R6a_raw_labels/frames.json')}
    values=[lookup[str(r['current_vehicle_frame'])] for r in rows]
    for r,v in zip(rows,values):assert str(r['sequence'])==v['sequence'] and int(r['receiver_timestamp_us'])==v['timestamp_us']
    zero=np.array([v['zero'] for v in values],bool);coop=np.array([v['coop']>0 for v in values],bool)
    off=np.asarray([v['offset'] for v in values],float)
    return dict(all=np.ones(len(rows),bool),zero=zero,colabelled=coop,nonzero=~zero),off

def fill(rows,off):
    out=off.copy();smooth=off.copy();span=np.zeros(len(rows));supported=np.zeros(len(rows),bool)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[str(r['sequence'])].append(i)
    for ids in groups.values():
        ts=np.array([int(rows[i]['receiver_timestamp_us']) for i in ids]);t,first=np.unique(ts,return_index=True)
        v=off[np.asarray(ids)[first]];ok=np.any(v!=0,axis=1);f=v.copy()
        if ok.any():
            f=np.column_stack([np.interp(t.astype(float),t[ok].astype(float),v[ok,k]) for k in (0,1)])
        ma=np.column_stack([np.convolve(np.pad(f[:,k],2,mode='edge'),np.ones(5)/5,mode='valid') for k in (0,1)])
        # Gap duration includes one nominal message slot, so a singleton is .1s.
        durations=np.zeros(len(t));support=np.zeros(len(t),bool);begin=None
        for j in range(len(t)+1):
            if j<len(t) and not ok[j]:
                if begin is None:begin=j
            elif begin is not None:
                dt=float(np.median(np.diff(t)))/1e6 if len(t)>1 else .1
                durations[begin:j]=(t[j-1]-t[begin])/1e6+dt
                support[begin:j]=bool(ok.any());begin=None
        pos=np.searchsorted(t,ts);out[ids]=f[pos];smooth[ids]=ma[pos];span[ids]=durations[pos];supported[ids]=support[pos]
    return out,smooth,span,supported

def packed(m,p,name,th):
    return dict(sequence=m['sequence'].astype(str),wait=m['wait_ms'],
        gt=np.array([common_mask(m['gt_boxes'][int(a):int(b)]).sum() for a,b in zip(m['gt_offsets'][:-1],m['gt_offsets'][1:])]),
        offset=p[name+'__offset'],score=p[name+'__score'],tp=p[name+'__tp'+str(th)])

def subset(d,mask):
    ids=np.flatnonzero(mask);pieces=[np.arange(int(d['offset'][i]),int(d['offset'][i+1])) for i in ids]
    take=np.concatenate(pieces) if pieces else np.array([],int)
    return dict(sequence=d['sequence'][ids],wait=d['wait'][ids],gt=d['gt'][ids],
        offset=np.r_[0,np.cumsum([len(x) for x in pieces])],score=d['score'][take],tp=d['tp'][take])

def statistic(d,mask,review=False):
    seqs=sorted(set(d['sequence'].astype(str)));rng=np.random.Generator(np.random.PCG64(2026110500))
    draw=rng.integers(0,len(seqs),(2000,len(seqs)));counts=np.array([np.bincount(x,minlength=len(seqs)) for x in draw],np.int16)
    sub=subset(d,mask);points=[];dist=[];rows=[]
    for w in WAITS:
        x=independent.panel(sub,w,seqs)
        pt=float(100*(independent.empirical(*x,np.ones((1,len(seqs)),np.int16))[0] if review else voc_ap(x[1],x[0],int(x[3].sum()))))
        dd=100*(independent.empirical(*x,counts) if review else weighted_ap_batch(*x,counts,batch_size=4))
        points.append(pt);dist.append(dd);ci=np.quantile(dd,[.025,.975])
        rows.append(dict(wait_ms=w,AP=pt,CI95=ci.tolist(),GT=int(x[3].sum()),conditions=int(np.sum(sub['wait']==w)),sequences=len(seqs),replicates=2000))
    dd=np.mean(dist,axis=0);rows.append(dict(wait_ms='four_wait_mean',AP=float(np.mean(points)),CI95=np.quantile(dd,[.025,.975]).tolist(),replicates=2000,sequences=len(seqs)))
    return rows,np.vstack(dist)

def save_stats(dest,m,rows,p,name,tag,review=False,gold=None):
    pops,_=annotations(rows);allrows=[];draws={}
    for th in (50,70):
        for pop in POPS:
            rr,dd=statistic(packed(m,p,name,th),pops[pop],review)
            allrows.extend(dict(r,population=pop,threshold=th,method=tag) for r in rr);draws[pop+'_AP'+str(th)]=dd
    if review:
        assert len(gold)==len(allrows)
        for a,b in zip(allrows,gold):
            assert (a['population'],a['threshold'],a['wait_ms'])==(b['population'],b['threshold'],b['wait_ms'])
            assert abs(a['AP']-b['AP'])<1e-8 and np.max(np.abs(np.array(a['CI95'])-b['CI95']))<1e-8,(a,b)
    atomic(dest/(tag+'_statistics.json'),allrows)
    if not review:np.savez_compressed(dest/(tag+'_draws.npz'),**draws)
    return allrows

def codes(m,rows,role):
    from run_peer07_spd_gt_source_split_ap_v1_20261031 import map_provenance
    mm=dict(m,dataset_index=np.arange(len(rows)))
    c,identity=map_provenance(mm,rows,role,ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label')
    assert identity['unmatched']==identity['ambiguous']==0,identity
    return c,identity
