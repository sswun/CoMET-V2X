"""R6d fixed-population label-only sensitivity, never inference or main GT.

Only inf-only GT at exact-zero frames translated by noncausal filled offset.
Three preregistered gap caps1s/3s/all; one nominal sample slot included in run
duration. Unsupported all-zero sequences stay unchanged and are counted.
Original official ROI population and original predictions/ranks frozen.
"""
import argparse,time,traceback
import numpy as np
from peer20_population_core_v1_20261105 import *
import run_peer20_population_v1_20261105 as core
from e1_receiver_value_core_v1_20261020 import iou32,greedy
import review_traf_v2xseq_adapted_fixed60_raw_20261006 as independent_geometry
from review_peer16_E4_E7_AP_v1_20261103 import geometry

TIERS=(('gap1s',1.),('gap3s',3.),('all_gaps',float('inf')))

def prepare():
    for d in h1.recipe.DOMAINS:
        for review in (False,True):
            register('R6d_'+d+('_independent' if review else ''),__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
                replicates=2000,disk_MiB='<120 compressed changed TP only, prediction/GT refs',wall_h='finite full analysis; no deadline'),
                domain=d,producer_contract=ev(BASE/('R6b_'+d)/'contract.json'),tiers=['<=1s','<=3s','all'],
                no_selection=True,ROI='Original official GT population frozen; do not remask shifted GT or move/rerank predictions',
                remaining='Native/official-late/module sensitivity requires separate external saved-box consumer; not claimed covered by core-only job')

def changed_truth(m,rows,role,cap):
    pops,off=annotations(rows);filled,_,span,support=fill(rows,off);c,identity=codes(m,rows,role)
    use=pops['zero']&support&(span<=cap);gt=m['gt_boxes'].copy();count=0
    for i in np.flatnonzero(use):
        ga,gb=map(int,m['gt_offsets'][i:i+2]);idx=ga+np.flatnonzero(c[ga:gb]==2);gt[idx,:2]+=filled[i].astype(np.float32);count+=len(idx)
    assert np.array_equal(gt[c!=2],m['gt_boxes'][c!=2])
    return gt,dict(changed_gt_occurrences=count,changed_conditions=int(use.sum()),unsupported_zero_conditions=int((pops['zero']&~support).sum()),
        unchanged_other_sources=True,source_mapping=identity,definition='Run first-to-last plus median sampling slot; boundary holes use nearest observed endpoint, reported label-assisted only')

def rematch(m,p,label,b,gt,review=False):
    result={}
    for th in (50,70):
        tp=[]
        for i in range(len(m['wait_ms'])):
            lo,hi=map(int,p[label+'__offset'][i:i+2]);ids=p[label+'__candidate'][lo:hi].astype(int);score=p[label+'__score'][lo:hi]
            ga,gb=map(int,m['gt_offsets'][i:i+2]);target=gt[ga:gb][common_mask(m['gt_boxes'][ga:gb])]
            if review:
                _,hit,_,n=independent_geometry.match_frame(b[ids],score,np.zeros(len(ids)),target,np.zeros(len(target)),th/100)
                assert n==len(target);tp.extend(hit)
            else:tp.extend((greedy(iou32(b[ids],target),np.arange(len(ids)),th/100)>=0).astype(np.int8))
        result[label+'__tp'+str(th)]=np.asarray(tp,np.int8)
    return result

def run(domain,review=False):
    name='R6d_'+domain+('_independent' if review else '');dest=BASE/name;src=BASE/('R6d_'+domain);source=BASE/('R6b_'+domain)
    try:
        wait(source/'terminal.json')
        if review:wait(src/'terminal.json')
        record(name,dict(status='running_auxiliary_GT_independent' if review else 'running_label_only_auxiliary_GT',resource_owner=True))
        dataset,_,_=data(domain);done=[]
        for role in ROLES:
            m,rows=dataset[role];targets={tier:changed_truth(m,rows,role,cap) for tier,cap in TIERS}
            for tier,(gt,identity) in targets.items():atomic(dest/(role+'_'+tier+'_GT_identity.json'),identity)
            for method in core.METHODS:
                with np.load(source/(role+'_'+method+'_state.npz')) as z:d=z['delta'].copy()
                b=geometry(m,d) if review else h1.recipe.geometry(m,d)[0]
                if method=='EGO':b[m['source']==1,:2]=1e6
                for seed in h1.recipe.SEEDS:
                    label=method+'_seed'+str(seed);tag=role+'_'+label
                    path=source/(tag+'_predictions.npz')
                    with np.load(path) as z:p={k:z[k] for k in z.files}
                    for tier,(gt,identity) in targets.items():
                        outcome=rematch(m,p,label,b,gt,review);key=tag+'_'+tier;targetpath=src/(key+'_TP.npz')
                        if review:
                            with np.load(targetpath) as z:assert all(np.array_equal(v,z[k]) for k,v in outcome.items())
                        else:np.savez_compressed(targetpath,**outcome)
                        pp=dict(p,**outcome)
                        save_stats(dest,m,rows,pp,label,key,review,load(src/(key+'_statistics.json')) if review else None)
                        done.append(dict(role=role,method=method,seed=seed,tier=tier,prediction=ev(path),modified_TP=ev(targetpath)))
                        atomic(dest/'progress.json',dict(created_at=now(),completed=len(done),last=key))
        atomic(dest/'identity.json',dict(official_GT_not_modified=True,fixed_original_ROI=True,reference_only=True,completed=done))
        status='complete_independently_reviewed' if review else 'complete_light_checked'
        atomic(dest/'terminal.json',dict(created_at=now(),status=status,completed=len(done),native_module_coverage_not_implied=True))
        record(name,dict(status=status,terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review'));p.add_argument('--domain',choices=h1.recipe.DOMAINS);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.domain,a.mode=='review')
