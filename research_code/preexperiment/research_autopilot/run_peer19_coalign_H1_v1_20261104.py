"""R3: three fixed PUBLIC fusion weights, H1 input only; no new training.

Join H1 and CoAlign conditions by sequence/timestamp/wait, not row ordinal.
Existing graph/K1 predictions are references only, never rerun. Each new seed
has a 64-condition profile and same-environment first-case replay per wait.
"""
import argparse,os,time,traceback
from datetime import datetime
import numpy as np
from peer19_v1_20261104 import *
import run_peer14_coalign_reference_completion_v1_20261102 as reference
import run_peer13_coalign_seed_confirmation_v1_20261102 as confirm
OUT=BASE/'coalign_H1'
RECIPE=BASE/'recipe_v3_interface_recovery_v2/strict60'
SEEDS=(20261102,20261103,20261104)

def weight(seed):
    return confirm.C15/'coalign_full_v1/fusion_fit/final.pth' if seed==SEEDS[0] else confirm.CYCLE/'coalign_confirmation_v1'/('seed'+str(seed))/'public_fit/final.pth'

def old_outputs(seed):
    if seed==SEEDS[0]:
        p=confirm.C15/'coalign_full_v1/adapted_evaluation';return p,p
    p=confirm.CYCLE/'coalign_confirmation_v1'/('seed'+str(seed))
    return p/'public_reference/evaluation',p/'public_graph/evaluation'

def variants(public,row):
    key=str(row['sequence']),int(row['receiver_timestamp_us']),int(row['wait_ms'])
    v=[x.copy() for x in public]
    if len(v)>1:v[1][:2,3]+=H1[key]
    return dict(H1_INPUT=v)

def namespace(seed):
    ns=reference.build('public','slow')
    ns.update(OUTPUT=OUT/('seed'+str(seed))/'evaluation',ARMS=['H1_INPUT'],
        FUSION_WEIGHT=weight(seed),make_variants=variants,
        REFERENCE_CONTRACT=BASE/'coalign_H1/contract.json')
    ns['OUTPUT'].parent.mkdir(parents=True,exist_ok=True)
    return ns

def prepare():
    register('coalign_H1',__file__,__doc__,dict(GPU_forwards=3*(13264+64+8),updates=0,
        GPU='local3090 index1 sole chain',CPU_threads=2,wall_h='bounded until11-06 06:00',disk_GiB='<1 compact native predictions; sole GT reused'),
        input_recipe=ev(BASE/'recipe_strict60_interface_recovery_v2/contract.json'),
        frozen_evaluator=ev(confirm.old.__file__),adapter=ev(reference.__file__),
        weights={str(s):ev(weight(s)) for s in SEEDS},
        references={str(s):[ev(p/'terminal.json') for p in old_outputs(s)] for s in SEEDS},
        deadline='2026-11-06T06:00:00+08:00',same_NMS_native_readout=True,H1_heading_not_applied_to_feature_inputs=True)
    register('coalign_H1_independent',__file__,'Independently analytic HWL/overlap/target-first AP and 2000 sequence bootstrap on H1 vs saved K1/graph, common and native ROI. Shared detection only.',dict(GPU_forwards=0,CPU_threads=1,disk_MiB='<60 compact match/CI arrays',wall_h='bounded36h'),producer=ev(OUT/'contract.json'))

def run(gpu):
    global H1
    begin=time.monotonic();record('coalign_H1',dict(status='waiting_H1_strict60_actual_state',resource_owner=False,owner=None))
    try:
        wait(RECIPE/'val21_state_ready.json',30);ready=load(RECIPE/'val21_state_ready.json')
        rows=confirm.old.conditions(load(confirm.old.BASE/'contract.json'),'val21',np)
        keys=[(str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms'])) for r in rows]
        hkeys=[(str(s),int(t),int(w)) for s,t,w in ready['keys']]
        assert len(keys)==len(set(keys))==13264 and len(hkeys)==len(set(hkeys))==13264 and set(keys)==set(hkeys)
        with np.load(RECIPE/'val21_state.npz') as z:d=z['delta'].copy()
        assert len(d)==len(hkeys) and np.isfinite(d).all();H1=dict(zip(hkeys,d))
        atomic(OUT/'H1_input_identity.json',dict(created_at=now(),recipe=ev(RECIPE/'val21_state_ready.json'),state=ev(RECIPE/'val21_state.npz'),keys_exact_population=True,key_join_not_ordinal=True,q=ready['q'],rows=len(rows)))
        # CPU source imports must not hide either physical GPU in this worker.
        os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        for seed in SEEDS:
            assert datetime.now().astimezone().isoformat()<'2026-11-06T06:00:00+08:00'
            record('coalign_H1',dict(status='running_profile_then_H1_input',seed=seed,resource_owner=True,GPU=gpu))
            ns=namespace(seed);ns['evaluate'](gpu,True,True)
            profile=ns['OUTPUT'].with_name('evaluation_profile')
            for p in profile.glob('replay_*.json'):
                r=load(p);assert r['shape_equal'] and r['max_box_error']==0 and r['max_score_error']==0,(p,r)
            ns['evaluate'](gpu,True,False)
            for p in ns['OUTPUT'].glob('replay_*.json'):
                r=load(p);assert r['shape_equal'] and r['max_box_error']==0 and r['max_score_error']==0,(p,r)
            atomic(OUT/('seed'+str(seed))/'terminal.json',dict(status='complete_light_checked',created_at=now(),native=ev(ns['OUTPUT']/'terminal.json'),weight=ev(weight(seed)),no_training=True))
        atomic(OUT/'terminal.json',dict(status='complete_light_checked',created_at=now(),seeds=list(SEEDS),wall_s=time.monotonic()-begin,independent_pending=True))
        record('coalign_H1',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()))
        record('coalign_H1',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise

def review():
    from collections import defaultdict
    from review_peer10_cached_hwl_independent_v1_20261101 import polygons,overlap_matrix,flags,metric
    import review_peer18_final_native_statistics_v1_20261104 as stats
    from object_history_selector_core_v1_20261021 import common_mask
    dest=BASE/'coalign_H1_independent';done=[];boots=[];begin=time.monotonic()
    try:
        for seed in SEEDS:
            wait(OUT/('seed'+str(seed))/'terminal.json',36)
            new=OUT/('seed'+str(seed))/'evaluation';k1,graph=old_outputs(seed)
            record('coalign_H1_independent',dict(status='running_independent_native_geometry_AP_CI',seed=seed,resource_owner=True))
            methods=dict(H1=(new,'H1_INPUT'),K1=(k1,'K1_INPUT'),GRAPH=(graph,'COALIGN_FULL'))
            packs={}
            for tag,(src,arm) in methods.items():
                recs=load(src/'records.json');keys=[(str(r['sequence']),int(r['decision_us']),int(r['wait_ms'])) for r in recs]
                if tag=='H1':population=keys
                assert keys==population
                for roi in ('native','common'):
                    for th in (50,70):packs[tag,roi,th]=dict(sequence=[],wait=[],gt=[],offset=[0],score=[],tp=[])
                for w in (0,100,200,400):
                    rs=[r for r in recs if r['wait_ms']==w]
                    with np.load(src/f'wait{w:03d}.npz') as z,np.load(ROOT/rs[0]['GT_shard']) as g:
                        for r in rs:
                            j=r['shard_case'];b=z[f'c{j}_{arm}_box3d_lidar'];score=z[f'c{j}_{arm}_scores'];cl=z[f'c{j}_{arm}_label_preds'];gt=g[f"c{r['GT_case']}_gt"]
                            for roi in ('native','common'):
                                keep=(cl==0)&(abs(b[:,0])<=100.8)&(abs(b[:,1])<=40);kg=gt[:,7]==0
                                if roi=='common':keep &= common_mask(b);kg &= common_mask(gt[:,:7])
                                ss=score[keep];order=np.argsort(-ss);ss=ss[order].astype(float)
                                ov=overlap_matrix(polygons(b[keep][order]),polygons(gt[kg,:7]))
                                for th in (50,70):
                                    f,n=flags(ov,np.zeros(ov.shape[1],np.int8),'all',th/100);p=packs[tag,roi,th]
                                    p['sequence'].append(r['sequence']);p['wait'].append(w);p['gt'].append(n);p['score'].extend(ss);p['tp'].extend(f[0]);p['offset'].append(len(p['score']))
                    for roi in ('native','common'):
                        for th in (50,70):
                            p=packs[tag,roi,th];ids=np.where(np.asarray(p['wait'])==w)[0];lo,hi=p['offset'][ids[0]],p['offset'][ids[-1]+1]
                            value=metric(np.asarray(p['tp'][lo:hi]),1-np.asarray(p['tp'][lo:hi]),np.asarray(p['score'][lo:hi]),sum(p['gt'][i] for i in ids))
                            row=dict(seed=seed,method=tag,roi=roi,threshold=th,wait_ms=w,AP=value)
                            if roi=='native':
                                gold=next(r for r in load(src/'panels.json') if r['wait_ms']==w)['metrics'][arm]['global_score_sort']['ap'+str(th)]*100
                                row['producer_error_pp']=value-gold;assert abs(value-gold)<1e-8,row
                            done.append(row);atomic(dest/'AP.json',done)
            seqs=sorted(set(packs['H1','native',70]['sequence']));draw=np.random.Generator(np.random.PCG64(20261103)).integers(0,len(seqs),(2000,len(seqs)));counts=np.asarray([np.bincount(x,minlength=len(seqs)) for x in draw],np.int16)
            for roi in ('native','common'):
                for th in (50,70):
                    for ref in ('K1','GRAPH'):
                        ds=[];ps=[]
                        for w in (0,100,200,400):
                            a=stats.panel(packs['H1',roi,th],w,seqs);b=stats.panel(packs[ref,roi,th],w,seqs);assert np.array_equal(a[3],b[3])
                            d=100*(stats.empirical(*a,counts)-stats.empirical(*b,counts));pt=float(100*(stats.empirical(*a,np.ones((1,len(seqs)),np.int16))[0]-stats.empirical(*b,np.ones((1,len(seqs)),np.int16))[0]));ds.append(d);ps.append(pt)
                        for w,pt,d in zip((0,100,200,400,'four_wait_mean'),ps+[float(np.mean(ps))],ds+[np.mean(ds,axis=0)]):
                            ci=np.percentile(d,[2.5,97.5]);boots.append(dict(seed=seed,roi=roi,threshold=th,comparison='H1_minus_'+ref,wait_ms=w,delta_pp=pt,CI95_low=float(ci[0]),CI95_high=float(ci[1])))
            atomic(dest/'bootstrap.json',boots)
        atomic(dest/'terminal.json',dict(status='complete_independently_reviewed',created_at=now(),AP=ev(dest/'AP.json'),bootstrap=ev(dest/'bootstrap.json'),independent_detection=False,wall_s=time.monotonic()-begin))
        record('coalign_H1_independent',dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(dest/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('coalign_H1_independent',dict(status='failed_retained',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','review']);p.add_argument('--gpu',type=int,default=1);a=p.parse_args();run(a.gpu) if a.mode=='run' else globals()[a.mode]()
