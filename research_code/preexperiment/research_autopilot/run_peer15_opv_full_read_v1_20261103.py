"""Full OPV test CPU consumer, all CAVs/frames, exact existing heading+U."""
import argparse, hashlib, subprocess, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from peer15_v1_20261103 import ROOT,BASE,C21,REMOTE_ROOT,atomic,ev,load,now,register,record
import run_peer14_opv2v_async_heading_U_v1_20261103 as prior
import object_history_selector_core_v1_20261021 as core
import run_shared_offset_optional_formal_v1_20261026 as formal
import run_v2v4real_current_q005_heading_anchor_gate_factorial_v1_20261030 as geometry
import run_crosscheckpoint_temporal_q005_task_refit_v1_20261029 as refit
from audit_common_offset_peer_v1_20261026 import pairs
from run_peer07_spd_gt_source_split_ap_v1_20261031 import iou32
from e1_receiver_value_core_v1_20261020 import greedy
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap

OUT=BASE/'opv_full_read';INPUT=BASE/'opv_full_test_v2/producer'
METHODS=('stale_same_U','CoMET_heading_U','oracle_clean_heading_U')

def prepare():
    register('opv_full_read',Path(__file__).resolve(),'Complete2170-frame all-CAV test can differ from bounded275-frame two-CAV test. Retain structured and iid results for all methods and waits; no AP-based selection, no inherited-label flip claim. Cold startup and no-packet fallback remain in the full population.',dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='1-6 after full sensor owner, finite18h wait',disk_MiB='<150 unique packed output boxes/scores and TP ledger; sensor/GT/cache inputs referenced'),detector_contract=ev(BASE/'opv_full_test_v2/contract.json'),canonical_census=ev(BASE/'opv_full_test_v2/canonical_census.json'),frozen_noise=ev(prior.old.BASE/'noise_frozen.json'),head={p:ev(prior.HEAD/p) for p in ('direct_u.pth','training_schedule.npz','source_maps.json')},protect=ev(prior.PROTECT),methods=list(METHODS),waits_ms=[0,400],protocol='P1 original simulation GT, fixed common XY population, no post-shift ROI/NMS, original39D+matched614b7e2c U and ordered protected prefix. Every source CAV has its own causal KF; ego retained once, all remote candidates combined and ranked jointly.',timing='10Hz ordinal common timestamp timeline as prior, synthetic fixed delay, not recorded SPD arrival; source history one earlier, no future. Frames0..3 at400ms explicitly ego-only fallback, identical population all methods.',noise_rule='Exact peer14 SPD-fit frozen distribution and seed; deterministic SHA-derived sequence/sender random streams, no test fit. Structured constant+AR1+jitter, iid XY.2m/yaw.2deg, source-time noisy pose for each sender. Oracle all clean poses.',collector='This worker uniquely collects peer15 full-test producer sensor shards, once after terminal; no other source collector or root mirror evaluation copy.')

def collect():
    start=time.monotonic();remote=str(REMOTE_ROOT/INPUT.relative_to(ROOT))
    while True:
        ok=subprocess.run(['ssh','remote-gpu','test','-f',remote+'/terminal.json'],capture_output=True).returncode==0
        if ok:break
        assert time.monotonic()-start<18*3600,'finite18h OPV detector wait'
        atomic(OUT/'waiting.json',dict(created_at=now(),status='waiting_full_test_detector',elapsed_s=time.monotonic()-start));time.sleep(30)
    r=subprocess.run(['rsync','-a','--ignore-existing','--stats','remote-gpu:'+remote+'/',str(INPUT)+'/'],capture_output=True,text=True);assert r.returncode==0,r.stderr
    atomic(OUT/'collection_receipt.json',dict(created_at=now(),source=remote,stdout=r.stdout,stderr=r.stderr,no_delete=True,no_overwrite=True,sole_source_collector=True))
    terminal=load(INPUT/'terminal.json');assert terminal['status']=='complete_light_checked' and terminal['frames']==terminal['planned_frames']==2170,terminal
    return load(INPUT/'records.json')

def sensor(ref):
    p=ROOT/ref['shard']['path'];assert ev(p)==ref['shard']
    with np.load(p) as z:return dict(pred_boxes=z[ref['boxes_key']].copy(),pred_scores=z[ref['scores_key']].copy(),pred_labels=np.zeros(len(z[ref['boxes_key']]),np.int32))

def noises(rows,arrays,family):
    cfg=load(prior.old.BASE/'noise_frozen.json')['parameters'];out={};states={};rngs={}
    for i,row in enumerate(rows):
        for sender in range(1,len(row['sensors'])):
            key=row['sequence']+'/'+row['sensors'][sender]['agent']
            if key not in rngs:
                seed=int.from_bytes(hashlib.sha256((str(cfg['seed'])+'/'+family+'/'+key).encode()).digest()[:8],'little');rngs[key]=np.random.default_rng(seed)
            rng=rngs[key]
            if family=='SPD_structured':
                if key not in states:states[key]=(np.array(cfg['sequence_means'])[rng.integers(len(cfg['sequence_means']))],rng.multivariate_normal(np.zeros(2),cfg['slow_initial_cov']))
                center,slow=states[key];slow=cfg['rho']*slow+rng.multivariate_normal(np.zeros(2),cfg['innovation_cov']);states[key]=(center,slow)
                t=center+slow+rng.multivariate_normal(np.zeros(2),cfg['jitter_cov']);theta=0.
            else:t=rng.normal(0,cfg['iid_sigma_translation_m'],2);theta=float(rng.normal(0,np.deg2rad(cfg['iid_sigma_yaw_degrees'])))
            D=np.eye(4);D[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]];D[:2,3]=t;out[i,sender]=D
    return out

def material(rows,arrays,detections,noise,wait,clean=False):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['sequence']].append(i)
    arr=defaultdict(list);offset=[0];metas=[];counts=[];measures=[]
    protect=float(load(prior.PROTECT)['protect_threshold'])
    # Sender-specific full conditions are subsequently joined by receiver index.
    for seq,indices in sorted(groups.items()):
        n=len(rows[indices[0]]['sensors']);assert all(len(rows[i]['sensors'])==n for i in indices)
        for sender in range(1,n):
            sources=[]
            for pos,i in enumerate(indices):
                a=arrays[i];D=np.eye(4) if clean else noise[i,sender]
                sources.append(dict(sequence=seq,frame=rows[i]['frame'],timestamp_us=pos*100000,pose=a['ego_pose_world']@D@a['remote_to_ego'][sender],raw=detections[i][sender]))
            for pos,i in enumerate(indices):
                s=pos-wait//100;source=sources[s] if s>=0 else None;previous=sources[s-1] if s>0 else None
                ego=dict(detections[i][0],gt_boxes=np.empty((0,7),np.float32),gt_labels=np.empty(0,np.int32))
                pool=core.make_condition(ego,source,previous,arrays[i]['ego_pose_world'],pos*100000,wait,protect)
                assert len(pool['gt_boxes'])==0 and pool['features'].shape[1]==39
                ei,si,ci=formal.identities(pool);chosen=np.r_[ei,ci];stale=np.r_[ei,si]
                src=np.r_[np.zeros(len(ei),np.int8),np.ones(len(si),np.int8)];protected=pool['is_protected'][stale]
                anchor=np.flatnonzero((src==0)&(protected==1));remote=np.flatnonzero(src==1);cv=pool['boxes'][chosen]
                ri,ai=pairs(cv[remote],cv[anchor],2.5,True);res=cv[anchor[ai],:2].astype(np.float64)-cv[remote[ri],:2]
                counts.append(len(ri));measures.append(res.mean(0) if len(res) else np.zeros(2))
                for key,value in dict(cv_boxes=cv,stale_boxes=pool['boxes'][stale],cv_features=pool['features'][chosen],stale_features=pool['features'][stale],raw_scores=pool['raw_scores'][stale],source=src,protected=protected).items():arr[key].append(value)
                offset.append(offset[-1]+len(chosen));metas.append(dict(receiver_index=i,sender=sender,sequence=seq+'/'+rows[i]['sensors'][sender]['agent'],step=pos,wait_ms=wait,source_timestamp_us=None if source is None else source['timestamp_us']))
    m={k:np.concatenate(v) for k,v in arr.items()};m.update(condition_offsets=np.asarray(offset,np.int64),sequence=np.asarray([r['sequence'] for r in metas],dtype='U128'),wait_ms=np.full(len(metas),wait),dataset_index=np.arange(len(metas)),receiver_timestamp_us=np.asarray([r['step']*100000 for r in metas]))
    f=np.zeros((len(counts),15));f[:,14]=np.log1p(counts);f[:,5:7]=measures
    delta=geometry.q005_frame(f,m,False)['q005_delta'] if not clean else np.zeros((len(counts),2));boxes,features=geometry.heading_geometry(m,delta)
    return m,boxes,features,delta,metas,np.asarray(counts)

def run():
    start=time.monotonic();cpu=time.process_time();c=load(OUT/'contract.json');assert c['script']==ev(__file__)
    record('opv_full_read',dict(status='waiting_unique_full_test_source_collector'))
    try:
        rows=collect();assert len(rows)==2170
        arrays=[];detections=[]
        for row in rows:
            p=ROOT/row['geometry']['path'];assert ev(p)==row['geometry']
            with np.load(p) as z:arrays.append({k:z[k].copy() for k in z.files})
            assert len(row['object_ids'])==len(set(row['object_ids']))==len(arrays[-1]['gt_hwl'])
            detections.append([sensor(ref) for ref in row['sensors']])
        panels=[]
        for family in ('SPD_structured','iid_02m_02deg'):
            noise=noises(rows,arrays,family)
            np.savez_compressed(OUT/(family+'_noise.npz'),receiver_sender=np.asarray(list(noise),np.int64),matrix=np.stack(list(noise.values())))
            for wait in (0,400):
                m,b,f,d,meta,n=material(rows,arrays,detections,noise,wait)
                cm,cb,cf,_,cmeta,_=material(rows,arrays,detections,noise,wait,True);assert meta==cmeta
                methods={METHODS[0]:(m,m['stale_boxes'],m['stale_features']),METHODS[1]:(m,b,f),METHODS[2]:(cm,cb,cf)}
                scores={k:refit.score_with_head(mm,ff,prior.HEAD) for k,(mm,bb,ff) in methods.items()}
                assert np.array_equal(scores[METHODS[1]],refit.score_with_head(m,f,prior.HEAD))
                by_receiver=defaultdict(list)
                for j,r in enumerate(meta):by_receiver[r['receiver_index']].append(j)
                saved={k:defaultdict(list) for k in METHODS};stats=defaultdict(lambda:dict(score=[],tp50=[],tp70=[],gt=0));gtcounts=[]
                for i,row in enumerate(rows):
                    gt=arrays[i]['gt_hwl'];gt=gt[core.common_mask(gt)];gtcounts.append(len(gt))
                    for k,(mm,bb,ff) in methods.items():
                        parts=[];rankparts=[];sourceparts=[];protectedparts=[];rawparts=[]
                        for jj,j in enumerate(by_receiver[i]):
                            lo,hi=map(int,mm['condition_offsets'][j:j+2]);idx=np.arange(lo,hi)
                            if jj:idx=idx[mm['source'][idx]==1]
                            parts.append(bb[idx]);rankparts.append(scores[k][idx]);sourceparts.append(mm['source'][idx]);protectedparts.append(mm['protected'][idx]);rawparts.append(mm['raw_scores'][idx])
                        boxes=np.concatenate(parts);raw=np.concatenate(rawparts);src=np.concatenate(sourceparts);protect=np.concatenate(protectedparts);u=np.concatenate(rankparts)
                        chosen,rank=formal.ordered(raw,src,protect,u,None);selected=boxes[chosen];overlap=iou32(selected,gt);tp={t:(greedy(overlap,np.arange(len(selected)),t/100)>=0).astype(np.int8) for t in (50,70)}
                        st=stats[k];st['score'].append(rank);st['gt']+=len(gt)
                        for t in (50,70):st['tp'+str(t)].append(tp[t])
                        v=saved[k];v['boxes'].append(selected);v['scores'].append(rank);v['source'].append(src[chosen]);v['length'].append(len(chosen))
                        for t in (50,70):v['tp'+str(t)].append(tp[t])
                    if i%128==0:atomic(OUT/'status.json',dict(created_at=now(),status='full_test_all_CAV_cpu_readout',family=family,wait_ms=wait,completed_frames=i+1,total_frames=len(rows),wall_s=time.monotonic()-start))
                packed=dict(sequence=np.asarray([r['sequence'] for r in rows]),frame=np.asarray([r['frame'] for r in rows]),GT_count=np.asarray(gtcounts),delta=d,anchor_count=n)
                for k,v in saved.items():
                    for field in ('boxes','scores','source','tp50','tp70'):packed[k+'__'+field]=np.concatenate(v[field])
                    packed[k+'__offset']=np.r_[0,np.cumsum(v['length'])]
                path=OUT/(family+'_wait'+str(wait)+'_predictions.npz');np.savez_compressed(path,**packed)
                atomic(OUT/(family+'_wait'+str(wait)+'_sender_conditions.json'),meta)
                for k,v in stats.items():
                    p=dict(noise=family,wait_ms=wait,method=k,frames=len(rows),sensor_frames=sum(len(r['sensors']) for r in rows),scope='full2170 standard test receiver frames, every available CAV, P1 simulation',predictions=ev(path))
                    for t in (50,70):
                        tp=np.concatenate(v['tp'+str(t)]);metric=voc_ap(tp.tolist(),(1-tp).tolist(),np.concatenate(v['score']).tolist(),v['gt'],True);p['AP'+str(t)]=100*metric['ap'];p['TP'+str(t)]=metric['tp'];p['FP'+str(t)]=metric['fp'];p['GT']=v['gt']
                    panels.append(p)
                atomic(OUT/'panels.json',panels)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',panels=ev(OUT/'panels.json'),frames=2170,all_CAV=True,GPU_forwards=0,updates=0,independent_review_pending=True,no_test_selection=True,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,no_model_wakeup=True))
        record('opv_full_read',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start));record('opv_full_read',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();globals()[a.mode]()
