"""Complete fixed K1 heading+U on existing bounded OPV2V detections.

Nonzero history is actually constructed. No GT enters pools, associations,
features, KF or readout. Simulation labels are read only after predictions.
"""
import argparse,os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from collections import defaultdict
import numpy as np
from peer14_cycle09_v1_20261103 import ROOT,AUDIT,CYCLE,C21,atomic,ev,load,now,register,record
import run_peer14_opv2v_controlled_read_v1_20261102 as old
import object_history_selector_core_v1_20261021 as core
import run_shared_offset_optional_formal_v1_20261026 as formal
import run_v2v4real_current_q005_heading_anchor_gate_factorial_v1_20261030 as geometry
import run_crosscheckpoint_temporal_q005_task_refit_v1_20261029 as refit
BASE=CYCLE/'opv2v_async_heading_U_v1'
HEAD=ROOT/load(AUDIT/'contract_v1.json')['inputs']['matched_q005_head']['path']
HEAD=HEAD.parent
PROTECT=ROOT/'artifacts/research_autopilot/status_audit_20261021_1200/object_history_selector_v1/train_dataset/summary.json'
METHODS=('stale_same_U','K1_heading_U','oracle_clean_heading_U')

def prepare():
    BASE.mkdir(exist_ok=False)
    register('opv2v_async_heading_U',__file__,
        'Close the actual asynchronous full heading+matched39D U gap, not a repeat of synchronous estimator-only. Reuse identical400 detections and frozen noise from failed global flip; no retuning. K1 may or may not improve clean P1 and harm inherited P2; all waits/families/validate/test/source signs retained, no selection. Exact SPD-trained readout is transferred without OPV fitting; not full OPV benchmark or main-domain evidence.',
        dict(GPU_forwards=0,training_updates=0,CPU_threads=1,expected_wall_h='<2 bounded1100 conditions and three fixed readouts',new_disk_MiB='<80 unique compressed candidate/features/scores/GT and metrics; original sensor detections/noise referenced'),
        inputs=ev(old.INPUT/'records.json'),detector_terminal=ev(old.INPUT/'terminal.json'),noise_fit=ev(old.BASE/'noise_frozen.json'),noise_realizations={f:ev(old.BASE/'producer'/(f+'_state_and_noise.json')) for f in ('SPD_structured','iid_02m_02deg')},
        kernels={k:ev(p) for k,p in dict(history=core.__file__,materialization=formal.__file__,heading_q005=geometry.__file__,score=refit.__file__).items()},
        head={p:ev(HEAD/p) for p in ('direct_u.pth','training_schedule.npz','source_maps.json')},protect=ev(PROTECT),
        timing='Author BaseDataset time_delay_calculation explicitly states10Hz; ordinal adjacent sampled frame tokens differ2, so100ms per sample, not frame_token*100ms. Mocked simulation timeline explicitly not recorded SPD arrivals.',
        time_source=ev(ROOT/'third_party/research_baselines_20260930/CooperTrim/3D_Detection_OPV2V/opencood/data_utils/datasets/basedataset.py'),
        waits_ms=[0,400],population='Each original16-frame sequence: receiver indices5..15 for both waits; stale remote i-wait/100, previous remote one earlier. Same receiver/GT population for all methods and protocols.',
        input_noise='Reuse frozen D at source time: noisy source world pose=ego_world(source) @ D(source) @ remote_to_ego(source). History uses its own older noisy pose; no decision-time noise applied retroactively. Oracle uses clean poses.',
        labels='P1 original current sim GT; P2 only original remote-exclusive GT transformed by original D(current). Membership and original XY common ROI frozen before intervention; ego-visible GT unchanged.',
        inference='Original39D feature builder, causal Hungarian1+50dt gate,50m/s cap; protected threshold from original SPD release; exact q.05/r.6 scalar KF on CV anchor residuals, original heading projection, matched614b7e2c U, original ordered protected prefix. No GT associations or fields fabricated. No new post-shift ROI/NMS.',
        scientific_scope='bounded controlled asynchronous transfer diagnostic, not new estimator, not OPV full AP, not SPD official protocol')

def material(rows,arrays,noise,wait,clean=False):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['split']+'/'+r['sequence']].append(i)
    items=[];counts=[];measurements=[];arr=defaultdict(list);offsets=[0];metas=[]
    protect=float(load(PROTECT)['protect_threshold'])
    for seq,ids in sorted(groups.items()):
        assert len(ids)==16 and all(int(rows[b]['frame'])-int(rows[a]['frame'])==2 for a,b in zip(ids,ids[1:])),seq
        sources=[]
        for pos,index in enumerate(ids):
            a=arrays[index];D=np.eye(4) if clean else np.asarray(noise[index]['noise_left_SE2'])
            pose=a['ego_pose_world']@D@a['remote_to_ego']
            sources.append(dict(sequence=seq,frame=rows[index]['frame'],timestamp_us=pos*100000,pose=pose,raw=dict(pred_boxes=a['agent1_boxes'],pred_scores=a['agent1_scores'],pred_labels=np.zeros(len(a['agent1_boxes']),np.int32))))
        for pos in range(5,16):
            index=ids[pos];a=arrays[index];s=pos-wait//100;previous=sources[s-1] if s>0 else None
            assert sources[s]['timestamp_us']<=pos*100000 and (previous is None or previous['timestamp_us']<sources[s]['timestamp_us'])
            ego=dict(pred_boxes=a['agent0_boxes'],pred_scores=a['agent0_scores'],pred_labels=np.zeros(len(a['agent0_boxes']),np.int32),gt_boxes=np.empty((0,7),np.float32),gt_labels=np.empty(0,np.int32))
            pool=core.make_condition(ego,sources[s],previous,a['ego_pose_world'],pos*100000,wait,protect)
            assert pool['features'].shape[1]==39 and len(pool['gt_boxes'])==0
            ei,si,ci=formal.identities(pool);chosen=np.r_[ei,ci];stale=np.r_[ei,si]
            source=np.r_[np.zeros(len(ei),np.int8),np.ones(len(si),np.int8)];protected=pool['is_protected'][stale]
            anchor=np.flatnonzero((source==0)&(protected==1));remote=np.flatnonzero(source==1)
            cv=pool['boxes'][chosen];sb=pool['boxes'][stale]
            from audit_common_offset_peer_v1_20261026 import pairs
            ri,ai=pairs(cv[remote],cv[anchor],2.5,True);res=cv[anchor[ai],:2].astype(np.float64)-cv[remote[ri],:2]
            counts.append(len(ri));measurements.append(res.mean(0) if len(res) else np.zeros(2))
            for key,value in dict(cv_boxes=cv,stale_boxes=sb,cv_features=pool['features'][chosen],stale_features=pool['features'][stale],raw_scores=pool['raw_scores'][stale],source=source,protected=protected).items():arr[key].append(value)
            offsets.append(offsets[-1]+len(chosen));metas.append(dict(original_index=index,split=rows[index]['split'],sequence=seq,frame=rows[index]['frame'],wait_ms=wait,receiver_timestamp_us=pos*100000,source_timestamp_us=sources[s]['timestamp_us'],previous_timestamp_us=previous['timestamp_us'] if previous else None))
            items.append(dict(associations=int((pool['features'][chosen,32]>0).sum()),nonzero_motion=int((pool['features'][chosen,38]>0).sum()),remote_candidates=len(remote),anchor_pairs=len(ri)))
    m={k:np.concatenate(v) for k,v in arr.items()};m.update(condition_offsets=np.asarray(offsets,np.int64),sequence=np.asarray([r['sequence'] for r in metas],dtype='U96'),wait_ms=np.full(len(metas),wait,np.int16),dataset_index=np.arange(len(metas)),receiver_timestamp_us=np.asarray([r['receiver_timestamp_us'] for r in metas]))
    f=np.zeros((len(counts),15),np.float64);f[:,14]=np.log1p(counts);f[:,5:7]=measurements
    delta=geometry.q005_frame(f,m,False)['q005_delta'] if not clean else np.zeros((len(counts),2))
    boxes,features=geometry.heading_geometry(m,delta)
    return m,boxes,features,delta,metas,items

def run():
    out=BASE/'producer';out.mkdir(exist_ok=False);start=time.monotonic();record('opv2v_async_heading_U',dict(status='running_actual_history_heading_U'))
    c=load(CYCLE/'opv2v_async_heading_U_contract.json')
    try:
        assert ev(__file__)==c['script'] and ev(old.INPUT/'records.json')==c['inputs'] and ev(PROTECT)==c['protect']
        for ref in list(c['kernels'].values())+list(c['head'].values())+list(c['noise_realizations'].values()):assert ev(ROOT/ref['path'])==ref
        rows=load(old.INPUT/'records.json');arrays=[]
        for r in rows:
            p=ROOT/r['shard']['path'];assert ev(p)==r['shard']
            with np.load(p) as z:arrays.append({k:z[k].copy() for k in z.files})
        from run_peer07_spd_gt_source_split_ap_v1_20261031 import iou32
        from run_peer07_v2_strict400_source_split_ap_v1_20261031 import outcomes
        from peer07_source_ap_producer_utils_v1_20261031 import metric
        panels=[];checks=[]
        for family in ('SPD_structured','iid_02m_02deg'):
            noise=load(old.BASE/'producer'/(family+'_state_and_noise.json'))
            assert len(noise)==len(rows) and all((n['split'],n['sequence'],n['frame'])==(r['split'],r['sequence'],r['frame']) for n,r in zip(noise,rows))
            for wait in (0,400):
                m,b,f,d,metas,diagnostic=material(rows,arrays,noise,wait)
                clean,cb,cf,_,cm,_=material(rows,arrays,noise,wait,True);assert cm==metas
                # Oracle candidate pre-filter may differ under clean pose: reference only.
                vals={'stale_same_U':(m,m['stale_boxes'],m['stale_features']),'K1_heading_U':(m,b,f),'oracle_clean_heading_U':(clean,cb,cf)}
                scores={k:refit.score_with_head(mm,ff,HEAD) for k,(mm,bb,ff) in vals.items()}
                assert np.array_equal(scores['K1_heading_U'],refit.score_with_head(m,f,HEAD))
                sub=out/(family+'_wait'+str(wait));sub.mkdir()
                stats=defaultdict(lambda:dict(scores=[],tp=[],fp=[],gt=0));archive={};sel=defaultdict(list)
                for k,(mm,bb,ff) in vals.items():
                    archive[k+'_boxes']=bb;archive[k+'_features']=ff;archive[k+'_scores']=scores[k];archive[k+'_offsets']=mm['condition_offsets'];archive[k+'_raw']=mm['raw_scores'];archive[k+'_source']=mm['source'];archive[k+'_protected']=mm['protected']
                for i,meta in enumerate(metas):
                    original=meta['original_index'];a=arrays[original]
                    keep=core.common_mask(a['gt_hwl']);gt=a['gt_hwl'][keep];remote=a['remote_exclusive'][keep]
                    labels={'P1_simulation':gt.copy(),'P2_inherited':gt.copy()};labels['P2_inherited'][remote]=old.transform(gt[remote],np.asarray(noise[original]['noise_left_SE2']))
                    assert np.array_equal(labels['P1_simulation'][~remote],labels['P2_inherited'][~remote])
                    for k,(mm,bb,ff) in vals.items():
                        lo,hi=mm['condition_offsets'][i:i+2];ids,rank=formal.ordered(mm['raw_scores'][lo:hi],mm['source'][lo:hi],mm['protected'][lo:hi],scores[k][lo:hi],None)
                        sel[k+'_ids'].append((lo+ids).astype(np.int64));sel[k+'_rank'].append(rank)
                        for protocol,target in labels.items():
                            overlap=iou32(bb[lo:hi][ids],target)
                            for group,mask in [('all',np.ones(len(gt),bool)),('ego_visible',~remote),('remote_exclusive',remote)]:
                                for th in (.5,.7):
                                    tp,fp,ignore=outcomes(overlap,mask,th);v=stats[meta['split'],protocol,group,k,th];valid=ignore==0;v['scores'].append(rank[valid]);v['tp'].append(tp[valid]);v['fp'].append(fp[valid]);v['gt']+=int(mask.sum())
                    if i%32==0:atomic(out/'status.json',dict(created_at=now(),status='actual_async_heading_U_scoring',family=family,wait_ms=wait,conditions=i+1,total=len(metas)))
                archive.update({k:np.concatenate(v) for k,v in sel.items()});archive['q005_delta']=d;np.savez_compressed(sub/'prediction_and_features.npz',**archive);atomic(sub/'conditions.json',metas)
                for key,v in stats.items():
                    value=metric(np.concatenate(v['tp']),np.concatenate(v['fp']),np.concatenate(v['scores']),v['gt']);panels.append(dict(noise=family,split=key[0],protocol=key[1],source_group=key[2],method=key[3],threshold=key[4],wait_ms=wait,**value))
                check=dict(noise=family,wait_ms=wait,conditions=len(metas),history_associations=sum(x['associations'] for x in diagnostic),nonzero_propagated_candidates=sum(x['nonzero_motion'] for x in diagnostic),anchor_pairs=sum(x['anchor_pairs'] for x in diagnostic),causal_history=True,GT_absent_in_inference=True,same_environment_readout_replay_exact=True)
                if wait==400:assert check['history_associations']>0 and check['nonzero_propagated_candidates']>0,'nonzero asynchronous heading input required'
                checks.append(check);atomic(sub/'terminal.json',dict(status='complete_light_checked',created_at=now(),checks=check,predictions=ev(sub/'prediction_and_features.npz'),conditions=ev(sub/'conditions.json')));atomic(out/'panels.json',panels)
        reads=[]
        for family in ('SPD_structured','iid_02m_02deg'):
            for wait in (0,400):
                lookup={(r['protocol'],r['source_group'],r['method'],r['threshold']):r for r in panels if r['noise']==family and r['wait_ms']==wait and r['split']=='test'}
                diff={p:lookup[p,'all','K1_heading_U',.7]['ap']-lookup[p,'all','stale_same_U',.7]['ap'] for p in ('P1_simulation','P2_inherited')}
                reads.append(dict(noise=family,wait_ms=wait,AP70_delta=diff,preregistered_flip=diff['P1_simulation']>0 and diff['P2_inherited']<0))
        atomic(out/'preregistered_read.json',reads);atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),panels=ev(out/'panels.json'),checks=checks,read=ev(out/'preregistered_read.json'),GPU_forwards=0,updates=0,wall_s=time.monotonic()-start,independent_review_pending=True,no_model_wakeup=True,scope=c['scientific_scope']));record('opv2v_async_heading_U',dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start));record('opv2v_async_heading_U',dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
