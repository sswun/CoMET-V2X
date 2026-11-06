"""Estimator-only P1/P2 controlled inheritance read on fixed sensor boxes."""
import argparse,ast,json,os,time,traceback,types
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from collections import defaultdict
import numpy as np
from peer14_cycle21_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,atomic,ev,load,now,register,record
BASE=CYCLE/'opv2v_controlled_read_v1'
INPUT=CYCLE/'opv2v_inheritance_v1/producer'

def prepare():
    BASE.mkdir(exist_ok=False)
    register('opv2v_controlled_read',__file__,'P1 positive/P2 negative q005-estimator change is a preregistered controlled diagnosis, not guaranteed. Same boxes, same noise, same readout; no GT-based associations. Keep both noise families and all source groups. Diagnostic estimator-only cannot substitute full K1 heading+U transfer.',dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='1-5 finite after400frames',new_disk_MiB='<25 metrics/SE2 noise/state/identities; detections referenced'))
    # Fit the noise without any OPV2V input, annotations, predictions or AP.
    data=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json'
    shared=load(AUDIT/'contract_v1.json')
    with np.load(ROOT/shared['inputs']['fit24_material']['path']) as z:seqs=set(map(str,z['sequence']))
    groups=defaultdict(list)
    for r in load(data):
        if str(r['vehicle_sequence']) in seqs:groups[str(r['vehicle_sequence'])].append(r)
    centers=[];slowall=[];innov=[];jit=[];lag0=[];lag1=[]
    for seq,rows in groups.items():
        rows.sort(key=lambda r:str(r['vehicle_frame']));a=np.array([[r['system_error_offset'][k] for k in ('delta_x','delta_y')] for r in rows]);center=a.mean(0);slow=np.array([a[max(0,i-2):i+3].mean(0) for i in range(len(a))])-center
        centers.append(center);slowall.extend(slow);jit.extend(a-center-slow);lag0.extend(slow[:-1]);lag1.extend(slow[1:])
    x,y=np.array(lag0),np.array(lag1);rho=float(np.clip(np.sum(x*y)/max(1e-12,np.sum(x*x)),-.95,.95));innovation=y-rho*x
    noise=dict(sequence_means=np.asarray(centers).tolist(),rho=rho,slow_initial_cov=np.cov(np.asarray(slowall).T).tolist(),innovation_cov=np.cov(innovation.T).tolist(),jitter_cov=np.cov(np.asarray(jit).T).tolist(),fit_sequences=sorted(groups),metadata=ev(data),fit_material=shared['inputs']['fit24_material'],structured_yaw_rad=0.,iid_sigma_translation_m=.2,iid_sigma_yaw_degrees=.2,seed=20261102)
    atomic(BASE/'noise_frozen.json',dict(created_at=now(),fit_only_SPD=True,parameters=noise,no_OPV2V_test_or_AP_read=True))

def transform(bb,T):
    result=bb.copy();result[:,:3]=bb[:,:3]@T[:3,:3].T+T[:3,3];result[:,6]+=np.arctan2(T[1,0],T[0,0]);return result

def estimator(counts,measurements,sequences):
    path=ROOT/'preexperiment/research_autopilot/run_crosscheckpoint_temporal_q005_task_refit_v1_20261029.py'
    functions=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='q005_frame'];assert len(functions)==1
    ns=dict(np=np,fixed=types.SimpleNamespace(anchor_inputs=lambda old:(counts,measurements)))
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ns)
    old=dict(feature=np.zeros((len(counts),5)));material=dict(sequence=np.asarray(sequences),wait_ms=np.zeros(len(counts),int))
    return ns['q005_frame'](old,material,False)['kf_delta'],ev(path)

def run():
    from audit_common_offset_peer_v1_20261026 import pairs
    from run_peer07_spd_gt_source_split_ap_v1_20261031 import iou32
    from run_peer07_v2_strict400_source_split_ap_v1_20261031 import outcomes
    from peer07_source_ap_producer_utils_v1_20261031 import metric
    out=BASE/'producer';out.mkdir(exist_ok=False);start=time.monotonic();record('opv2v_controlled_read',dict(status='waiting_fixed_sensor_detections'))
    while not (INPUT/'terminal.json').exists():
        if time.monotonic()-start>24*3600:raise TimeoutError('finite24h detector collector wait')
        time.sleep(30)
    assert load(INPUT/'terminal.json')['status']=='complete_light_checked'
    c=load(CYCLE/'opv2v_controlled_read_contract.json');assert ev(__file__)==c['script'];noise=load(BASE/'noise_frozen.json')['parameters'];rows=load(INPUT/'records.json');panels=[]
    def nms(boxes,scores):
        order=np.argsort(-scores,kind='stable');keep=[]
        while len(order):
            i=int(order[0]);keep.append(i);other=order[1:];order=other[iou32(boxes[i:i+1],boxes[other]).ravel()<=.2] if len(other) else other
        return np.array(keep,int)
    try:
        for family in ('SPD_structured','iid_02m_02deg'):
            rng=np.random.default_rng(noise['seed']+(0 if family=='SPD_structured' else 1));state={};prepared=[];counts=[];measurements=[];sequences=[]
            for r in rows:
                with np.load(ROOT/r['shard']['path']) as z:a={k:z[k].copy() for k in z.files}
                assert ev(ROOT/r['shard']['path'])==r['shard'];key=(r['split'],r['sequence'])
                if family=='SPD_structured':
                    if key not in state:state[key]=(np.array(noise['sequence_means'])[rng.integers(len(noise['sequence_means']))],rng.multivariate_normal(np.zeros(2),noise['slow_initial_cov']))
                    center,slow=state[key];slow=noise['rho']*slow+rng.multivariate_normal(np.zeros(2),noise['innovation_cov']);state[key]=(center,slow)
                    t=center+slow+rng.multivariate_normal(np.zeros(2),noise['jitter_cov']);theta=0.
                else:t=rng.normal(0,.2,2);theta=float(rng.normal(0,np.deg2rad(.2)))
                D=np.eye(4);D[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]];D[:2,3]=t
                ego=a['agent0_boxes'];clean=transform(a['agent1_boxes'],a['remote_to_ego']);noisy=transform(clean,D)
                ri,ei=pairs(noisy,ego,2.5,True);residual=ego[ei,:2]-noisy[ri,:2]
                counts.append(len(ri));measurements.append(residual.mean(0) if len(ri) else np.zeros(2));sequences.append(r['split']+'/'+r['sequence']);prepared.append((r,a,ego,clean,noisy,D))
            delta,kernel=estimator(np.array(counts),np.array(measurements),sequences);stats=defaultdict(lambda:dict(scores=[],tp=[],fp=[],gt=0));details=[]
            for i,(r,a,ego,clean,noisy,D) in enumerate(prepared):
                corrected=noisy.copy();corrected[:,:2]+=delta[i];scores=np.r_[a['agent0_scores'],a['agent1_scores']]
                labels={'P1_simulation':a['gt_hwl'].copy(),'P2_inherited':a['gt_hwl'].copy()};remote=a['remote_exclusive'];labels['P2_inherited'][remote]=transform(labels['P2_inherited'][remote],D)
                assert np.array_equal(labels['P1_simulation'][~remote],labels['P2_inherited'][~remote])
                for method,bb in (('stale',noisy),('q005_KF_estimator_only',corrected),('oracle_clean_pose',clean)):
                    boxes=np.concatenate([ego,bb]);keep=nms(boxes,scores);boxes=boxes[keep];ss=scores[keep]
                    for protocol,gt in labels.items():
                        overlap=iou32(boxes,gt)
                        for group,mask in [('all',np.ones(len(gt),bool)),('ego_visible',~remote),('remote_exclusive',remote)]:
                            for th in (.5,.7):
                                tp,fp,ignore=outcomes(overlap,mask,th);v=stats[r['split'],protocol,group,method,th];valid=ignore==0;v['scores'].append(ss[valid]);v['tp'].append(tp[valid]);v['fp'].append(fp[valid]);v['gt']+=int(mask.sum())
                details.append(dict(split=r['split'],sequence=r['sequence'],frame=r['frame'],noise_left_SE2=D.tolist(),q005_delta=delta[i].tolist(),pairs=int(counts[i])))
                if i%16==0:atomic(out/'status.json',dict(status='controlled_P1_P2_read',created_at=now(),family=family,frames=i+1,total=len(rows)))
            for key,v in stats.items():
                result=metric(np.concatenate(v['tp']),np.concatenate(v['fp']),np.concatenate(v['scores']),v['gt']);panels.append(dict(noise=family,split=key[0],protocol=key[1],source_group=key[2],method=key[3],threshold=key[4],**result))
            atomic(out/(family+'_state_and_noise.json'),details);atomic(out/'panels.json',panels)
        read=[]
        for family in ('SPD_structured','iid_02m_02deg'):
            lookup={(r['protocol'],r['source_group'],r['method'],r['threshold']):r for r in panels if r['noise']==family and r['split']=='test'}
            diff={p:lookup[p,'all','q005_KF_estimator_only',.7]['ap']-lookup[p,'all','stale',.7]['ap'] for p in ('P1_simulation','P2_inherited')}
            read.append(dict(noise=family,test_AP70_delta=diff,sign_flip=diff['P1_simulation']>0 and diff['P2_inherited']<0,scope='Estimator-only, bounded frame sample; not full K1/U/heading or main protocol.'))
        atomic(out/'preregistered_read.json',read)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),panels=ev(out/'panels.json'),read=ev(out/'preregistered_read.json'),estimator_kernel=kernel,noise_frozen=ev(BASE/'noise_frozen.json'),GPU_forwards=0,updates=0,wall_s=time.monotonic()-start,independent_review_pending=True,no_model_wakeup=True));record('opv2v_controlled_read',dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('opv2v_controlled_read',dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();prepare() if a.mode=='prepare' else run()
