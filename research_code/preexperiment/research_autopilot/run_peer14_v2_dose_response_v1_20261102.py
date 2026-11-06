"""Fixed V2 population beta intervention, exact cached/native source joins."""
import argparse,os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from collections import defaultdict
import numpy as np
import run_peer13_v2_auxiliary_ap_v1_20261102 as old
from peer14_cycle21_v1_20261102 import ROOT,CYCLE,atomic,ev,load,now,register,record
BASE=CYCLE/'v2_dose_response_v1';NAME='v2_dose_response'
PROTOCOLS=('official','beta025','beta050','beta075','auxiliary_label_registered')

def prepare():
    BASE.mkdir(exist_ok=False)
    register(NAME,__file__,'Report all five beta points on fixed saved current/native populations, not physical ground truth. beta0 exactly replays official and beta1 replays already completed C15 auxiliary AP; retain flips and no-flips. No method selection by beta.',dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='2-10',new_disk_MiB='<30 metrics only; predictions/labels referenced'))
    c=load(CYCLE/(NAME+'_contract.json'));c.update(original=ev(old.__file__),native_population_contract=ev(old.BASE/'native_population_v6_contract.json'),native_replay=ev(old.BASE/'producer_v6/panels.json'),cached_replay=ev(old.BASE/'producer_v2/panels.json'),label_rule=ev(old.LABELS/'label_rule_frozen.json'),transforms=ev(old.LABELS/'replacement_transforms.json'),lineage=load(old.BASE/'contract.json')['inputs']['lineage'],betas=[0,.25,.5,.75,1],
        intervention='Only remote-exclusive targets: R(beta*theta) x + beta*t in left ego coordinates, same frozen label-assisted SE2 as C15. beta0 exact official; beta1 uses saved R exactly. z/dimensions/local labels/GT population/prediction identities/score order/ROI untouched. This is translation+angle scaling, not SE2 group exponential.',scope='Current8 cached methods and same-budget native PUBLIC/ONLINE x two input factors; both saved roles/two waits; no new native hold inference. Historical methods and registration point remain explicit followups, not implicitly covered.')
    atomic(CYCLE/(NAME+'_contract.json'),c)

def labels(frame,tr,cached,gt,classmap):
    if cached:
        assert np.array_equal(gt,np.asarray(frame['formal_cache_boxes'],np.float32));ll=frame['formal_labels']
        assert [int(v['resolved_id']) for v in ll]==list(map(int,frame['cache_to_raw_yaml_mapping']['mapped_resolved_ids']))
    else:
        possible=[v for v in frame['official_labels'] if classmap[v['class_index']]==0 and -100<=v['box_ego_lidar_hwl'][0]<=100 and -40<=v['box_ego_lidar_hwl'][1]<=40];assert len(possible)>=len(gt)
        ll=[];used=set()
        for b in gt:
            hits=[i for i,v in enumerate(possible) if np.max(np.abs(np.asarray(v['box_ego_lidar_hwl'])[:6]-b[:6]))<1e-4 and abs(np.arctan2(np.sin(v['box_ego_lidar_hwl'][6]-b[6]),np.cos(v['box_ego_lidar_hwl'][6]-b[6])))<1e-4]
            assert len(hits)==1 and hits[0] not in used;used.add(hits[0]);ll.append(possible[hits[0]])
    assert len(ll)==len(gt);remote=np.array([v['category']=='remote_only' for v in ll],bool);result={}
    for p,beta in zip(PROTOCOLS,[0,.25,.5,.75,1]):
        a=gt.copy();theta=beta*tr['theta'];R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) if beta!=1 else np.asarray(tr['R']);t=beta*np.asarray(tr['t'])
        if beta:a[remote,:2]=gt[remote,:2]@R.T+t;a[remote,6]=gt[remote,6]+theta
        assert np.array_equal(a[~remote],gt[~remote]) and np.array_equal(a[:,2:6],gt[:,2:6]);result[p]=a
    assert np.array_equal(result['official'],gt)
    return remote,result

def run():
    c=load(CYCLE/(NAME+'_contract.json'));assert ev(__file__)==c['script']
    for k in ('original','native_population_contract','native_replay','cached_replay','label_rule','transforms','lineage'):assert ev(ROOT/c[k]['path'])==c[k],k
    frames=old.frame_map(ROOT/c['lineage']['path']);trs={(str(r['split']),str(r['sequence']),str(r['timestamp'])):r for r in load(ROOT/c['transforms']['path'])};classmap=load(ROOT/c['native_population_contract']['path'])['class_map']
    refs=load(ROOT/c['native_replay']['path']);panels=[];out=BASE/'producer';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();record(NAME,dict(status='running_beta_scan'))
    def finish(stats,identity):
        pp=old.finish(stats,identity)
        for p in pp:
            if p['protocol'] in ('official','auxiliary_label_registered'):
                rr=[v for v in refs if all(v.get(k)==p.get(k) for k in ('family','role','method','protocol','source_group','wait_ms','threshold'))];assert len(rr)==1,(p,len(rr));assert abs(p['AP']-rr[0]['AP'])<1e-8,(p,rr[0])
        panels.extend(pp);atomic(out/'panels.json',panels)
    try:
        vc=load(old.V2/'contract_recovery_v6.json');head=old.V2/'producer_v1/q005_cv_matched39d_head'
        for role in ('val21','mechanism_holdout14'):
            split='val' if role=='val21' else 'train';ds=old.fac.PackedDataset(ROOT/vc['datasets'][split]);m,rows=old.transfer.material(vc,role,ds);states=old.science.core.load_npz(old.TRANSFER/f'{role}_state.npz');pred=old.science.core.load_npz(old.TRANSFER/f'{role}_predictions.npz');saved=old.science.core.load_npz(old.V2/'producer_v1'/f'{role}_q005_state.npz');assert np.array_equal(saved['state_key_sequence'],m['sequence'])
            boxes={};scores={}
            for n in ('K2_infra','K3_K1_expanded','K3_K2_expanded'):state={k.split('__',1)[1]:v for k,v in states.items() if k.startswith(n+'__')};boxes[n],_,_=old.science.geometry(m,state)
            boxes['K1_exact'],_=old.fac.heading_geometry(m,saved['q005_delta'])
            for n in ('stale_raw','stale_same_U','CV_raw','CV_same_U'):
                kind='stale' if n.startswith('stale') else 'cv';boxes[n]=m[kind+'_boxes'];scores[n]=m['raw_scores'] if n.endswith('_raw') else old.science.core.refit.score_with_head(m,m[kind+'_features'],head)
            targets={i:labels(frames[(split,str(r['sequence']),str(r['timestamp']))],trs[(split,str(r['sequence']),str(r['timestamp']))],True,ds.pool(int(m['dataset_index'][i]))['gt_boxes'].astype(np.float32),classmap) for i,r in enumerate(rows)}
            for method,bb in boxes.items():
                st=defaultdict(lambda:dict(scores=[],flags=[],gt=0))
                for i,r in enumerate(rows):
                    ref=None
                    if method in scores:
                        a,b=map(int,m['condition_offsets'][i:i+2]);ids,ss=old.fac.formal.ordered(m['raw_scores'][a:b],m['source'][a:b],m['protected'][a:b],scores[method][a:b],None);ids=a+ids
                    else:
                        a,b=map(int,pred[method+'__offset'][i:i+2]);ids=pred[method+'__candidate'][a:b];ss=pred[method+'__score'][a:b];ref={k:pred[method+'__'+k][a:b] for k in ('tp50','tp70')}
                    remote,gt=targets[i];old.add(st,bb[ids],ss,gt,remote,r['wait_ms'],r['sequence'],ref)
                    if i%128==0:atomic(out/'status.json',dict(status='cached_beta',role=role,method=method,conditions=i+1,wall_s=time.monotonic()-start))
                finish(st,dict(family='V2_current_cached',role=role,method=method))
        for train in ('public','online'):
            producer=old.AUDIT/('cycle15_peer11_v1/v2_'+train);records=load(producer/'records.json')
            for role in ('val21','mechanism_holdout14'):
                split='val' if role=='val21' else 'train'
                for wait in (0,400):
                    rs=[r for r in records if r['role']==role and r['wait_ms']==wait];stats={arm:defaultdict(lambda:dict(scores=[],flags=[],gt=0)) for arm in ('PUBLIC','ONLINE_K1')}
                    with np.load(producer/f'{role}_wait{wait:03d}.npz') as z:
                        for i,r in enumerate(rs):
                            n=r['shard_case'];gt=z[f'c{n}_gt'];gt=gt[gt[:,7]==0,:7];key=(split,str(r['sequence']),str(r['timestamp']));remote,targets=labels(frames[key],trs[key],False,gt,classmap)
                            for arm in stats:
                                bb=z[f'c{n}_{arm}_box3d_lidar'];ss=z[f'c{n}_{arm}_scores'];cl=z[f'c{n}_{arm}_label_preds'];keep=(cl==0)&(bb[:,0]>=-100)&(bb[:,0]<=100)&(bb[:,1]>=-40)&(bb[:,1]<=40);bb=bb[keep];ss=ss[keep];order=np.argsort(-ss);old.add(stats[arm],bb[order],ss[order],targets,remote,wait,r['sequence'])
                            if i%128==0:atomic(out/'status.json',dict(status='native_beta',training=train,role=role,wait_ms=wait,conditions=i+1,wall_s=time.monotonic()-start))
                    for arm,st in stats.items():finish(st,dict(family='V2_native_TraF',role=role,method=train+'_trained__'+arm,training=train,input=arm))
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),panels=ev(out/'panels.json'),both_endpoint_AP_replays_exact=True,all_populations_unchanged=True,scope=c['scope'],independent_review_pending=True,cost=dict(wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,GPU_forwards=0),no_model_wakeup=True));record(NAME,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),partial_panels=len(panels)));record(NAME,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();prepare() if a.mode=='prepare' else run()
