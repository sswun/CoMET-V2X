"""Fixed-prediction official/label-only V2 AP, current cached and native arms."""
import argparse,json,os,time,traceback,resource
from collections import defaultdict
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from peer13_cycle15_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,atomic,ev,load,now,update,task
import run_peer10_v2_transfer_v1_20261101 as transfer
from run_peer07_v2_strict400_source_split_ap_v1_20261031 import outcomes
from peer07_source_ap_producer_utils_v1_20261031 import metric
science=transfer.science;fac=transfer.fac
BASE=CYCLE/'v2_auxiliary_ap_v1';LABELS=CYCLE/'v2_auxiliary_labels_v1/producer_v2'
V2=AUTO/'status_audit_20261030_v2v4real_current_q005_heading_anchor_gate_factorial_v1'
TRANSFER=AUDIT/'cycle03_v1/v2_transfer_float_recovery_v2'
TID='peer13_v2_auxiliary_ap_v1_20261102'
GROUPS=('all','local_visible','remote_exclusive');ROLES=('mechanism_holdout14','val21')

def prepare():
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):(AUTO/n).read_text();refs[n]=ev(AUTO/n)
    BASE.mkdir(exist_ok=False)
    inputs={'lineage':ev(transfer.ROOT/'artifacts/research_autopilot/status_audit_20261029_peer_revision/v2_annotation_lineage_z4_recovery_v6/frame_lineage.jsonl'),'v2_contract':ev(V2/'contract_recovery_v6.json')}
    for role in ROLES:
        for kind in ('state','predictions'):inputs[role+'_'+kind]=ev(TRANSFER/f'{role}_{kind}.npz')
    for train in ('public','online'):
        p=AUDIT/('cycle15_peer11_v1/v2_'+train);inputs[train+'_records']=ev(p/'records.json');inputs[train+'_panels']=ev(p/'panels.json')
        for role in ROLES:
            for w in (0,400):inputs[f'{train}_{role}_{w}']=ev(p/f'{role}_wait{w:03d}.npz')
    atomic(BASE/'contract.json',dict(created_at=now(),script=ev(__file__),instructions=refs,inputs=inputs,label_dependency=str((LABELS/'terminal.json').relative_to(ROOT)),
        prediction_before_result='Report official and independently registered remote-only auxiliary labels on all fixed current K1/K2/K3 and V2-native PUBLIC/ONLINE training/input factors; rankings may change either way. No estimator, readout, checkpoint, gate or labels selected on method AP.',
        methods='Cached stale_raw, stale_same_U, CV_raw, CV_same_U, exact original K1 and K2/K3 raw candidates; native same-budget PUBLIC/ONLINE weights x PUBLIC/ONLINE inputs. Current own V2 inputs/head only, not SPD transfer or old truncated U16 states.',
        population='Cached unchanged historical pool, no post-geometry ROI. Native class0 predictions/current GT, original inclusive XY[-100,100]x[-40,40]. Original GT identity/ROI frozen before auxiliary movement; dimensions/z/class/local targets unchanged.',
        source_join='Exact cached-to-raw formal indices and resolved IDs; native unique full HWL/center/yaw agreement within1e-4 to raw current official labels. Ambiguity/missing identity fails, never detection-derived source attribution.',
        evaluation='Canonical float32 HWL IoU, score-ranked greedy; target-first matching then ignore other-source regions. Official full-GT prediction TP replay and native pooled AP replay required. Auxiliary is annotation sensitivity, not physical ground truth.',
        cost=dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='0.5-3 shared thermal quota',new_disk_MiB='<100 outcomes/small metrics, shared inputs/GT/predictions referenced'),no_model_wakeup=True))
    update(TID,task('K5'),dict(status='registered_ready',contract=ev(BASE/'contract.json'),prediction_before_result=load(BASE/'contract.json')['prediction_before_result']))

def frame_map(path):
    return {(str(f['split']),str(f['sequence']),str(f['timestamp'])):f for f in map(json.loads,path.open())}

def target(frame,transform,cached,gt):
    if cached:
        assert np.array_equal(gt,np.asarray(frame['formal_cache_boxes'],np.float32))
        mapping=frame['cache_to_raw_yaml_mapping'];labels=[frame['formal_labels'][i] for i in mapping['mapped_raw_formal_indices']]
        assert [int(v['resolved_id']) for v in labels]==list(map(int,mapping['mapped_resolved_ids']))
    else:
        possible=[v for v in frame['official_labels'] if v['class_index']==0 and -100<=v['box_ego_lidar_hwl'][0]<=100 and -40<=v['box_ego_lidar_hwl'][1]<=40]
        assert len(possible)==len(gt),('native population',len(possible),len(gt),frame['sequence'],frame['timestamp'])
        labels=[];used=set()
        for box in gt:
            hits=[]
            for j,label in enumerate(possible):
                other=np.asarray(label['box_ego_lidar_hwl']);yaw=abs(np.arctan2(np.sin(box[6]-other[6]),np.cos(box[6]-other[6])))
                if np.max(np.abs(box[:6]-other[:6]))<1e-4 and yaw<1e-4:hits.append(j)
            assert len(hits)==1 and hits[0] not in used,('native exact label identity',hits)
            used.add(hits[0]);labels.append(possible[hits[0]])
    assert len(labels)==len(gt)
    remote=np.array([v['category']=='remote_only' for v in labels]);aux=gt.copy();R=np.asarray(transform['R']);t=np.asarray(transform['t']);theta=transform['theta']
    assert set(int(v['resolved_id']) for v in labels if v['category']=='remote_only').issubset(set(transform['remote_only_resolved_ids']))
    aux[remote,:2]=gt[remote,:2]@R.T+t;aux[remote,6]=gt[remote,6]+theta
    assert np.array_equal(aux[~remote],gt[~remote]) and np.array_equal(aux[:,2:6],gt[:,2:6])
    return remote,dict(official=gt,auxiliary_label_registered=aux)

def add(stats,bb,ss,gt_by_protocol,remote,wait,sequence,ref=None):
    assert len(bb)==len(ss) and np.isfinite(bb).all() and np.isfinite(ss).all() and np.all(np.diff(ss)<=0)
    for protocol,gt in gt_by_protocol.items():
        iou=fac.task_recipe.iou32(bb,gt)
        for group,mask in zip(GROUPS,(np.ones(len(gt),bool),~remote,remote)):
            for threshold in (50,70):
                flags=outcomes(iou,mask,threshold/100);assert np.all(sum(flags)==1)
                if ref is not None and protocol=='official' and group=='all' and 'tp'+str(threshold) in ref:assert np.array_equal(flags[0],ref['tp'+str(threshold)]),('saved TP replay',sequence,wait,threshold)
                v=stats[protocol,group,int(wait),threshold];v['scores'].append(ss);v['flags'].append(np.stack(flags));v['gt']+=int(mask.sum())

def finish(stats,identity):
    result=[]
    for (protocol,group,wait,threshold),v in stats.items():
        scores=np.concatenate(v['scores']);f=np.concatenate(v['flags'],axis=1);keep=f[2]==0;m=metric(f[0,keep],f[1,keep],scores[keep],v['gt'])
        result.append(dict(identity,protocol=protocol,source_group=group,wait_ms=wait,threshold=threshold,AP=m['ap'],gt=v['gt'],tp=m['tp'],fp=m['fp'],ignored=int(f[2].sum())))
    return result

def run():
    c=load(BASE/'contract.json');assert ev(__file__)==c['script'];out=BASE/'producer';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time()
    update(TID,task('K5'),dict(status='waiting_frozen_label_rule'))
    try:
        while not (LABELS/'terminal.json').exists():time.sleep(20)
        terminal=load(LABELS/'terminal.json');assert terminal['status']=='labels_built_frozen_AP_pending';assert load(LABELS/'recovery_identity.json')['selection_and_all_transforms_unchanged']
        assert all(ev(ROOT/v['path'])==v for v in c['inputs'].values())
        frames=frame_map(ROOT/c['inputs']['lineage']['path']);transforms={(str(r['split']),str(r['sequence']),str(r['timestamp'])):r for r in load(LABELS/'replacement_transforms.json')}
        atomic(out/'frozen_labels_before_AP.json',dict(terminal=ev(LABELS/'terminal.json'),rule=ev(LABELS/'label_rule_frozen.json'),transforms=ev(LABELS/'replacement_transforms.json')))
        panels=[];identities=[];vc=load(V2/'contract_recovery_v6.json');head=V2/'producer_v1/q005_cv_matched39d_head'
        for role in ROLES:
            split='val' if role=='val21' else 'train';ds=fac.PackedDataset(ROOT/vc['datasets'][split]);m,rows=transfer.material(vc,role,ds);states=science.core.load_npz(TRANSFER/f'{role}_state.npz');pred=science.core.load_npz(TRANSFER/f'{role}_predictions.npz')
            saved=science.core.load_npz(V2/'producer_v1'/f'{role}_q005_state.npz');assert np.array_equal(saved['state_key_sequence'],m['sequence'])
            box={};score={}
            for name in ('K2_infra','K3_K1_expanded','K3_K2_expanded'):
                state={k.split('__',1)[1]:v for k,v in states.items() if k.startswith(name+'__')};box[name],f,_=science.geometry(m,state)
            box['K1_exact'],f=fac.heading_geometry(m,saved['q005_delta'])
            for name in ('stale_raw','stale_same_U'):box[name]=m['stale_boxes'];score[name]=m['raw_scores'] if name.endswith('_raw') else science.core.refit.score_with_head(m,m['stale_features'],head)
            for name in ('CV_raw','CV_same_U'):box[name]=m['cv_boxes'];score[name]=m['raw_scores'] if name.endswith('_raw') else science.core.refit.score_with_head(m,m['cv_features'],head)
            gtcache={}
            for i,r in enumerate(rows):
                key=(split,str(r['sequence']),str(r['timestamp']));gt=ds.pool(int(m['dataset_index'][i]))['gt_boxes'].astype(np.float32);gtcache[i]=target(frames[key],transforms[key],True,gt)
            for method,bb in box.items():
                stats=defaultdict(lambda:dict(scores=[],flags=[],gt=0))
                for i,r in enumerate(rows):
                    ref=None
                    if method in score:
                        a,b=map(int,m['condition_offsets'][i:i+2]);ids,ss=fac.formal.ordered(m['raw_scores'][a:b],m['source'][a:b],m['protected'][a:b],score[method][a:b],None);ids=a+ids
                    else:
                        a,b=map(int,pred[method+'__offset'][i:i+2]);ids=pred[method+'__candidate'][a:b];ss=pred[method+'__score'][a:b];ref={k:pred[method+'__'+k][a:b] for k in ('tp50','tp70')}
                    remote,targets=gtcache[i];add(stats,bb[ids],ss,targets,remote,r['wait_ms'],r['sequence'],ref)
                    if i%128==0:atomic(out/'status.json',dict(status='cached_AP',role=role,method=method,conditions=i+1,wall_s=time.monotonic()-start))
                panels.extend(finish(stats,dict(family='V2_current_cached',role=role,method=method)));atomic(out/'panels.json',panels)
            identities.append(dict(family='V2_current_cached',role=role,head=ev(head/'direct_u.pth'),state=ev(TRANSFER/f'{role}_state.npz'),predictions=ev(TRANSFER/f'{role}_predictions.npz'),all_saved_TP_exact=True))
        for train in ('public','online'):
            producer=AUDIT/('cycle15_peer11_v1/v2_'+train);records=load(producer/'records.json');reference=load(producer/'panels.json')
            for role in ROLES:
                split='val' if role=='val21' else 'train'
                for wait in (0,400):
                    rs=[r for r in records if r['role']==role and r['wait_ms']==wait];stats={arm:defaultdict(lambda:dict(scores=[],flags=[],gt=0)) for arm in ('PUBLIC','ONLINE_K1')}
                    with np.load(producer/f'{role}_wait{wait:03d}.npz') as z:
                        for i,r in enumerate(rs):
                            n=r['shard_case'];gt=z[f'c{n}_gt'];gt=gt[gt[:,7]==0,:7];key=(split,str(r['sequence']),str(r['timestamp']));remote,targets=target(frames[key],transforms[key],False,gt)
                            for arm in stats:
                                bb=z[f'c{n}_{arm}_box3d_lidar'];ss=z[f'c{n}_{arm}_scores'];cl=z[f'c{n}_{arm}_label_preds'];keep=(cl==0)&(bb[:,0]>=-100)&(bb[:,0]<=100)&(bb[:,1]>=-40)&(bb[:,1]<=40);bb=bb[keep];ss=ss[keep];order=np.argsort(-ss);add(stats[arm],bb[order],ss[order],targets,remote,wait,r['sequence'])
                            if i%128==0:atomic(out/'status.json',dict(status='native_AP',training=train,role=role,wait_ms=wait,conditions=i+1,wall_s=time.monotonic()-start))
                    for arm,st in stats.items():
                        pp=finish(st,dict(family='V2_native_TraF',role=role,method=train+'_trained__'+arm,training=train,input=arm));panels.extend(pp)
                        ref=next(r for r in reference if r['role']==role and r['wait_ms']==wait)['metrics'][arm]['global_score_sort']
                        for p in pp:
                            if p['protocol']=='official' and p['source_group']=='all':assert abs(p['AP']-100*ref['ap'+str(p['threshold'])])<1e-8,('native AP replay',p,ref)
                    atomic(out/'panels.json',panels)
            identities.append(dict(family='V2_native_TraF',training=train,records=ev(producer/'records.json'),weight=ev(producer/'final.pth'),official_AP_replayed=True))
        atomic(out/'input_identity.json',identities)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),panels=ev(out/'panels.json'),identities=ev(out/'input_identity.json'),official_GT_unchanged=True,auxiliary_population_frozen=True,full_independent_review_pending=True,remaining_release='Historical V2 external/old-prefix identities not silently covered by current candidates.',cost=dict(wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,GPU_forwards=0,updates=0),no_model_wakeup=True));update(TID,task('K5'),dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),completed_panels=len(locals().get('panels',[])),wall_s=time.monotonic()-start));update(TID,task('K5'),dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();prepare() if a.mode=='prepare' else run()
