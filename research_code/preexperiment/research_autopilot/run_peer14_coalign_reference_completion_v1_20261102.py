"""Unanswered CoAlign fair-weight slow ceiling and ego evidence, no new method.

Queue after the sole current confirmation GPU owners. Author PUBLIC/ONLINE
seed1 weights stay fixed. Annotation-assisted inputs are never deployable.
"""
import argparse,inspect,time,traceback,copy,os,subprocess
from peer14_cycle21_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C15,C18,atomic,ev,load,now,register,record
import run_peer13_coalign_seed_confirmation_v1_20261102 as confirm
BASE=CYCLE/'coalign_reference_completion_v1'

def replace(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)

def variants(public,row):
    import numpy as np
    frame=str(row['current_vehicle_frame']);result={}
    for name,field in [('OFFICIAL_SEQMEAN_REFERENCE','official_seqmean'),('OFFICIAL_MA5_REFERENCE','official_MA5')]:
        v=[x.copy() for x in public]
        if len(v)>1:v[1][:2,3]+=SLOW[field][frame]
        result[name]=v
    return result

def build(training,mode):
    old=confirm.old;stage=confirm.stage;ns=dict(old.__dict__,scorers=confirm.fixed.scorers,OUTPUT=BASE/training/mode/'evaluation',ARMS=['EGO_ONLY_REFERENCE'] if mode=='ego' else ['OFFICIAL_SEQMEAN_REFERENCE','OFFICIAL_MA5_REFERENCE'],MODE=mode,FUSION_WEIGHT=C15/'coalign_full_v1'/('fusion_fit/final.pth' if training=='public' else 'online_fusion_fit/final.pth'),make_variants=lambda public,row:dict(EGO_ONLY_REFERENCE=[__import__('numpy').eye(4)]) if mode=='ego' else variants(public,row),REFERENCE_CONTRACT=CYCLE/'coalign_reference_completion_contract.json',record=lambda label,value:None)
    ns['OUTPUT'].parent.mkdir(parents=True,exist_ok=True)
    s=inspect.getsource(old.evaluate)
    changes=[("name='adapted_evaluation' if adapted else 'author_evaluation';out=BASE/(name+('_profile' if profile else ''))","name='frozen_reference';out=OUTPUT.with_name(OUTPUT.name+'_profile') if profile else OUTPUT"),
        ("from run_peer07_native_cvh_source_split_ap_v1_20261031 import polygons,matrix,metric\n        from run_peer10_native_source_aux_v1_20261101 import outcome","polygons,matrix,metric,outcome=scorers()"),
        ("load(BASE/'pipeline_contract.json')","load(BASE/'pipeline_contract_v3.json')"),("ev(BASE/'pipeline_contract.json')","ev(REFERENCE_CONTRACT)"),
        ("weight=BASE/'fusion_fit/final.pth' if adapted else stage.ASSETS/'fusion_DAIR/net_epoch_bestval_at13.pth'",'weight=FUSION_WEIGHT'),
        ("arms=['COALIGN_FULL','NO_GRAPH','K1_INPUT']",'arms=ARMS'),("for j,source in enumerate(ident):","for j,source in []:"),
        ("if len(public)>1 and all(len(x) for x in stagecorners):","if False:"),
        ("corner_to_center(corners.detach().cpu().numpy(),'lwh')","corner_to_center(corners.detach().cpu().numpy(),'hwl')"),
        ("variants=dict(COALIGN_FULL=graph,NO_GRAPH=public,K1_INPUT=k1);gi=","variants=make_variants(public,row);gi="),
        ("inputs=model_input(pc,transforms,pre,np,torch,device)","inputs=model_input(pc[:1] if MODE=='ego' else pc,transforms,pre,np,torch,device)"),
        ("for wait in (0,100,200,400):","for wait in ((0,) if MODE=='ego' else (0,100,200,400)):"),
        ("complete_CoAlign=not profile","complete_CoAlign=False")]
    for a,b in changes:s=replace(s,a,b)
    exec(compile(s,__file__+'::frozen_own_model_reference','exec'),ns);return ns

def prepare():
    BASE.mkdir(exist_ok=False)
    register('coalign_reference_completion',__file__,'CoAlign complete-system vs K1 gap remains unmeasured against its own slow-label ceiling and matched ego-only evidence. Same first-seed PUBLIC/ONLINE fusion weights, no new fit/method. Label-assisted MA5/seqmean may alter upper gaps either way; ego probe wait0 alone. Reuse existing deploy/reference AP, no repeated complete-system forward.',dict(GPU_forwards_per_lane=29844,profile_plus_replay_per_lane='<180',updates=0,GPU_wall_h_per_lane='2-5 under CPU thermal quota',new_disk_GiB='<1 unique compressed predictions; shared GT referenced'))
    import numpy as np
    from collections import defaultdict
    groups=defaultdict(list);data=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json'
    for r in load(data):groups[str(r['vehicle_sequence'])].append(r)
    slow={k:{} for k in ('official_seqmean','official_MA5')}
    for seq,rows in groups.items():
        rows.sort(key=lambda r:str(r['vehicle_frame']));a=np.array([[r['system_error_offset'][k] for k in ('delta_x','delta_y')] for r in rows]);mean=a.mean(0)
        for i,r in enumerate(rows):f=str(r['vehicle_frame']);slow['official_seqmean'][f]=mean.tolist();slow['official_MA5'][f]=a[max(0,i-2):i+3].mean(0).tolist()
    atomic(BASE/'slow_offsets_frozen.json',dict(metadata=ev(data),offline_annotation_reference=True,fields=slow))
    c=load(CYCLE/'coalign_reference_completion_contract.json');c.update(original=ev(confirm.old.__file__),fixed_scoring=ev(confirm.fixed.__file__),slow=ev(BASE/'slow_offsets_frozen.json'),population_contract=ev(C18/'coalign_sources_ROI_v2/contract.json'),weights={n:ev(C15/'coalign_full_v1'/f) for n,f in [('public','fusion_fit/final.pth'),('online','online_fusion_fit/final.pth')]},scope='Only val21; each lane after existing seed confirmation. Ego-only one wait0 (avoid four identical forward groups), two slow-input references all4waits. Same original GT/ROI; common/source/aux CPU consumer queued. No CoAlign hold/noisy-training novelty claim.')
    atomic(CYCLE/'coalign_reference_completion_contract.json',c)
    for t in ('public','online'):
        for m in ('ego','slow'):build(t,m)
        record('coalign_reference_'+t,dict(status='registered_finite_successor',contract=ev(CYCLE/'coalign_reference_completion_contract.json')))
    record('coalign_reference_sources',dict(status='registered_finite_successor',contract=ev(CYCLE/'coalign_reference_completion_contract.json')))
    record('coalign_reference_completion',dict(status='complete_light_checked',administrative_contract_only=True,scientific_children_remain_open=True))

def run(training,gpu):
    global SLOW
    c=load(CYCLE/'coalign_reference_completion_contract.json');assert c['script']==ev(__file__) and c['original']==ev(confirm.old.__file__) and c['fixed_scoring']==ev(confirm.fixed.__file__);import numpy as np
    SLOW={k:{f:np.array(v) for f,v in x.items()} for k,x in load(BASE/'slow_offsets_frozen.json')['fields'].items()};assert c['slow']==ev(BASE/'slow_offsets_frozen.json') and c['weights'][training]==ev(ROOT/c['weights'][training]['path'])
    out=BASE/training;out.mkdir(exist_ok=True);start=time.monotonic();prior='embodied-peer13-coalign-confirmation-'+('gpu0' if training=='public' else 'gpu1')+'-v1-20261102.service'
    record('coalign_reference_'+training,dict(status='waiting_sole_confirmation_GPU_owner',prior_unit=prior))
    try:
        while subprocess.check_output(['systemctl','--user','show',prior,'-p','ActiveState'],text=True).strip() in ('ActiveState=active','ActiveState=activating'):
            if time.monotonic()-start>18*3600:raise TimeoutError('finite18h current GPU wait')
            time.sleep(30)
        for mode in ('ego','slow'):
            ns=build(training,mode);ns['evaluate'](gpu,True,True);ns['evaluate'](gpu,True,False)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),contract=ev(CYCLE/'coalign_reference_completion_contract.json'),models_unchanged=True,no_training=True,no_model_wakeup=True));record('coalign_reference_'+training,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('coalign_reference_'+training,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

def sources():
    import run_peer13_coalign_confirmation_sources_v1_20261102 as old
    ns,spec=old.namespace();spec=copy.deepcopy(spec);spec['native_source_aux']['native_eval_range']=[-100.8,-40,-3.5,100.8,40,1.5];ns['CYCLE']=BASE;out=BASE/'sources';out.mkdir();start=time.monotonic();results=[];record('coalign_reference_sources',dict(status='waiting_fixed_new_reference_predictions'))
    try:
        for training in ('public','online'):
            for mode in ('ego','slow'):
                p=BASE/training/mode/'evaluation'
                while not (p/'terminal.json').exists():
                    if (BASE/training/'terminal.json').exists() and load(BASE/training/'terminal.json')['status']=='failed_retained':raise RuntimeError('failedGPU reference')
                    if time.monotonic()-start>24*3600:raise TimeoutError('finite24h reference wait')
                    time.sleep(30)
                ns['compute'](str(p.relative_to(AUDIT)),spec);results.append(ev(BASE/'native_source_aux'/p.relative_to(AUDIT)/'terminal.json'))
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),results=results,no_model_wakeup=True));record('coalign_reference_sources',dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),results=results));record('coalign_reference_sources',dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','public','online','sources']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();prepare() if a.mode=='prepare' else sources() if a.mode=='sources' else run(a.mode,a.gpu)
