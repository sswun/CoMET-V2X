"""Fixed full-system seeds: fresh uncertainty training plus paired fusion fits.

Reference arms skip uncertainty inference; graph arms wait for the corresponding
seed's sole stage1 final. Each scientific arm is evaluated exactly once.
"""
import argparse,inspect,subprocess,time,traceback
import run_peer13_coalign_full_v1_20261102 as stage
import run_peer13_coalign_pipeline_v1_20261102 as old
import run_peer13_coalign_pipeline_v2_20261102 as fixed
from peer13_cycle18_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C15,atomic,ev,load,now,update,task
BASE=CYCLE/'coalign_confirmation_v1';REMOTE=stage.REMOTE
def record(label,values):
    tid='peer13_coalign_confirmation_'+label+'_v1_20261102'
    if REMOTE:
        p=BASE/'remote_queue_delta.json';rows=load(p) if p.exists() else {};rows[tid]=dict(id=tid,parent_task_id=task('K5'),**values);atomic(p,rows)
    else:update(tid,task('K5'),values)
def prepare():
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/n).read_text();refs[n]=ev(AUTO/n)
    BASE.mkdir(exist_ok=False)
    atomic(BASE/'contract.json',dict(created_at=now(),script=ev(__file__),stage_source=ev(stage.__file__),fusion_source=ev(old.__file__),fixed_scoring=ev(fixed.__file__),instructions=refs,science_contract=ev(old.BASE/'pipeline_contract_v3.json'),
        prediction_before_result='Full CoAlign uncertainty-head and fusion-training stochasticity may change graph/no-graph/K1 rankings under either label protocol. Fixed seeds20261103/20261104 before complete first-seed interpretation. Retain every sign; not a new K1/K2/K3 microvariant or a novelty claim for noisy training.',
        seeds=[20261103,20261104],stage1='Fresh original uncertainty model, unchanged fit-only sensor population/12epochs/augmentation/loss/optimizer. Seed alone changes; same seed stage1 shared by both fusion-training arms.',
        fusion='Same author DAIR init/frozen feature extractor/2000x4/AdamW1e-4. Paired PUBLIC and ONLINE inputs, same seed/order/budget. No val or cal model selection.',
        evaluation='val21 only, all4waits; NO_GRAPH/K1_INPUT once before stage1 on4090; FULL once after corresponding stage1 on3090.64condition profile and fixed same-environment replay precede each new evaluator. Sole GT and input ledgers referenced, not copied.',
        schedule=dict(GPU0='currentONLINE evaluation -> seed20261103stage1 -> two correspondingFULL evaluations',GPU1='currentPUBLIC evaluation -> seed20261104stage1 -> two correspondingFULL evaluations',remote4090='current author evaluation -> four paired fusion fits -> four two-arm reference evaluations'),
        cost=dict(stage1_updates=2*25140,stage1_examples=2*100560,fusion_updates=4*2000,fusion_microbatches=4*8000,scientific_fusion_forwards=4*3*13264,profile_fusion_forwards=4*3*64,estimated_wall_h='3-7 bounded parallel chains',new_disk_GiB='<5 final/active rolling and unique predictions; no duplicate GT/raw data'),no_model_wakeup=True))
    for label in ('gpu0','gpu1','remote','sources_local','sources_remote'):record(label,dict(status='registered_finite_successor',contract=ev(BASE/'contract.json')))
def verified():
    c=load(BASE/'contract.json')
    for k in ('script','stage_source','fusion_source','fixed_scoring','science_contract'):assert ev(ROOT/c[k]['path'])==c[k],k
    return c
def replace(source,a,b,n=1):
    assert source.count(a)==n,(a,source.count(a));return source.replace(a,b)
def stage1(seed,gpu):
    ns=dict(stage.__dict__,record=lambda label,values:record('stage1_seed'+str(seed),values),SEED_OUT=BASE/('seed'+str(seed))/'stage1')
    sample=inspect.getsource(stage.sensor_sample)
    sample=replace(sample,'pcd_to_np,shuffle_points','read_pcd,shuffle_points')
    sample=replace(sample,"pcd_to_np(str(ROOT/row['pointcloud']['path']))","read_pcd(str(ROOT/row['pointcloud']['path']))[0]")
    exec(compile(sample,__file__+'::unchanged_DAIR_intensity_reader','exec'),ns)
    source=inspect.getsource(stage.stage1);source=replace(source,"out=BASE/'stage1'",'out=SEED_OUT');source=replace(source,"seed=c['stage1']['seed']",'seed='+str(seed))
    SEED_OUT=ns['SEED_OUT'];SEED_OUT.parent.mkdir(parents=True,exist_ok=True)
    atomic(SEED_OUT.parent/'stage1_execution_identity.json',dict(seed=seed,contract=ev(BASE/'contract.json'),manifest=ev(stage.BASE/'fit_sensor_manifest.json'),original_contract=ev(stage.BASE/'contract.json')))
    exec(compile(source,__file__+'::fixed_budget_stage1_seed','exec'),ns);ns['stage1'](gpu)
def fit(seed,training,gpu):
    dest=BASE/('seed'+str(seed))/(training+'_fit');dest.parent.mkdir(parents=True,exist_ok=True)
    ns=dict(old.__dict__,FIT_OUT=dest,record=lambda label,values:record('fit_'+str(seed)+'_'+training,values))
    source=inspect.getsource(old.fit);source=replace(source,"out=BASE/'fusion_fit'",'out=FIT_OUT');source=replace(source,'seed=20261102','seed='+str(seed))
    if training=='online':
        a="with np.load(ROOT/c['shared_inputs']['fit24_material']['path']) as z:m={k:z[k].copy() for k in ('gt_boxes','gt_offsets')}"
        source=replace(source,a,a+"\n        with np.load(ROOT/c['shared_inputs']['fit_q005_frame']['path']) as z:fitdelta=z['q005_delta'].copy()\n        assert len(fitdelta)==len(rows)")
        a="pc,trans,ident=pair(rows[i],cat,cfg,pre,np);batch=model_input(pc,trans,pre,np,torch,device)"
        source=replace(source,a,"pc,trans,ident=pair(rows[i],cat,cfg,pre,np)\n                if len(trans)>1:trans[1][:2,3]+=fitdelta[i]\n                batch=model_input(pc,trans,pre,np,torch,device)")
    exec(compile(source,__file__+'::fixed_paired_fusion_seed','exec'),ns);ns['fit'](gpu)
    atomic(dest/'seed_training_identity.json',dict(seed=seed,training=training,contract=ev(BASE/'contract.json'),same_init_order_budget=True,weight=ev(dest/'final.pth')))
def fetch_remote_fit(seed,training):
    dest=BASE/('seed'+str(seed))/(training+'_fit');dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--include=terminal.json','--include=final.pth','--include=seed_training_identity.json','--exclude=*','remote-gpu:/path/to/remote_project/'+str(dest.relative_to(ROOT))+'/',str(dest)+'/'],check=True)
    return (dest/'terminal.json').exists()
def evaluate(seed,training,kind,gpu,profile=False):
    dest=BASE/('seed'+str(seed))/(training+'_'+kind)/'evaluation';dest.parent.mkdir(parents=True,exist_ok=True)
    arms=['NO_GRAPH','K1_INPUT'] if kind=='reference' else ['COALIGN_FULL']
    ns=dict(old.__dict__,scorers=fixed.scorers,OUTPUT=dest,ARMS=arms,FUSION_WEIGHT=BASE/('seed'+str(seed))/(training+'_fit/final.pth'),STAGE_WEIGHT=BASE/('seed'+str(seed))/'stage1/final.pth' if kind=='graph' else stage.BASE/'stage1_v2/final.pth',record=lambda label,values:record('eval_'+str(seed)+'_'+training+'_'+kind,values))
    source=inspect.getsource(old.evaluate)
    changes=[("name='adapted_evaluation' if adapted else 'author_evaluation';out=BASE/(name+('_profile' if profile else ''))","name='seed_confirmation';out=OUTPUT.with_name(OUTPUT.name+'_profile') if profile else OUTPUT"),
        ("from run_peer07_native_cvh_source_split_ap_v1_20261031 import polygons,matrix,metric\n        from run_peer10_native_source_aux_v1_20261101 import outcome","polygons,matrix,metric,outcome=scorers()"),
        ("load(BASE/'pipeline_contract.json')","load(BASE/'pipeline_contract_v3.json')"),("ev(BASE/'pipeline_contract.json')","ev(CONFIRMATION_CONTRACT)"),
        ("weight=BASE/'fusion_fit/final.pth' if adapted else stage.ASSETS/'fusion_DAIR/net_epoch_bestval_at13.pth'",'weight=FUSION_WEIGHT'),("stageweight=BASE/'stage1_v2/final.pth'",'stageweight=STAGE_WEIGHT'),
        ("arms=['COALIGN_FULL','NO_GRAPH','K1_INPUT']",'arms=ARMS'),("for j,source in enumerate(ident):","for j,source in (enumerate(ident) if 'COALIGN_FULL' in arms else []):"),
        ("if len(public)>1 and all(len(x) for x in stagecorners):","if 'COALIGN_FULL' in arms and len(public)>1 and all(len(x) for x in stagecorners):"),
        ("**stagecfg['box_align']['args']","**cfg['box_align']['args']"),("corner_to_center(corners.detach().cpu().numpy(),'lwh')","corner_to_center(corners.detach().cpu().numpy(),'hwl')"),
        ("variants=dict(COALIGN_FULL=graph,NO_GRAPH=public,K1_INPUT=k1);gi=", "variants={k:v for k,v in dict(COALIGN_FULL=graph,NO_GRAPH=public,K1_INPUT=k1).items() if k in arms};gi=")]
    for a,b in changes:source=replace(source,a,b)
    ns['CONFIRMATION_CONTRACT']=BASE/'contract.json';exec(compile(source,__file__+'::one_copy_per_scientific_arm','exec'),ns);ns['evaluate'](gpu,True,profile)
    if not profile:atomic(dest/'confirmation_input_identity.json',dict(seed=seed,training=training,arms=arms,contract=ev(BASE/'contract.json'),stage1_used=kind=='graph',shared_stage1_within_seed=True))
def gpu_lane(label,gpu):
    c=verified();out=BASE/label;out.mkdir(exist_ok=False);seed=20261103 if label=='gpu0' else 20261104
    from run_peer12_vit_confirmation_v1_20261102 import wait_unit
    try:
        wait_unit('embodied-peer13-coalign-'+('online-factor' if label=='gpu0' else 'adapted')+'-v4-20261102.service',out)
        stage1(seed,gpu)
        for training in ('public','online'):
            start=time.monotonic()
            while not fetch_remote_fit(seed,training):
                if time.monotonic()-start>18*3600:raise TimeoutError('finite fit wait')
                time.sleep(30)
            assert load(BASE/('seed'+str(seed))/(training+'_fit/terminal.json'))['status']=='training_complete_light_checked'
            evaluate(seed,training,'graph',gpu,True);evaluate(seed,training,'graph',gpu)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),no_model_wakeup=True));record(label,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record(label,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise
def remote_lane(gpu):
    verified();out=BASE/'remote';out.mkdir(exist_ok=False)
    from run_peer12_vit_confirmation_v1_20261102 import wait_unit
    try:
        wait_unit('embodied-peer13-coalign-author-v3-20261102.service',out)
        for seed in (20261103,20261104):
            for training in ('public','online'):fit(seed,training,gpu)
        for seed in (20261103,20261104):
            for training in ('public','online'):
                evaluate(seed,training,'reference',gpu,True);evaluate(seed,training,'reference',gpu)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),no_model_wakeup=True));record('remote',dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('remote',dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gpu0','gpu1','remote']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='remote':remote_lane(a.gpu)
    else:gpu_lane(a.mode,a.gpu)
