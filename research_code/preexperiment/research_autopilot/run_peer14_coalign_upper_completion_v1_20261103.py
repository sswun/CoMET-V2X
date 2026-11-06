"""CoAlign unmeasured annotation ceilings and a matched oracle-training control.

No new estimator, no deployment claim for annotation inputs, val21 only.
"""
import argparse,inspect,time,traceback,subprocess
from peer14_cycle00_v1_20261103 import ROOT,AUDIT,CYCLE,C21,C15,C18,atomic,ev,load,now,register,record
import run_peer14_coalign_reference_completion_v1_20261102 as ref
BASE=CYCLE/'coalign_upper_v1'

def fields():
    import numpy as np
    c=load(ref.confirm.old.BASE/'contract.json');rows=ref.confirm.old.conditions(c,'val21',np)
    with np.load(ROOT/c['shared_inputs']['val21_gate_material']['path']) as z:delta=z['q005_base_delta'].copy()
    online={(str(r['current_vehicle_frame']),int(r['wait_ms'])):d for r,d in zip(rows,delta)}
    meta=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json'
    official={str(r['vehicle_frame']):np.array([r['system_error_offset'][k] for k in ('delta_x','delta_y')]) for r in load(meta)}
    ref.SLOW={k:{f:np.array(v) for f,v in x.items()} for k,x in load(ref.BASE/'slow_offsets_frozen.json')['fields'].items()}
    return official,online

def build(weight,out,mode):
    import numpy as np
    official,online=fields();ns=ref.build('public','ego' if mode=='ego' else 'slow')
    arms={'official':['OFFICIAL_REFERENCE'],'factor':['PUBLIC_INPUT','ONLINE_K1_INPUT','OFFICIAL_REFERENCE'],'slow':['OFFICIAL_SEQMEAN_REFERENCE','OFFICIAL_MA5_REFERENCE'],'ego':['EGO_ONLY_REFERENCE']}[mode]
    def variants(public,row):
        if mode=='ego':return dict(EGO_ONLY_REFERENCE=[np.eye(4)])
        if mode=='slow':return ref.variants(public,row)
        frame=str(row['current_vehicle_frame']);result={}
        for arm in arms:
            v=[x.copy() for x in public]
            if len(v)>1 and arm!='PUBLIC_INPUT':v[1][:2,3]+=official[frame] if arm=='OFFICIAL_REFERENCE' else online[frame,int(row['wait_ms'])]
            result[arm]=v
        return result
    ns.update(OUTPUT=out,FUSION_WEIGHT=weight,REFERENCE_CONTRACT=CYCLE/'coalign_upper_contract.json',ARMS=arms,make_variants=variants)
    return ns

def prepare():
    BASE.mkdir(exist_ok=False)
    register('coalign_upper',__file__,'Complete unmeasured full-offset ceiling to distinguish slow/full annotation dependence in CoAlign, then quantify its seed sensitivity. Same saved models; no new estimator. Separately one OFFICIAL-trained first-seed same-budget fusion control is annotation-assisted training analysis, not novel/noisy method or deployable oracle. No sign/peak selection.',dict(local_forwards_per_lane=99480,remote_new_training_updates=2000,remote_microbatches=8000,remote_scientific_forwards=82900,profile_replay='64 per evaluator plus original same-environment replay',estimated_local_h='3-5 from actual ~29844/h, finite',estimated_remote_h='3-5 after current sole owner and author slow chain',new_disk_GiB='<3 unique compressed predictions/final and active rolling; no GT/raw copies'),reference_adapter=ev(ref.__file__),original_fusion=ev(ref.confirm.old.__file__),confirmation=ev(ref.confirm.__file__),input_contract=ev(ref.confirm.old.BASE/'pipeline_contract_v3.json'),offset_metadata=ev(ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json'),slow=ev(ref.BASE/'slow_offsets_frozen.json'),scope='CoAlign sharedTraF GT/own prediction ROI internal references, common/source CPU downstream. Reference-only skips stage1 and graph, no complete-system claim. Main official protocol untouched; inference label inputs ONLY named nondeployable reference arms.')
    plans={}
    for training in ('public','online'):
        weight=C15/'coalign_full_v1'/('fusion_fit/final.pth' if training=='public' else 'online_fusion_fit/final.pth')
        plan=[dict(name='first_'+training+'_official',weight=str(weight.relative_to(ROOT)),modes=['official'])]
        for seed in (20261103,20261104):plan.append(dict(name=f'seed{seed}_{training}',weight=str((ref.confirm.BASE/('seed'+str(seed))/(training+'_fit/final.pth')).relative_to(ROOT)),modes=['ego','slow','official']))
        plans[training]=plan
        record('upper_'+training,dict(status='registered_ready',contract=ev(CYCLE/'coalign_upper_contract.json')))
    c=load(CYCLE/'coalign_upper_contract.json');c['plans']=plans;c['first_weights']={t:ev(ROOT/plans[t][0]['weight']) for t in plans};c['author_weight']=ev(ref.confirm.stage.ASSETS/'fusion_DAIR/net_epoch_bestval_at13.pth');atomic(CYCLE/'coalign_upper_contract.json',c)
    for n in ('upper_remote','upper_sources','upper_collect'):record(n,dict(status='registered_finite_successor',contract=ev(CYCLE/'coalign_upper_contract.json')))
    # Compile all shapes without CUDA; not a model qualification.
    checks={m:callable(build(ROOT/plans['public'][0]['weight'],BASE/'build_only'/m,m)['evaluate']) for m in ('official','factor','slow','ego')};atomic(BASE/'build_tests.json',dict(checks=checks,GPU_forwards=0));assert all(checks.values())

def evaluate(weight,out,mode,gpu):
    ns=build(weight,out,mode)
    for profile in (True,False):ns['evaluate'](gpu,True,profile)
    atomic(out/'reference_identity.json',dict(weight=ev(weight),mode=mode,contract=ev(CYCLE/'coalign_upper_contract.json'),stage1_calls=0,graph_calls=0,not_complete_system=True,annotation_reference_only=mode!='ego'))

def verified():
    c=load(CYCLE/'coalign_upper_contract.json')
    for k in ('script','reference_adapter','original_fusion','confirmation','input_contract','offset_metadata','slow'):assert ev(ROOT/c[k]['path'])==c[k],k
    return c

def lane(training,gpu):
    c=verified();out=BASE/training;out.mkdir();start=time.monotonic();record('upper_'+training,dict(status='running_coalign_fixed_weight_ceilings'))
    try:
        for p in c['plans'][training]:
            weight=ROOT/p['weight'];assert weight.exists()
            for mode in p['modes']:evaluate(weight,out/p['name']/mode/'evaluation',mode,gpu)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),wall_s=time.monotonic()-start,no_model_wakeup=True));record('upper_'+training,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('upper_'+training,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

def official_fit(gpu):
    old=ref.confirm.old;dest=BASE/'official_fit';ns=dict(old.__dict__,FIT_OUT=dest,record=lambda label,v:None)
    s=inspect.getsource(old.fit);s=ref.replace(s,"out=BASE/'fusion_fit'",'out=FIT_OUT')
    a="pc,trans,ident=pair(rows[i],cat,cfg,pre,np);batch=model_input(pc,trans,pre,np,torch,device)"
    b="pc,trans,ident=pair(rows[i],cat,cfg,pre,np)\n                if len(trans)>1:trans[1][:2,3]+=OFFICIAL[str(rows[i]['current_vehicle_frame'])]\n                batch=model_input(pc,trans,pre,np,torch,device)"
    s=ref.replace(s,a,b);ns['OFFICIAL']=fields()[0];exec(compile(s,__file__+'::same_budget_official_input_fit','exec'),ns);ns['fit'](gpu)
    atomic(dest/'training_identity.json',dict(seed=20261102,training_input='OFFICIAL_REFERENCE_ANNOTATION_ASSISTED',contract=ev(CYCLE/'coalign_upper_contract.json'),same_init_order_budget_as_first_PUBLIC_ONLINE=True,weight=ev(dest/'final.pth'),not_new_method=True))

def remote(gpu):
    c=verified();out=BASE/'remote';out.mkdir();start=time.monotonic();record('upper_remote',dict(status='waiting_sole_author_reference_owner'))
    try:
        unit='embodied-peer14-coalign-author-reference-parent-v2-v1-20261102.service'
        while subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState'],text=True).strip() in ('ActiveState=active','ActiveState=activating'):
            if time.monotonic()-start>18*3600:raise TimeoutError('finite18h prior owner')
            time.sleep(30)
        evaluate(ROOT/c['author_weight']['path'],BASE/'author_official/evaluation','official',gpu)
        official_fit(gpu);weight=BASE/'official_fit/final.pth'
        for mode in ('factor','ego','slow'):evaluate(weight,BASE/'official_trained'/mode/'evaluation',mode,gpu)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),wall_s=time.monotonic()-start,no_model_wakeup=True));record('upper_remote',dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('upper_remote',dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','public','online','remote']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();prepare() if a.mode=='prepare' else remote(a.gpu) if a.mode=='remote' else lane(a.mode,a.gpu)
