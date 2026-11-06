"""Gate-authorized R7c: fixed A labels and keyed H1 inputs, two architectures.

Reuse the original K4 trainers, initial weights, loss, optimizer and 2000x4
budget. Only fit label loading and causal pose inputs change. Where has its
own detector's H1; CoAlign uses the strict60 H1 already used by R3. CoAlign is
the native fusion input-replacement arm, NOT a newly trained complete graph
system. Three fixed paired seeds; val21 only, no new selection or adoption.
"""
import argparse, copy, fcntl, hashlib, inspect, json, os, subprocess, sys, time, traceback
from pathlib import Path
from peer21_v1_20261105 import *

OUT=BASE/'R7c'
SEEDS={'where':(20261101,20261102,20261103),'coalign':(20261102,20261103,20261104)}

def key(r):
    return str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms'])

def gate_verified():
    p=BASE/'R7b_gate_postcheck';t=load(p/'terminal.json')
    assert t['status']=='complete_independently_reviewed' and t['AP_rows_verified']==720
    assert t['original_contrasts_verified']==320 and t['both_tier_contrasts']==640
    assert ev(ROOT/t['decision']['path'])==t['decision']
    d=load(p/'decision_verified.json')
    assert d['qualifies'] and all(d['checks'].values()) and d['selected']=='A'
    assert ev(ROOT/d['original_decision']['path'])==d['original_decision']
    for pair in t['sources']:
        for ref in pair.values():assert ev(ROOT/ref['path'])==ref
    return ev(p/'terminal.json')

def pose_map(domain,role,dataset):
    import numpy as np
    root=P19/('recipe_v3_interface_recovery_v2' if domain=='strict60' and role=='val21' else 'recipe_v3')/domain
    path=root/(role+'_state.npz')
    m,rows=dataset[role];keys=[key(r) for r in rows]
    assert keys==[(str(s),int(t),int(w)) for s,t,w in zip(m['sequence'],m['receiver_timestamp_us'],m['wait_ms'])]
    ready=root/(role+'_state_ready.json')
    if ready.exists():assert keys==[tuple(v) for v in load(ready)['keys']]
    with np.load(path) as z:delta=z['delta'].copy()
    assert len(keys)==len(delta)==len(set(keys)) and np.isfinite(delta).all()
    return dict(zip(keys,delta)),dict(state=ev(path),conditions=len(keys),
        key_SHA256=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest())

def prepare():
    import numpy as np
    from run_peer16_cached_methods_v1_20261103 import materials
    import run_peer12_where_native_execution_v6_20261102 as execution
    import run_peer13_coalign_pipeline_v1_20261102 as co
    gate=gate_verified();assert not OUT.exists();OUT.mkdir()
    label=BASE/'R7b/fit24_label_deltas_A.json'
    assert ev(label)==load(BASE/'R7b/A_seed20261101_contract.json')['R7b']['labels']
    scope=dict(gate=gate,selected_labels=ev(label),selection='A frozen by cal8, not reselected per architecture',
        engineering='Only label loader and keyed H1 input substitution; original trainers/model/loss/budget unchanged',
        evaluation='val21 all four waits; official/all/zero/colabelled/nonzero; shared original ROI and GT; 2000 paired sequence draws',
        no_new_adoption_gate=True,not_complete_CoAlign_system_training=True,
        storage='Shared inputs referenced; one final weight and compressed native predictions per new identity; active rolling only')
    identities={}
    for architecture,domain in (('where','where2comm'),('coalign','strict60')):
        data,_,_=materials(domain);by={};identities[architecture]={}
        for role in ('fit24','val21'):
            by[role],identities[architecture][role]=pose_map(domain,role,data)
        if architecture=='where':
            c=copy.deepcopy(load(execution.CYCLE/'where_execution_contract_v6.json'))
            refs=c['inputs']
            for role,material,cond,dest,field in (
                ('fit24','fit24_material','fit_conditions','fit_q005_frame','q005_delta'),
                ('val21','val21_material','val_conditions','val21_gate_material','q005_base_delta')):
                rows=[json.loads(l) for l in (ROOT/refs[cond]['path']).read_text().splitlines()]
                with np.load(ROOT/refs[material]['path']) as z:
                    selected=[rows[int(i)] for i in z['dataset_index']]
                    assert [key(r) for r in selected]==[(str(s),int(t),int(w)) for s,t,w in zip(z['sequence'],z['receiver_timestamp_us'],z['wait_ms'])]
                keys=[key(r) for r in selected];assert len(keys)==len(set(keys)) and set(keys)<=set(by[role])
                if role=='val21':assert set(keys)==set(by[role])
                else:assert {w for s,t,w in keys}=={0,400}
                p=OUT/('where_'+role+'_H1_delta.npz');np.savez_compressed(p,**{field:np.asarray([by[role][k] for k in keys])})
                refs[dest]=ev(p);identities[architecture][role].update(native_rows=len(keys),native_delta=ev(p),all_keys_verified=True)
            for seed in SEEDS['where']:
                spec=copy.deepcopy(c);spec['K4'].update(seed=seed,arms=['COMET_H1'])
                spec['R7c']=scope;atomic(OUT/('where_seed'+str(seed)+'_contract.json'),spec)
        else:
            c=load(co.BASE/'contract.json');rows=co.conditions(c,'fit24',np)
            fitkeys=[key(r) for r in rows];assert len(fitkeys)==len(set(fitkeys)) and set(fitkeys)==set(by['fit24'])
            np.savez_compressed(OUT/'coalign_fit_H1.npz',delta=np.asarray([by['fit24'][k] for k in fitkeys]))
            val=co.conditions(c,'val21',np);valkeys=[key(r) for r in val]
            assert len(valkeys)==len(set(valkeys))==13264 and set(valkeys)==set(by['val21'])
            np.savez_compressed(OUT/'coalign_val_H1.npz',delta=np.asarray([by['val21'][k] for k in valkeys]))
            with np.load(ROOT/c['shared_inputs']['fit24_material']['path']) as z:
                boxes=z['gt_boxes'];offset=z['gt_offsets']
            delta=np.zeros((len(boxes),2),float);labels=load(label)['frames'];cache={};changed=0;maxerr=0.
            labelroot=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label'
            for i,r in enumerate(rows):
                frame=str(r['current_vehicle_frame'])
                if frame not in labels:continue
                if frame not in cache:cache[frame]=load(labelroot/(frame+'.json'))
                objects=cache[frame];centres=np.asarray([[o['3d_location'][k] for k in ('x','y')] for o in objects])
                a,b=map(int,offset[i:i+2])
                for j in range(a,b):
                    dist=np.linalg.norm(centres-boxes[j,:2],axis=1);hits=np.flatnonzero(dist<.001)
                    assert len(hits)==1,('fit_GT_source_identity',frame,j)
                    q=int(hits[0]);maxerr=max(maxerr,float(dist[q]))
                    if objects[q]['from_side']=='inf':delta[j]=labels[frame]['delta'];changed+=1
            assert changed>0 and maxerr<.001
            np.savez_compressed(OUT/'coalign_fit_label_delta.npz',delta=delta)
            identities[architecture].update(fit_rows=len(rows),val_rows=len(val),changed_inf_objects=changed,
                original_GT=ev(ROOT/c['shared_inputs']['fit24_material']['path']),max_source_error_m=maxerr,
                fit_delta=ev(OUT/'coalign_fit_H1.npz'),val_delta=ev(OUT/'coalign_val_H1.npz'),
                label_delta=ev(OUT/'coalign_fit_label_delta.npz'),source_label_SHA256={f:ev(labelroot/(f+'.json')) for f in cache})
        del data
    deps=[ev(ROOT/'preexperiment/research_autopilot'/n) for n in (
        'run_peer12_where_native_execution_v6_20261102.py','run_peer12_where_native_pair_v1_20261102.py',
        'run_peer13_coalign_pipeline_v1_20261102.py','run_peer14_coalign_reference_completion_v1_20261102.py',
        'run_peer20_K4_decontamination_v1_20261105.py','run_peer19_coalign_H1_v1_20261104.py')]
    atomic(OUT/'input_qualification.json',dict(created_at=now(),status='passed_keyed_geometry_and_label_identity',
        architectures=identities,scope=scope,actual_backward_pending=True))
    for arch in ('where','coalign'):
        register('R7c_'+arch,__file__,__doc__,dict(updates=6000,microbatches=24000,
            scientific_forwards=39792,profile_replay='Where 204 minimum; CoAlign 216; recorded separately',
            CPU_threads=2,disk_GiB='<4 per architecture final/active rolling/sole predictions',
            device='local3090GPU1' if arch=='where' else 'remote4090GPU0'),
            **scope,seeds=list(SEEDS[arch]),dependencies=deps,input_qualification=ev(OUT/'input_qualification.json'),
            initial_weight=ev(execution.old.WEIGHT if arch=='where' else co.stage.ASSETS/'fusion_DAIR/net_epoch_bestval_at13.pth'))
    record('R7c_conditional',dict(status='passed_gate_R7c_input_qualified_two_finite_lanes_registered',gate=gate,owner=None,resource_owner=False))

def verified(arch):
    c=load(BASE/('R7c_'+arch)/'contract.json');assert c['script']==ev(__file__)
    for ref in c['dependencies']:assert ev(ROOT/ref['path'])==ref
    assert ev(ROOT/c['initial_weight']['path'])==c['initial_weight']
    assert ev(OUT/'input_qualification.json')==c['input_qualification']
    return c

def where_setup(training=False,actual=None):
    import run_peer12_where_native_execution_v6_20261102 as ex
    import run_peer20_K4_decontamination_v1_20261105 as labels
    ex.setup();original=ex.old.build
    if training:
        delta=load(BASE/'R7b/fit24_label_deltas_A.json')['frames']
        def build(root):
            h,b,roots,c,s=original(root);patched=labels.install_label_loader(b,delta)
            def audited(*args,**kw):
                ds=patched(*args,**kw);assert ds.peer20_label_patch['changed_inf_objects']>0
                atomic(actual/'actual_label_loader.json',dict(created_at=now(),**ds.peer20_label_patch,
                    selected='A',H1_input=True,labels=ev(BASE/'R7b/fit24_label_deltas_A.json')))
                return ds
            return h,audited,roots,c,s
        ex.old.build=build
    return ex

def where_worker(mode,seed,gpu):
    from run_peer21_clean_H1_native_v1_20261105 import environment
    environment();verified('where');out=OUT/'where'/('seed'+str(seed));out.mkdir(parents=True,exist_ok=True)
    ex=where_setup(mode=='train',out)
    contract=OUT/('where_seed'+str(seed)+'_contract.json')
    if mode=='train':
        sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(contract),'--output',str(out/'training'),'--arm','COMET_H1','--gpu',str(gpu)]
        source=ex.old.training_source().replace('ONLINE_K1','COMET_H1')
        ns=dict(__name__='R7c_where_pose_and_label_only',__file__=__file__)
        exec(compile(source,__file__+'::where_train','exec'),ns);ns['main']()
    else:
        from run_peer12_native_profile_v1_20261101 import install_reader,attach_projection_timer
        for profile,dest in ((True,out/'evaluation_profile'),(False,out/'evaluation')):
            sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(contract),'--output',str(dest),
                '--weight',str(out/'training/final.pth'),'--arms','COMET_H1','--gpu',str(gpu)]
            source=ex.old.source_for('where',False,profile).replace('ONLINE_K1','COMET_H1')
            ns=dict(__name__='R7c_where_original_eval',__file__=__file__,install_reader=install_reader,
                attach_projection_timer=attach_projection_timer,cache_enabled=False)
            exec(compile(source,__file__+'::where_eval','exec'),ns);ns['main']()

def coalign_worker(mode,seed,gpu):
    import numpy as np
    import run_peer13_coalign_pipeline_v1_20261102 as co
    verified('coalign');out=OUT/'coalign'/('seed'+str(seed));out.mkdir(parents=True,exist_ok=True)
    if mode=='train':
        with np.load(OUT/'coalign_fit_H1.npz') as z:pose=z['delta'].copy()
        with np.load(OUT/'coalign_fit_label_delta.npz') as z:delta=z['delta'].copy()
        original=co.native_labels
        def labels(m,i,cfg,post,anchors,np):
            a,b=map(int,m['gt_offsets'][i:i+2]);local=dict(m)
            boxes=m['gt_boxes'][a:b].copy();boxes[:,:2]+=delta[a:b]
            local.update(gt_boxes=boxes,gt_offsets=np.asarray([0,len(boxes)]))
            return original(local,0,cfg,post,anchors,np)
        ns=dict(co.__dict__,TRAIN_OUT=out/'training',POSE=pose,native_labels=labels,
            record=lambda k,v:None)
        s=inspect.getsource(co.fit)
        for a,b in (("out=BASE/'fusion_fit'",'out=TRAIN_OUT'),('seed=20261102','seed='+str(seed)),
            ('pc,trans,ident=pair(rows[i],cat,cfg,pre,np);batch=model_input(pc,trans,pre,np,torch,device)',
             "pc,trans,ident=pair(rows[i],cat,cfg,pre,np)\n                if len(trans)>1:trans[1][:2,3]+=POSE[i]\n                batch=model_input(pc,trans,pre,np,torch,device)")):
            assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
        exec(compile(s,__file__+'::coalign_label_and_input_only','exec'),ns);ns['fit'](gpu)
        atomic(out/'training_identity.json',dict(created_at=now(),seed=seed,selected_tier='A',
            no_stage1_retraining=True,not_complete_CoAlign=True,original_trainer_loss_optimizer_unchanged=True,
            input=ev(OUT/'coalign_fit_H1.npz'),labels=ev(OUT/'coalign_fit_label_delta.npz')))
    else:
        import run_peer14_coalign_reference_completion_v1_20261102 as ref
        c=load(co.BASE/'contract.json');rows=co.conditions(c,'val21',np)
        with np.load(OUT/'coalign_val_H1.npz') as z:pose=z['delta'].copy()
        mapping=dict(zip([key(r) for r in rows],pose));assert len(mapping)==13264
        def variants(public,row):
            v=[x.copy() for x in public]
            if len(v)>1:v[1][:2,3]+=mapping[key(row)]
            return dict(COMET_H1=v)
        ns=ref.build('public','slow');ns.update(OUTPUT=out/'evaluation',ARMS=['COMET_H1'],
            FUSION_WEIGHT=out/'training/final.pth',make_variants=variants,
            REFERENCE_CONTRACT=BASE/'R7c_coalign/contract.json',record=lambda k,v:None)
        # Imports used by the CPU source scorer must never hide physical CUDA.
        os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        ns['evaluate'](gpu,True,True);ns['evaluate'](gpu,True,False)
        for d in (out/'evaluation_profile',out/'evaluation'):
            for p in d.glob('replay_*.json'):
                v=load(p);assert v['shape_equal'] and v['max_box_error']==v['max_score_error']==0

def lane(arch,gpu):
    dest=BASE/('R7c_'+arch);lock=(dest/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verified(arch);completed=[]
    if arch=='where':
        from run_peer21_clean_H1_native_v1_20261105 import environment
        environment()
    try:
        record('R7c_'+arch,dict(status='running_gate_authorized_clean_H1_cross_architecture',resource_owner=True))
        for seed in SEEDS[arch]:
            for mode in ('train','eval'):
                command=[sys.executable,'-u',str(Path(__file__).resolve()),mode,'--arch',arch,'--seed',str(seed),'--gpu',str(gpu)]
                atomic(dest/'progress.json',dict(created_at=now(),seed=seed,mode=mode,completed=completed))
                with (dest/(str(seed)+'_'+mode+'.log')).open('x') as log:
                    rc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                assert rc==0,(arch,seed,mode,rc)
                terminal=OUT/arch/('seed'+str(seed))/('training' if mode=='train' else 'evaluation')/'terminal.json'
                value=load(terminal);assert value['status']!='failed_retained'
                completed.append(dict(seed=seed,mode=mode,terminal=ev(terminal)))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',completed=completed,
            independent_AP_CI_pending=True,no_root_table_mutation=True))
        record('R7c_'+arch,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed=completed))
        record('R7c_'+arch,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','lane','train','eval'))
    p.add_argument('--arch',choices=('where','coalign'));p.add_argument('--seed',type=int);p.add_argument('--gpu',type=int,default=1)
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='lane':lane(a.arch,a.gpu)
    elif a.arch=='where':where_worker(a.mode,a.seed,a.gpu)
    else:coalign_worker(a.mode,a.seed,a.gpu)
