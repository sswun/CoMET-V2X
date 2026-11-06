"""R7b bounded TraF A/B labels, exact causal all-ego H1 pose inputs.

Only label loader and keyed pose input replacement change the original native
trainer/evaluator. Fixed e60, 2000 updates, three paired seeds; no model changes.
All six final models are evaluated on cal8 before a global A/B choice, tie A.
No root-table adoption and no R7c until independently confirmed gate.
"""
import argparse, copy, fcntl, hashlib, json, os, subprocess, sys, traceback
from collections import defaultdict
import numpy as np
from peer21_v1_20261105 import *
import run_peer20_K4_decontamination_v1_20261105 as old

OUT=BASE/'R7b';SEEDS=(20261101,20261102,20261103);TIERS=('A','B')
ROLES=('fit24','calibration8','val21','mechanism_holdout14')
RECIPE=P19/'recipe_v3/author'

def key(r):return str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms'])

def label_maps():
    a=load(P20/'R7_resource_recovery_v3/fit24_label_deltas.json')['frames'];b=copy.deepcopy(a)
    frame={r['frame']:r for r in load(P20/'R6a_raw_labels/frames.json') if r['role']=='fit24'}
    groups=defaultdict(list)
    with np.load(P20/'R6a_raw_labels/direct_residuals.npz') as z:
        for ident,res in zip(z['identity'],z['residual_and_offset']):
            role,seq,f,kind,*_=str(ident).split('|')
            if role=='fit24' and kind=='coop':groups[f].append(res[:2].tolist())
    saturated=0
    for f,v in groups.items():
        if len(v)<3:continue
        off=np.asarray(frame[f]['offset']);axes=np.isclose(off,-1.1,atol=1e-8,rtol=0)|np.isclose(off,.66,atol=1e-8,rtol=0)
        if not axes.any():continue
        med=np.median(v,axis=0)
        # Separate order-statistic implementation qualifies the median, not
        # a comparison with the main-session diagnostic output.
        ordered=sorted(x[0] for x in v);ordered_y=sorted(x[1] for x in v);n=len(v)
        manual=np.array([(s[(n-1)//2]+s[n//2])/2 for s in (ordered,ordered_y)])
        assert np.array_equal(med,manual)
        d=np.where(axes,med-off,0.)
        prior=np.asarray(b.get(f,dict(delta=[0.,0.]))['delta'])
        b[f]=dict(delta=(prior+d).tolist(),sequence=frame[f]['sequence'],saturated_axes=axes.tolist(),
            coop_targets=len(v),official=off.tolist(),label_displacement_median=med.tolist(),zero=frame[f]['zero'])
        saturated+=1
    return a,b,dict(A_frames=len(a),B_frames=len(b),saturated_frames=saturated,
        B_axes_only=True,min_coop_targets=3,training_only=True)

def prepare():
    assert load(P20/'R6a_raw_labels_independent_v2/terminal.json')['confirmed']
    assert load(P19/'recipe_independent/author/terminal.json')['status']=='complete_independently_reviewed'
    assert load(RECIPE/'q_calibration.json')['selected_q']==.05
    assert not OUT.exists(),OUT
    OUT.mkdir(parents=True)
    oldc=load(P20/'R7_environment_v2/seed20261102_contract.json')
    from run_peer16_cached_methods_v1_20261103 import materials
    dataset,_,_=materials('author');maps={};identities={}
    for role in ROLES:
        m,rows=dataset[role];keys=[key(r) for r in rows]
        assert keys==[(str(s),int(t),int(w)) for s,t,w in zip(m['sequence'],m['receiver_timestamp_us'],m['wait_ms'])]
        with np.load(RECIPE/(role+'_state.npz')) as z:d=z['delta'].copy()
        with np.load(P19/'recipe_v3_interface_recovery_v2/author'/(role+'_state.npz')) as z:assert np.array_equal(d,z['delta'])
        assert len(d)==len(keys)==len(set(keys)) and np.isfinite(d).all()
        if role in ('val21','mechanism_holdout14'):
            assert keys==[tuple(k) for k in load(RECIPE/(role+'_state_ready.json'))['keys']]
        maps[role]=dict(zip(keys,d));identities[role]=dict(state=ev(RECIPE/(role+'_state.npz')),conditions=len(keys),
            key_digest=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest())
    del dataset
    # Each native input uses the original native catalogue/global_index. H1 is
    # joined by all three temporal keys; never borrow another detector's index.
    refs=copy.deepcopy(oldc['inputs'])
    trainrows=[json.loads(l) for l in (ROOT/refs['train_conditions']['path']).read_text().splitlines()]
    for role,material_key,conditions_key,delta_key,array_name in (
        ('fit24','fit24_material','fit_conditions','fit_q005_frame','q005_delta'),
        ('val21','val21_material','val_conditions','val21_gate_material','q005_base_delta'),
        ('mechanism_holdout14','mechanism_holdout14_material','train_conditions','mechanism_holdout14_gate_material','q005_base_delta'),
        ('calibration8','calibration8_material','train_conditions','calibration8_gate_material','q005_base_delta')):
        if role=='calibration8':
            ids=[i for i,r in enumerate(trainrows) if r['role']==role]
            rows=[trainrows[i] for i in ids];assert len(rows)==4988
            path=OUT/'calibration8_native_identity.npz'
            np.savez_compressed(path,sequence=np.asarray([r['sequence'] for r in rows]),
                receiver_timestamp_us=np.asarray([r['receiver_timestamp_us'] for r in rows]),
                wait_ms=np.asarray([r['wait_ms'] for r in rows]),dataset_index=np.asarray(ids))
            refs[material_key]=ev(path)
        else:
            cond=[json.loads(l) for l in (ROOT/refs[conditions_key]['path']).read_text().splitlines()]
            with np.load(ROOT/refs[material_key]['path']) as z:
                rows=[cond[int(i)] for i in z['dataset_index']]
                assert [key(r) for r in rows]==[(str(s),int(t),int(w)) for s,t,w in zip(z['sequence'],z['receiver_timestamp_us'],z['wait_ms'])]
        keys=[key(r) for r in rows];assert len(keys)==len(set(keys)) and set(keys).issubset(maps[role])
        if role!='fit24':assert set(keys)==set(maps[role])
        else:assert set(w for s,t,w in keys)=={0,400}
        path=OUT/(role+'_H1_native_delta.npz');np.savez_compressed(path,**{array_name:np.asarray([maps[role][k] for k in keys])})
        refs[delta_key]=ev(path);identities[role].update(native_conditions=len(keys),native_delta=ev(path),native_keys_checked=True)
    a,b,labelinfo=label_maps()
    for tier,delta in zip(TIERS,(a,b)):
        atomic(OUT/('fit24_label_deltas_'+tier+'.json'),dict(created_at=now(),frames=delta,roles=['fit24'],
            raw_label_source=ev(P20/'R6a_raw_labels/direct_residuals.npz'),noncausal_label_only=True,tier=tier))
        for seed in SEEDS:
            c=copy.deepcopy(oldc);c['inputs']=refs;c['K4']['seed']=seed;c['K4']['arms']=['COMET_H1']
            c['R7b']=dict(tier=tier,input='Causal all-ego CoMET H1, author q=.05',
                original_budget_contract=ev(P20/'R7_environment_v2/seed20261102_contract.json'),
                pose_identity=identities,labels=ev(OUT/('fit24_label_deltas_'+tier+'.json')),
                train=ev(old.TRAIN),evaluate=ev(old.EVAL),fixed_source_substitutions="ONLINE_K1 -> COMET_H1; hold/cal role+identity substitutions only")
            atomic(OUT/(tier+'_seed'+str(seed)+'_contract.json'),c)
    atomic(OUT/'input_qualification.json',dict(created_at=now(),status='passed_keyed_input_and_label_contract',
        roles=identities,labels=labelinfo,initial_weight=oldc['K4']['initial_weight'],
        actual_backward_pending=True,GT_inference_forbidden=True,official_main_GT_unchanged=True))
    for lane,seeds in [('local',(SEEDS[0],SEEDS[2])),('remote',(SEEDS[1],))]:
        register('R7b_'+lane,__file__,__doc__,dict(GPU_updates=2000*2*len(seeds),microbatches=8000*2*len(seeds),
            native_forwards=2*len(seeds)*(4988+13264+9744+12),CPU_threads=2,
            disk_GiB='<2 per final model plus compact predictions; only active resume retained',
            expected='all6 models 12000 updates/48000 microbatches; no baseline reruns'),
            seeds=list(seeds),tiers=list(TIERS),qualification=ev(OUT/'input_qualification.json'),
            preregistered_gate='val21 selected tier: PUBLIC AP50 and AP70 paired2000 CI low>0; H1 AP50>=+1 and CI low>0, AP70>=-.30; hold14 report only',
            tier_selection='cal8 official AP70 mean over3seeds/4waits; exact tie A; before val/hold consumption',
            inference_input='COMET_H1 (not ONLINE_K1); no GT or label-median inference input',
            environment='existing spconv2.1 and thermal-compatible numpy shim',
            immutable_dependencies=[ev(old.TRAIN),ev(old.EVAL),ev(Path(old.__file__))])
    for name in ('R7b_collector','R7b_statistics','R7b_independent','R7b_gate','R7c_conditional'):
        register(name,__file__,'Known dependency consumer, R7c only after independent R7b adoption gate; no automatic main-table mutation.',
            dict(GPU_updates=0,CPU_threads=1,replicates=2000,disk_MiB='<250 compact AP/CI; raw predictions shared'),
            status_note='implementation_and_dispatch_pending_not_completed',dependencies=['R7b_local','R7b_remote'])
        record(name,dict(status='planned_dependency_implementation_pending',owner=None,resource_owner=False))

def corrected_source(mode):
    code=(old.TRAIN if mode=='train' else old.EVAL).read_text()
    assert 'ONLINE_K1' in code;code=code.replace('ONLINE_K1','COMET_H1')
    if mode in ('hold','cal'):
        import run_peer10_k4_hold14_native_v1_20261101 as hold
        for a,b in hold.CHANGES:
            if mode=='cal':b=b.replace('mechanism_holdout14','calibration8').replace('9744','4988')
            assert code.count(a)==1,(a,code.count(a));code=code.replace(a,b)
    return code

def worker(a):
    cpath=OUT/(a.tier+'_seed'+str(a.seed)+'_contract.json');c=load(cpath)
    for field in ('train','evaluate'):
        ref=c['R7b'][field];assert ev(ROOT/ref['path'])==ref
    if a.mode=='train':
        import qualify_traf_native4_cpu_v2_20261010 as qual
        original=qual.build;ref=c['R7b']['labels'];assert ev(ROOT/ref['path'])==ref
        delta=load(ROOT/ref['path'])['frames']
        def build(root):
            h,b,roots,cc,s=original(root);patched=old.install_label_loader(b,delta)
            def audited(*args,**kw):
                ds=patched(*args,**kw);assert ds.peer20_label_patch['changed_inf_objects']>0
                atomic(OUT/a.tier/('seed'+str(a.seed))/'actual_label_loader.json',dict(
                    created_at=now(),**ds.peer20_label_patch,labels=ref,tier=a.tier,H1_input=True))
                return ds
            return h,audited,roots,cc,s
        qual.build=build
        sys.argv=[str(old.TRAIN),'--workspace',str(ROOT),'--contract',str(cpath),'--output',str(a.output),'--arm','COMET_H1','--gpu',str(a.gpu)]
    else:
        sys.argv=[str(old.EVAL),'--workspace',str(ROOT),'--contract',str(cpath),'--output',str(a.output),
            '--weight',str(OUT/a.tier/('seed'+str(a.seed))/'training/final.pth'),'--arms','COMET_H1','--gpu',str(a.gpu)]
    code=corrected_source(a.mode);ns=dict(__name__='peer21_frozen_native_pose_label_only',__file__=str(old.TRAIN if a.mode=='train' else old.EVAL))
    exec(compile(code,__file__+'::'+a.mode,'exec'),ns);ns['main']()

def environment():
    inherited=[p for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p and 'traf-align-spconv21' not in p and 'compat_traf_numpy124' not in p]
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'preexperiment/research_autopilot/compat_traf_numpy124_thermal_v4'),str(ROOT/'.conda/traf-align-spconv21'),*inherited])

def lane(which,gpu):
    name='R7b_'+which;dest=BASE/name;dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    results=[];seeds=(SEEDS[0],SEEDS[2]) if which=='local' else (SEEDS[1],)
    try:
        record(name,dict(status='running_H1_clean_native_fixed_budget',resource_owner=True))
        # Interleave A/B per seed. All fixed runs stay in this finite chain;
        # each cal evaluation precedes its formal runs, no result-based choice.
        for seed in seeds:
            for tier in TIERS:
                for mode,folder in [('train','training'),('cal','calibration8'),('val','val21'),('hold','mechanism_holdout14')]:
                    out=OUT/tier/('seed'+str(seed))/folder;out.parent.mkdir(parents=True,exist_ok=True)
                    cmd=[str(ROOT/'.conda/opencood-linux/bin/python'),'-u',str(Path(__file__).resolve()),mode,
                        '--tier',tier,'--seed',str(seed),'--output',str(out),'--gpu',str(gpu)]
                    atomic(dest/'progress.json',dict(created_at=now(),seed=seed,tier=tier,mode=mode,completed=results,command=cmd))
                    with (dest/(tier+'_'+str(seed)+'_'+mode+'.log')).open('x') as log:
                        rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                    assert rc==0,(tier,seed,mode,rc)
                    results.append(dict(tier=tier,seed=seed,mode=mode,terminal=ev(out/'terminal.json')))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',completed=results,
            independent_AP_CI_and_global_cal_choice_pending=True,automatic_adoption=False))
        record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed=results))
        record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','local','remote','train','cal','val','hold'))
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--tier',choices=TIERS);p.add_argument('--gpu',type=int,default=1);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:
        environment()
        lane(a.mode,a.gpu) if a.mode in ('local','remote') else worker(a)
