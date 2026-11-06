"""Versioned sparse-material adapter; do not invent absent box variants.

All present stale/cv/huber/supervised_constant/kf remote arrays are shifted
together. stale and cv are required by the ACTUAL heading/readout. Some formal
material schemas omit unused historical variants. Full-schema injection is
bitwise identical to v1; no estimator parameter or adoption gate changes.
"""
import argparse,inspect,subprocess,sys,time,types
from pathlib import Path
from peer26_v1_20261106 import *
NATIVE=BASE/'R8c_native_material_v2'
DOUT=BASE/'R8d_material_v2';DREVIEW=BASE/'R8d_material_v2_independent'

def inject(m,kind,amplitude=0.,seed=2026111401):
    import peer26_R8_estimators_v1_20261106 as old
    assert 'stale_boxes' in m and 'cv_boxes' in m
    # Exact original algorithm with only its invalid schema assertion removed.
    source=inspect.getsource(old.inject).replace('assert key in m,key','if key not in m:continue')
    ns=dict(old.__dict__);exec(compile(source,__file__+'::all_present_arrays','exec'),ns)
    return ns['inject'](m,kind,amplitude,seed)

def adapter():
    import peer26_R8_estimators_v1_20261106 as old
    return types.SimpleNamespace(**dict(old.__dict__,inject=inject))

def prepare():
    import numpy as np
    import test_peer26_R8_v1_20261106 as tests
    import peer26_R8_estimators_v1_20261106 as old
    m=tests.TestR8().material();a,d=old.inject(m,'T1',3.);b,e=inject(m,'T1',3.)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    sparse={k:v for k,v in m.items() if k not in ('huber_boxes','supervised_constant_boxes')}
    c,f=inject(sparse,'T1',3.);assert set(c)==set(sparse)
    assert all(np.array_equal(c[k],a[k]) for k in sparse)
    for name in ('R8c_native_inputs_material_v2','R8d_material_v2_chain','material_adapter_v2'):
        register(name,__file__,__doc__,dict(CPU_threads=1,GPU_forwards=0,updates=0,replicates=2000,
            disk_MiB='<5 adapter metadata; same original compact scientific outputs in new versioned namespaces',wall_h='same fixed bounded original scope'),
            original_injector=ev(old.__file__),actual_consumer=ev(ROOT/'preexperiment/research_autopilot/run_crosscheckpoint_current_q005_heading_geometry_peer06_v1_20261030.py'),
            original_native_failure=ev(BASE/'R8c_native_inputs/terminal.json'),
            no_optional_variant_fabrication=True,mandatory=['stale_boxes','cv_boxes'],
            estimator_parameters_unchanged=True,engineering_compatibility_only=True,no_val_AP_used_for_adjustment=True)
    atomic(BASE/'material_adapter_v2/engineering_qualification.json',dict(created_at=now(),status='passed',tests=3,
        full_schema_all_arrays_bitwise=True,sparse_schema_all_present_arrays_exact=True,no_absent_arrays_invented=True,
        algorithm_adjustments=0,material_compatibility_fix=1,not_scientific_adoption=True))
    NATIVE.mkdir(exist_ok=False)

def native_namespace():
    import run_peer26_R8c_native_v1_20261106 as original
    ns=dict(original.__dict__,OUT=NATIVE,__file__=__file__)
    for name in ('jobs','key','forward','lane','dispatch'):
        source=inspect.getsource(getattr(original,name))
        if name=='dispatch':
            source=source.replace("BASE/'R8c_native_inputs/terminal.json'","BASE/'R8c_native_inputs_material_v2/terminal.json'")
            source=source.replace('files=[Path(__file__).resolve(),',
                "files=[Path(__file__).resolve(),ROOT/'preexperiment/research_autopilot/run_peer26_R8c_native_v1_20261106.py',")
        exec(compile(source,__file__+'::native_'+name,'exec'),ns)
    return ns

def native_inputs():
    import run_peer26_R8c_native_v1_20261106 as original
    import peer26_R8_estimators_v1_20261106 as old
    ns=native_namespace();source=inspect.getsource(original.inputs)
    source=source.replace('import peer26_R8_estimators_v1_20261106 as model','model=ADAPTER')
    source=source.replace("'R8c_native_inputs'","'R8c_native_inputs_material_v2'")
    source=source.replace("BASE/'R8c_native_inputs/terminal.json'","BASE/'R8c_native_inputs_material_v2/terminal.json'")
    ns['ADAPTER']=adapter();exec(compile(source,__file__+'::native_inputs','exec'),ns);ns['inputs']()

def new_case(*args,**kwargs):
    import run_peer26_R8d_v1_20261106 as original
    # Recompile with the caller's current output namespace, not a hot edit of
    # any already-running process or frozen scientific script.
    ns=dict(original.__dict__,model=adapter())
    source=inspect.getsource(ORIGINAL_CASE)
    source=source.replace('simultaneous_injection_keys=list(model.BOX_KEYS)',
        'simultaneous_injection_keys=[k for k in model.BOX_KEYS if k in m],absent_unused_variant_keys=[k for k in model.BOX_KEYS if k not in m]')
    exec(compile(source,__file__+'::same_case_sparse_schema','exec'),ns);return ns['case'](*args,**kwargs)

ORIGINAL_CASE=None

def d_chain():
    global ORIGINAL_CASE
    import run_peer26_R8d_v1_20261106 as original
    ORIGINAL_CASE=original.case;oldout=original.OUT
    # Do not interrupt the fixed v1 run. If it completes there is no recovery.
    while not (BASE/'R8d_chain/terminal.json').exists():time.sleep(30)
    oldterm=load(BASE/'R8d_chain/terminal.json')
    if oldterm['status']!='failed_retained':
        atomic(BASE/'R8d_material_v2_chain/terminal.json',dict(created_at=now(),status='complete_light_checked',
            recovery_not_needed=True,original=ev(BASE/'R8d_chain/terminal.json')));return
    assert 'huber_boxes' in str(oldterm) or 'supervised_constant_boxes' in str(oldterm) or 'kf_boxes' in str(oldterm),oldterm
    def shared_real():
        assert load(oldout/'real_state_terminal.json')['status']=='complete_independently_reviewed'
        for domain in ('author','strict60','v2xvit_seed2024','v2xvit_seed2025','where2comm'):
            for role in ('calibration8','val21','mechanism_holdout14'):
                p=oldout/domain/(role+'_state_identity.json');atomic(DOUT/domain/p.name,dict(load(p),shared_original=ev(p)))
        atomic(DOUT/'real_state_terminal.json',dict(load(oldout/'real_state_terminal.json'),shared_original=ev(oldout/'real_state_terminal.json')))
    def case(*args,**kwargs):
        domain,role,kind,amp=args[:4];real=kwargs.get('real',False) or (len(args)>4 and args[4])
        name=domain+'_'+role+'_'+('real' if real else kind+'_'+str(amp));p=oldout/'cases'/name
        if (p/'terminal.json').exists() and load(p/'terminal.json')['status']=='complete_independently_reviewed':
            target=DOUT/'cases'/name;atomic(target/'summary.json',load(p/'summary.json'))
            atomic(target/'shared_case.json',dict(directory=str(p),terminal=ev(p/'terminal.json')))
            atomic(target/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',shared_original=ev(p/'terminal.json')))
            return load(p/'summary.json')
        return new_case(*args,**kwargs)
    original.OUT=DOUT;original.REVIEW=DREVIEW
    ns=dict(original.__dict__,model=adapter(),real_states=shared_real,case=case,
        record=lambda n,v:record('R8d_material_v2_chain' if n=='R8d_chain' else 'R8d_material_v2',v))
    exec(compile(inspect.getsource(original.gate),__file__+'::same_gate','exec'),ns)
    source=inspect.getsource(original.chain).replace("dest=BASE/'R8d_chain'","dest=BASE/'R8d_material_v2_chain'")
    exec(compile(source,__file__+'::same_chain_new_namespace','exec'),ns);ns['chain']()

def late(domain):
    global ORIGINAL_CASE
    import run_peer26_R8c_late_v1_20261106 as original
    import run_peer26_R8d_v1_20261106 as d
    ORIGINAL_CASE=d.case;d.OUT=DOUT;d.REVIEW=DREVIEW;d.model=adapter();d.case=new_case
    ns=dict(original.__dict__,d=d,model=adapter())
    for name in ('add_motion','recovery'):
        exec(compile(inspect.getsource(getattr(original,name)),__file__+'::'+name,'exec'),ns)
    source=inspect.getsource(original.run)
    source=source.replace("base=shared\n", "base=Path(load(shared/'shared_case.json')['directory']) if (shared/'shared_case.json').exists() else shared\n")
    exec(compile(source,__file__+'::late_sparse_schema','exec'),ns);ns['run'](domain)

def consume(mode):
    import consume_peer26_R8c_native_v1_20261106 as original
    original.native_run.OUT=NATIVE
    original.collect() if mode=='collect' else original.statistics()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','inputs','d_chain','late','dispatch','local','remote','forward','collect','statistics'))
    p.add_argument('--domain');p.add_argument('--case');p.add_argument('--gpu',type=int,default=1);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='inputs':native_inputs()
    elif a.mode=='d_chain':d_chain()
    elif a.mode=='late':late(a.domain)
    elif a.mode in ('collect','statistics'):consume(a.mode)
    else:
        ns=native_namespace()
        ns['dispatch']() if a.mode=='dispatch' else ns['forward'](a) if a.mode=='forward' else ns['lane'](a.mode,a.gpu)
