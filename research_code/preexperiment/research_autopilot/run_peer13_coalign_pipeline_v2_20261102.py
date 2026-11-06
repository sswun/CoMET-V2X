"""Pre-execution corrections to frozen pipeline v1; no old predictions replaced."""
import argparse, inspect, os, subprocess, time, traceback
import run_peer13_coalign_pipeline_v1_20261102 as old
from peer13_cycle15_v1_20261102 import ROOT,AUTO,CYCLE,atomic,ev,load,now,update,task

BASE=old.BASE

def record(label,values):
    tid='peer13_coalign_pipeline_'+label+'_v2_20261102'
    if old.REMOTE:
        p=BASE/'pipeline_remote_delta_v2.json';rows=load(p) if p.exists() else {}
        rows[tid]=dict(id=tid,parent_task_id=task('K5'),**values);atomic(p,rows)
    else:update(tid,task('K5'),values)

def prepare():
    for name in ('current_handoff.md','main_chat_directive.md'):(AUTO/name).read_text()
    c=dict(load(BASE/'pipeline_contract.json'))
    c.update(created_at=now(),script=ev(__file__),predecessor=ev(BASE/'pipeline_contract.json'),
        static_corrections='v1 never executed. Restore environment after importing CPU scorers; author corner_to_center output hwl matches canonical native shards/scorer/GT; graph options belong to fusion config not stage1 config. GT remains references only.',
        immutable_source=ev(old.__file__),no_model_wakeup=True)
    atomic(BASE/'pipeline_contract_v2.json',c)
    for label in ('author','adapted','author_sources','adapted_sources'):
        record(label,dict(status='registered_finite_successor',contract=ev(BASE/'pipeline_contract_v2.json'),prediction_before_result=c['prediction_before_result'],estimated_cost=c['estimated_cost']))
    for label in ('author_evaluation','fusion_fit_and_evaluation','author_sources','adapted_sources'):
        old.record(label,dict(status='complete_light_checked',executed=False,registration_only=True,superseded_by='peer13_coalign_pipeline_v2_20261102',reason='Static pre-execution corrections; no scientific result.'))

def scorers():
    # The CPU modules intentionally mask CUDA. Importing their pure geometry
    # functions in a GPU worker must not change its CUDA/thread environment.
    keys=('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
    saved={k:os.environ.get(k) for k in keys}
    try:
        from run_peer07_native_cvh_source_split_ap_v1_20261031 import polygons,matrix,metric
        from run_peer10_native_source_aux_v1_20261101 import outcome
    finally:
        for k,v in saved.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    assert all(os.environ.get(k)==v for k,v in saved.items())
    return polygons,matrix,metric,outcome

def namespace():
    c=load(BASE/'pipeline_contract_v2.json')
    assert ev(__file__)==c['script'] and ev(old.__file__)==c['immutable_source']
    ns=dict(old.__dict__,record=record,scorers=scorers)
    source=inspect.getsource(old.evaluate)
    changes=[("from run_peer07_native_cvh_source_split_ap_v1_20261031 import polygons,matrix,metric\n        from run_peer10_native_source_aux_v1_20261101 import outcome","polygons,matrix,metric,outcome=scorers()"),
        ("load(BASE/'pipeline_contract.json')","load(BASE/'pipeline_contract_v2.json')"),
        ("ev(BASE/'pipeline_contract.json')","ev(BASE/'pipeline_contract_v2.json')"),
        ("**stagecfg['box_align']['args']","**cfg['box_align']['args']"),
        ("corner_to_center(corners.detach().cpu().numpy(),'lwh')","corner_to_center(corners.detach().cpu().numpy(),'hwl')")]
    for before,after in changes:
        assert source.count(before)==1,(before,source.count(before));source=source.replace(before,after)
    exec(compile(source,__file__+'::pre_execution_adapter_corrections','exec'),ns)
    return ns

def lane(mode,gpu):
    out=BASE/(mode+'_lane_v2');out.mkdir(exist_ok=False);record(mode,dict(status='waiting_finite_dependency',unit='embodied-peer13-coalign-'+mode+'-v2-20261102.service'))
    try:
        ns=namespace()
        if mode=='adapted':
            from run_peer12_vit_confirmation_v1_20261102 import wait_unit
            wait_unit('embodied-peer13-vit-gpu1-v2-20261102.service',out)
            record(mode,dict(status='running_actual_same_budget_fusion_fit'))
            ns['fit'](gpu)
        ns['wait_stage1'](out)
        record(mode,dict(status='running_full_system_profile_then_three_arm_val21'))
        ns['evaluate'](gpu,mode=='adapted',True)
        ns['evaluate'](gpu,mode=='adapted',False)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),no_model_wakeup=True));record(mode,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as error:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(error),traceback=traceback.format_exc()));record(mode,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

def sources(mode):
    # Reuse canonical five-source fixed-population scorer, replacing only the
    # GT fetch with the immutable per-case reference, never copying GT shards.
    from run_peer12_native_source_additions_v1_20261101 import compute_namespace
    ns,c=compute_namespace();ns['CYCLE']=BASE
    source=inspect.getsource(ns['compute']) if False else None
    import run_peer10_native_source_aux_v1_20261101 as scorer
    s=inspect.getsource(scorer.compute)
    s=s.replace("gt=shard[f'c{idx}_gt']","gt=gtshard[f\"c{r['GT_case']}_gt\"]")
    s=s.replace("with np.load(producer/f'wait{wait:03d}.npz',allow_pickle=False) as shard:","with np.load(producer/f'wait{wait:03d}.npz',allow_pickle=False) as shard, np.load(ROOT/next(r['GT_shard'] for r in records if r['wait_ms']==wait),allow_pickle=False) as gtshard:")
    exec(compile(s,__file__+'::shared_GT_reference_consumer','exec'),ns)
    label=mode+'_sources';out=BASE/(label+'_v2');out.mkdir(exist_ok=False)
    producer=BASE/('adapted_evaluation' if mode=='adapted' else 'author_evaluation');terminal=producer/'terminal.json';parent=BASE/(mode+'_lane_v2/terminal.json')
    try:
        while not terminal.exists():
            if parent.exists() and load(parent)['status']=='failed_retained':raise RuntimeError(('failed producer',str(parent)))
            atomic(out/'status.json',dict(status='waiting_finite_prediction',created_at=now()));time.sleep(30)
        ns['compute'](str(producer.relative_to(old.AUDIT)),c)
        result=BASE/'native_source_aux'/str(producer.relative_to(old.AUDIT))/'terminal.json'
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),scoring=ev(result),no_model_wakeup=True));record(label,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as error:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(error),traceback=traceback.format_exc()));record(label,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','author','adapted','author_sources','adapted_sources']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode.endswith('_sources'):sources(a.mode.split('_')[0])
    else:lane(a.mode,a.gpu)
