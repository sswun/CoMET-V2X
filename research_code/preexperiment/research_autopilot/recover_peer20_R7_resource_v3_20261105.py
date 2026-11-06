"""Resource-interruption recovery, same R7 science, fresh local fixed-budget runs.

The interrupted seed1 lacks CUDA RNG in its resume file; do not claim exact
resumption. Retain it and charge its work separately. Remote seed2 is untouched.
"""
import argparse, inspect, os
from pathlib import Path
from peer20_v1_20261105 import *
import run_peer20_K4_decontamination_v1_20261105 as old

OUT=BASE/'R7_resource_recovery_v3'
NAME='R7_resource_recovery_v3_local'

def prepare():
    assert load(BASE/'R6a_raw_labels_independent_v2/terminal.json')['confirmed']
    register(NAME,__file__,__doc__,dict(GPU_updates=4000,microbatches=16000,
        native_forwards=46032,CPU_threads=2,disk_GiB='<4; fresh finals and compact predictions'),
        interrupted_status=ev(BASE/'R7_environment_v2/seed20261101/training/status.json'),
        resource_abort=ev(BASE/'R7-environment-v2-local_resource/resource_abort.json'),
        discarded_work=load(BASE/'R7_environment_v2/seed20261101/training/status.json'),
        recovery='Restart same seeds from original e60; no partial weight or optimizer reused; no CUDA-RNG equivalence claim',
        initial_contracts=[ev(BASE/'R7_environment_v2'/f'seed{s}_contract.json') for s in (20261101,20261103)],
        remote_seed_untouched=True,science_unchanged=True)
    for s in (20261101,20261103):
        atomic(OUT/f'seed{s}_contract.json',load(BASE/'R7_environment_v2'/f'seed{s}_contract.json'))
    atomic(OUT/'fit24_label_deltas.json',load(BASE/'R7_environment_v2/fit24_label_deltas.json'))

def audited_loader(builder,delta):
    patched=old.install_label_loader(builder,delta)
    def build(*a,**k):
        ds=patched(*a,**k)
        assert ds.peer20_label_patch['changed_inf_objects']>0
        atomic(OUT/'label_loader_actual_audit.json',dict(created_at=now(),
            **ds.peer20_label_patch,source=ev(OUT/'fit24_label_deltas.json'),
            hook='Actual training dataset.coop_label_dict after source-provenance-preserving patch'))
        return ds
    return build

def run(a):
    inherited=[p for p in os.environ.get('PYTHONPATH','').split(os.pathsep)
        if p and 'traf-align-spconv21' not in p and 'compat_traf_numpy124' not in p]
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'preexperiment/research_autopilot/compat_traf_numpy124_thermal_v4'),
        str(ROOT/'.conda/traf-align-spconv21'),*inherited])
    ns=dict(old.__dict__,OUT=OUT,__file__=str(Path(__file__).resolve()),install_label_loader=audited_loader)
    exec(compile(inspect.getsource(old.worker),__file__+'::worker','exec'),ns)
    code=inspect.getsource(old.lane).replace("name='R7_'+which","name='R7_resource_recovery_v3_'+which")
    exec(compile(code,__file__+'::lane','exec'),ns)
    if a.mode=='local':ns['lane']('local',a.gpu)
    else:ns['worker'](a.mode,a.seed,a.output,a.gpu)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','local','train','val','hold'))
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--seed',type=int);p.add_argument('--output',type=Path)
    a=p.parse_args();prepare() if a.mode=='prepare' else run(a)
