"""Prepare nested reference output parents; original frozen science unchanged."""
import argparse
from peer14_cycle00_v1_20261103 import CYCLE,ROOT,atomic,ev,load,now,register,record
import run_peer14_coalign_upper_completion_v1_20261103 as original
BASE=CYCLE/'coalign_upper_parent_recovery_v2'
BUILD=original.build

def build(weight,out,mode):
    out.parent.mkdir(parents=True,exist_ok=True)
    return BUILD(weight,out,mode)

def prepare():
    BASE.mkdir(exist_ok=False)
    register('coalign_upper_parent_recovery',__file__,'Repair only pre-CUDA missing nested output parents. Original reference transforms, same models, fit order/budget and evaluation unchanged. Fresh outputs preserve zero-science v1 failures.',dict(GPU_forwards='original frozen upper contract, no extra scientific repeats',CPU_threads='GPU2',new_disk_GiB='original planned unique outputs'),original=ev(original.__file__),original_contract=ev(CYCLE/'coalign_upper_contract.json'))
    for t in ('public','online','remote'):record('parent_recovery_upper_'+t,dict(status='registered_ready',contract=ev(CYCLE/'coalign_upper_parent_recovery_contract.json')))
    for t in ('public','online'):
        p=original.BASE/t/'terminal.json';v=load(p);assert v['status']=='failed_retained' and 'FileNotFoundError' in v['error']
    atomic(BASE/'bootstrap_repair.json',dict(created_at=now(),original_failures={t:ev(original.BASE/t/'terminal.json') for t in ('public','online')},extra_GPU_forwards=0,original_scientific_script_unchanged=True))
    record('coalign_upper',dict(status='complete_light_checked',administrative_contract_only=True,scientific_children_open=True))
    record('coalign_upper_parent_recovery',dict(status='complete_light_checked',administrative_contract_only=True,scientific_children_open=True))

def run(t,gpu):
    c=load(CYCLE/'coalign_upper_parent_recovery_contract.json');assert c['script']==ev(__file__) and c['original']==ev(original.__file__) and c['original_contract']==ev(CYCLE/'coalign_upper_contract.json')
    original.BASE=BASE;original.build=build;original.record=lambda n,v:record('parent_recovery_'+n,v)
    original.remote(gpu) if t=='remote' else original.lane(t,gpu)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','public','online','remote']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode,a.gpu)
