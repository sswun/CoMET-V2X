"""Peer23 table-only contracts; scientific experiment freeze remains in force."""
import inspect
from pathlib import Path
from peer22_v1_20261106 import ROOT,AUTO,P19,P20,P21,REMOTE_ROOT,atomic,ev,load,now,update,task
from peer22_v1_20261106 import BASE as P22
import peer14_cycle00_v1_20261103 as launcher
BASE=AUTO/'status_audit_20261106_peer23'
DIRECTIVE=ROOT/'docs/review_20261031/peer23_budget_overfit_and_experiment_freeze.md'

def record(name,values):
    update('peer23_'+name+'_v1_20261106',task('K4' if name.startswith('tier2') else 'K5'),values)

def register(name,script,prediction,cost,**extra):
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    p=BASE/name/'contract.json';assert not p.exists(),p
    atomic(p,dict(created_at=now(),script=ev(Path(script).resolve()),instructions=refs,directive=ev(DIRECTIVE),
        prediction_before_result=prediction,cost=cost,experiment_freeze=True,official_main_unchanged=True,
        selection='cal8 official only; existing R7d contract unchanged',no_hold_off=True,no_model_wakeup=True,
        original_release_closed=False,**extra))
    record(name,dict(status='registered_ready',contract=ev(p),estimated_cost=cost,prediction_before_result=prediction,
        owner=None,resource_owner=False))

def start(name,script,args=()):
    assert not (BASE/'launch_receipts'/(name+'.json')).exists()
    ns=dict(launcher.__dict__,CYCLE=BASE)
    source=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer23-').replace('-v1-20261103','-v1-20261106')
    exec(compile(source,__file__+'::guarded_table_only_launch','exec'),ns)
    unit=ns['start'](name,Path(script).name,args,False,-1,25)+'.service'
    record(name.replace('-','_'),dict(status='running',owner=unit,resource_owner=True))
    return unit
