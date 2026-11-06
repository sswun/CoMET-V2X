"""Peer15 registrations: paper assets under CoMET, never model event wakes."""
import inspect,shlex,subprocess
from peer14_cycle12_v1_20261103 import ROOT,AUTO,AUDIT,CYCLE as C12,C09,C06,C21,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher
BASE=AUTO/'status_audit_20261103_peer15'

def record(name,values):
    tid='peer15_'+name+'_v1_20261103'
    if ROOT==REMOTE_ROOT:
        p=BASE/'remote_queue_delta.json';d=load(p) if p.exists() else {};d[tid]=dict(id=tid,parent_task_id=task('K1'),**values);atomic(p,d)
    else:update(tid,task('K1'),values)

def register(name,script,prediction,cost,**extras):
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/n).read_text();refs[n]=ev(AUTO/n)
    p=BASE/name/'contract.json';assert not p.exists(),p
    atomic(p,dict(created_at=now(),script=ev(script),instructions=refs,prediction_before_result=prediction,cost=cost,no_model_wakeup=True,official_main_unchanged=True,**extras))
    record(name,dict(status='registered_ready',contract=ev(p),estimated_cost=cost,prediction_before_result=prediction))

def start(name,script,args=(),remote=False,gpu=-1):
    assert '/' not in script
    ns=dict(launcher.__dict__,CYCLE=BASE)
    src=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer15-')
    exec(compile(src,__file__+'::launch','exec'),ns)
    return ns['start'](name,script,args,remote,gpu,25)

if __name__=='__main__':
    BASE.mkdir(exist_ok=True)
    assert not (BASE/'entry.json').exists()
    atomic(BASE/'entry.json',dict(created_at=now(),handoff=ev(AUTO/'current_handoff.md'),directive=ev(AUTO/'main_chat_directive.md')))
    atomic(BASE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
