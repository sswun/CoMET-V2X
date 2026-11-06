"""Additive peer16 registrations; existing peer15 work is not renamed."""
import inspect
from pathlib import Path
from peer15_v1_20261103 import ROOT,AUTO,AUDIT,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher
BASE=AUTO/'status_audit_20261103_peer16'
def record(name,values):
    tid='peer16_'+name+'_v1_20261103'
    if ROOT==REMOTE_ROOT:
        import fcntl
        BASE.mkdir(exist_ok=True)
        with (BASE/'.remote_delta.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            p=BASE/'remote_queue_delta.json';d=load(p) if p.exists() else {};d[tid]=dict(id=tid,parent_task_id=task('K1'),**values);atomic(p,d)
    else:update(tid,task('K1'),values)
def register(name,script,prediction,cost,**extras):
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/n).read_text();refs[n]=ev(AUTO/n)
    p=BASE/name/'contract.json';assert not p.exists(),p
    atomic(p,dict(created_at=now(),script=ev(Path(script).resolve()),instructions=refs,prediction_before_result=prediction,cost=cost,no_model_wakeup=True,official_main_unchanged=True,**extras))
    record(name,dict(status='registered_ready',contract=ev(p),estimated_cost=cost,prediction_before_result=prediction))
def start(name,script,args=(),remote=False,gpu=-1):
    ns=dict(launcher.__dict__,CYCLE=BASE)
    s=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer16-')
    exec(compile(s,__file__+'::launch','exec'),ns)
    return ns['start'](name,script,args,remote,gpu,25)
