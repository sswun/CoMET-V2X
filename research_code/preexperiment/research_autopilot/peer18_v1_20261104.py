"""Bounded peer18 tasks and controlled independent experiment launch."""
import inspect
from pathlib import Path
from peer17_v1_20261104 import ROOT,AUTO,AUDIT,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher
BASE=AUTO/'status_audit_20261104_peer18'
P17=AUTO/'status_audit_20261104_peer17'
P16=AUTO/'status_audit_20261103_peer16'
def record(name,values):
    tid='peer18_'+name+'_v1_20261104'
    if ROOT==REMOTE_ROOT:
        import fcntl
        with (BASE/'.remote_delta.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            p=BASE/'remote_queue_delta.json';d=load(p) if p.exists() else {}
            d[tid]=dict(id=tid,parent_task_id=task('K1'),**values);atomic(p,d)
    else:update(tid,task('K1'),values)
def register(name,script,prediction,cost,**extra):
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/n).read_text();refs[n]=ev(AUTO/n)
    p=BASE/name/'contract.json';assert not p.exists(),p
    atomic(p,dict(created_at=now(),script=ev(Path(script).resolve()),instructions=refs,
        prediction_before_result=prediction,cost=cost,official_main_unchanged=True,no_model_wakeup=True,
        deadline='2026-11-06T06:00:00+08:00',stop='2026-11-06T12:00:00+08:00',
        original_release_closed=False,**extra))
    record(name,dict(status='registered_ready',contract=ev(p),owner=None,resource_owner=False))
def start(name,script,args=(),remote=False,gpu=-1):
    ns=dict(launcher.__dict__,CYCLE=BASE)
    src=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer18-').replace('-v1-20261103','-v1-20261104')
    exec(compile(src,__file__+'::launch','exec'),ns)
    return ns['start'](name,script,args,remote,gpu,25)
