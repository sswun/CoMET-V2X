"""Peer22 immutable contracts, additive queues and guarded launches."""
import inspect
from peer21_v1_20261105 import ROOT,AUTO,P19,P20,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher
P21=AUTO/'status_audit_20261105_peer21'
BASE=AUTO/'status_audit_20261106_peer22'
DIRECTIVE=ROOT/'docs/review_20261031/peer22_tier2_attribution_and_budget_dose.md'
def record(name,values):
    tid='peer22_'+name+'_v1_20261106';parent=task('K4' if name.startswith('R7') else 'K5')
    if ROOT==REMOTE_ROOT:
        p=BASE/'remote_queue_delta.json';v=load(p) if p.exists() else {}
        v[tid]=dict(id=tid,parent_task_id=parent,**values);atomic(p,v)
    else:update(tid,parent,values)
def register(name,script,prediction,cost,**extra):
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    path=BASE/name/'contract.json';assert not path.exists(),path
    atomic(path,dict(created_at=now(),script=ev(script),instructions=refs,directive=ev(DIRECTIVE),
        prediction_before_result=prediction,cost=cost,selection='cal8 official only',
        official_main_unchanged=True,no_hold_off=True,no_model_wakeup=True,
        original_release_closed=False,**extra))
    record(name,dict(status='registered_ready',contract=ev(path),estimated_cost=cost,
        prediction_before_result=prediction,owner=None,resource_owner=False))
def start(name,script,args=(),remote=False,gpu=-1):
    assert not (BASE/'launch_receipts'/(name+'.json')).exists()
    ns=dict(launcher.__dict__,CYCLE=BASE)
    source=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer22-').replace('-v1-20261103','-v1-20261106')
    exec(compile(source,__file__+'::launch','exec'),ns)
    unit=ns['start'](name,script,args,remote,gpu,25)+'.service'
    record(name.replace('-','_'),dict(status='running',owner=unit+('@4090' if remote else ''),resource_owner=True))
    return unit
