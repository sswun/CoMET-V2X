"""Peer21 additive immutable contracts and guarded sole-owner task launches."""
import inspect
from pathlib import Path
from peer20_v1_20261105 import ROOT,AUTO,AUDIT,P19,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher

P20=AUTO/'status_audit_20261105_peer20'
BASE=AUTO/'status_audit_20261105_peer21'
DIRECTIVE=ROOT/'docs/review_20261031/peer21_offset_grid_saturation_and_clean_native.md'

def record(name,values):
    tid='peer21_'+name+'_v1_20261105';parent=task('K4' if name.startswith('R7') else 'K5')
    if ROOT==REMOTE_ROOT:
        p=BASE/'remote_queue_delta.json';d=load(p) if p.exists() else {}
        d[tid]=dict(id=tid,parent_task_id=parent,**values);atomic(p,d)
    else:update(tid,parent,values)

def register(name,script,prediction,cost,**extra):
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    p=BASE/name/'contract.json';assert not p.exists(),p
    atomic(p,dict(created_at=now(),script=ev(Path(script).resolve()),instructions=refs,
        directive=ev(DIRECTIVE),prediction_before_result=prediction,cost=cost,
        selection='cal8 official AP only',official_main_unchanged=True,no_hold_off=True,
        inference_no_labels_or_future=True,no_model_wakeup=True,
        original_release_closed=False,intermediate_deadlines_cancelled=True,**extra))
    record(name,dict(status='registered_ready',contract=ev(p),estimated_cost=cost,
        prediction_before_result=prediction,owner=None,resource_owner=False))

def start(name,script,args=(),remote=False,gpu=-1):
    assert not (BASE/'launch_receipts'/(name+'.json')).exists()
    ns=dict(launcher.__dict__,CYCLE=BASE)
    code=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer21-').replace('-v1-20261103','-v1-20261105')
    exec(compile(code,__file__+'::guarded_launch','exec'),ns)
    return ns['start'](name,Path(script).name,args,remote,gpu,25)
