"""Peer17 final-recipe registrations; immutable predecessors stay unchanged."""
import inspect
from pathlib import Path
from peer16_v1_20261103 import ROOT,AUTO,AUDIT,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher
BASE=AUTO/'status_audit_20261104_peer17'

def record(name,values):
    parent='peer17_final_recipe_v1_20261104' if name.startswith('final_recipe_') else task('K1')
    update('peer17_'+name+'_v1_20261104',parent,values)

def register(name,script,prediction,cost,**extras):
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/n).read_text();refs[n]=ev(AUTO/n)
    p=BASE/name/'contract.json';assert not p.exists(),p
    atomic(p,dict(created_at=now(),script=ev(Path(script).resolve()),instructions=refs,
        prediction_before_result=prediction,cost=cost,no_model_wakeup=True,
        official_main_unchanged=True,**extras))
    record(name,dict(status='registered_ready',contract=ev(p),estimated_cost=cost,
        prediction_before_result=prediction))

def start(name,script,args=(),gpu=-1):
    ns=dict(launcher.__dict__,CYCLE=BASE)
    src=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer17-').replace('-v1-20261103','-v1-20261104')
    exec(compile(src,__file__+'::launch','exec'),ns)
    return ns['start'](name,script,args,False,gpu,25)
