"""Peer20 immutable contracts, sole-owner launches and additive queue updates."""
import inspect
import time
from pathlib import Path
from peer19_v1_20261104 import ROOT,AUTO,AUDIT,P17,P16,P18,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher

BASE=AUTO/'status_audit_20261105_peer20'
P19=AUTO/'status_audit_20261104_peer19'
DIRECTIVE=ROOT/'docs/review_20261031/peer20_official_offset_placeholder_and_population_split.md'

def record(name,values):
    update('peer20_'+name+'_v1_20261105',task('K4') if name.startswith('R7') else task('K1'),values)

def register(name,script,prediction,cost,**extra):
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    p=BASE/name/'contract.json'
    assert not p.exists(),p
    atomic(p,dict(created_at=now(),script=ev(Path(script).resolve()),instructions=refs,
        directive=ev(DIRECTIVE),prediction_before_result=prediction,cost=cost,
        official_main_unchanged=True,selection='cal8 official AP only; no population-based inference or hold-off search',
        no_model_wakeup=True,intermediate_deadlines_cancelled=True,
        external_nodes=['2026-11-10 abstract','2026-11-16 paper'],original_release_closed=False,**extra))
    record(name,dict(status='registered_ready',contract=ev(p),estimated_cost=cost,
        prediction_before_result=prediction,owner=None,resource_owner=False))

def start(name,script,args=(),remote=False,gpu=-1):
    receipt=BASE/'launch_receipts'/(name+'.json')
    assert not receipt.exists(),receipt
    ns=dict(launcher.__dict__,CYCLE=BASE)
    source=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer20-').replace('-v1-20261103','-v1-20261105')
    exec(compile(source,__file__+'::launch','exec'),ns)
    return ns['start'](name,Path(script).name,args,remote,gpu,25)

def wait(path,accepted=('complete_light_checked','complete_independently_reviewed')):
    while not path.exists():time.sleep(30)
    value=load(path)
    assert value['status'] in accepted,(path,value)
    return value
