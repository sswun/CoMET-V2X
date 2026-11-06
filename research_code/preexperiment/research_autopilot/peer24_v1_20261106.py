"""Peer24 research-only registrations; published paper numbers stay frozen."""
import inspect
from pathlib import Path
from peer23_v1_20261106 import ROOT,AUTO,P19,P20,P21,P22,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher
P23=AUTO/'status_audit_20261106_peer23'
BASE=AUTO/'status_audit_20261106_peer24'
DIRECTIVE=ROOT/'docs/review_20261031/peer24_archive_and_research_track.md'
TABLE=ROOT/'docs/research_v2/R8a'

def record(name,values):
    update('peer24_'+name+'_v1_20261106',task('K5'),values)

def register(name,script,prediction,cost,**extra):
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    p=BASE/name/'contract.json';assert not p.exists(),p
    atomic(p,dict(created_at=now(),script=ev(Path(script).resolve()),instructions=refs,directive=ev(DIRECTIVE),
        prediction_before_result=prediction,cost=cost,paper_numbers_frozen=True,research_output='docs/research_v2/',
        official_main_unchanged=True,no_hold_off=True,no_model_wakeup=True,original_release_closed=False,**extra))
    record(name,dict(status='registered_ready',contract=ev(p),estimated_cost=cost,prediction_before_result=prediction,
        owner=None,resource_owner=False))

def start(name,script,args=()):
    assert not (BASE/'launch_receipts'/(name+'.json')).exists()
    ns=dict(launcher.__dict__,CYCLE=BASE)
    source=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer24-').replace('-v1-20261103','-v1-20261106')
    exec(compile(source,__file__+'::guarded_CPU','exec'),ns)
    unit=ns['start'](name,Path(script).name,args,False,-1,25)+'.service'
    record(name.replace('-','_'),dict(status='running',owner=unit,resource_owner=True))
    return unit
