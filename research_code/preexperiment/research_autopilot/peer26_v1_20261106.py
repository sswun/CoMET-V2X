"""Additive, research-only peer25/26 identities and guarded launches."""
import inspect
from pathlib import Path
from peer24_v1_20261106 import ROOT,AUTO,P19,P20,P21,P22,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher
P24=AUTO/'status_audit_20261106_peer24'
BASE=AUTO/'status_audit_20261106_peer26'
DEADLINE='2026-11-14T12:00:00+08:00'

def record(name,values):update('peer26_'+name+'_v1_20261106',task('K5'),values)

def register(name,script,prediction,cost,**extra):
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    path=BASE/name/'contract.json';assert not path.exists(),path
    atomic(path,dict(created_at=now(),script=ev(Path(script).resolve()),instructions=refs,
        directives=[ev(ROOT/'docs/review_20261031'/n) for n in (
            'peer25_R8_screen_and_motion_mean_candidate.md','peer26_innovation_push_to_1114_stress_and_acquisition.md')],
        prediction_before_result=prediction,cost=cost,paper_numbers_frozen=True,root_table_unchanged=True,
        no_hold_off=True,no_model_wakeup=True,original_release_closed=False,
        scientific_launch_before=DEADLINE,independent_completion_before='2026-11-14T20:00:00+08:00',**extra))
    record(name,dict(status='registered_ready',owner=None,resource_owner=False,contract=ev(path),
        estimated_cost=cost,prediction_before_result=prediction))

def start(name,script,args=(),remote=False,gpu=-1):
    assert not (BASE/'launch_receipts'/(name+'.json')).exists()
    ns=dict(launcher.__dict__,CYCLE=BASE)
    source=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer26-').replace('-v1-20261103','-v1-20261106')
    exec(compile(source,__file__+'::guarded_launch','exec'),ns)
    unit=ns['start'](name,Path(script).name,args,remote,gpu,25)+'.service'
    record(name.replace('-','_'),dict(status='running',owner=unit,resource_owner=True))
    return unit
