"""Peer19 immutable registration, sole-owner launch and shared-input helpers."""
import inspect,time
from pathlib import Path
from peer18_v1_20261104 import ROOT,AUTO,AUDIT,P17,P16,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as launcher
BASE=AUTO/'status_audit_20261104_peer19'
P18=AUTO/'status_audit_20261104_peer18'
def record(name,values):update('peer19_'+name+'_v1_20261104',task('K1'),values)
def register(name,script,prediction,cost,**extra):
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    p=BASE/name/'contract.json';assert not p.exists(),p
    atomic(p,dict(created_at=now(),script=ev(Path(script).resolve()),instructions=refs,
        directive=ev(ROOT/'docs/review_20261031/peer19_e2e_verdict_and_anchor_recipe.md'),
        prediction_before_result=prediction,cost=cost,official_main_unchanged=True,
        no_model_wakeup=True,original_release_closed=False,**extra))
    record(name,dict(status='registered_ready',contract=ev(p),estimated_cost=cost,resource_owner=False,owner=None))
def start(name,script,args=(),gpu=-1):
    ns=dict(launcher.__dict__,CYCLE=BASE)
    source=inspect.getsource(launcher.start).replace('embodied-peer14-c00-','embodied-peer19-').replace('-v1-20261103','-v1-20261104')
    exec(compile(source,__file__+'::launch','exec'),ns)
    return ns['start'](name,Path(script).name,args,False,gpu,25)
def wait(path,hours=26,accepted=('complete_light_checked','complete_independently_reviewed')):
    begin=time.monotonic()
    while not path.exists():
        assert time.monotonic()-begin<hours*3600,path
        time.sleep(30)
    value=load(path);assert value['status'] in accepted,(path,value)
    return value
