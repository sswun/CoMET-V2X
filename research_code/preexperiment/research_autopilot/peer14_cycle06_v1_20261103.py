"""06h additive registration and controlled launch helpers; calendar only."""
import inspect
from peer14_cycle03_v1_20261103 import ROOT,AUTO,AUDIT,CYCLE as C03,C00,C21,C18,C15,REMOTE_ROOT,atomic,ev,load,now,update,task
import peer14_cycle00_v1_20261103 as prior
CYCLE=AUDIT/'cycle06_peer14_v1_20261103'

def record(name,values):
    tid='peer14_cycle06_'+name+'_v1_20261103'
    if str(ROOT).startswith('/home/user/'):
        p=CYCLE/'remote_queue_delta.json';d=load(p) if p.exists() else {};d[tid]=dict(id=tid,parent_task_id=task('K5'),**values);atomic(p,d)
    else:update(tid,task('K5'),values)

def register(name,script,prediction,cost,**extra):
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/n).read_text();refs[n]=ev(AUTO/n)
    p=CYCLE/(name+'_contract.json')
    atomic(p,dict(created_at=now(),script=ev(script),instructions=refs,prediction_before_result=prediction,cost=cost,official_main_unchanged=True,no_model_wakeup=True,**extra))
    record(name,dict(status='registered_ready',contract=ev(p)))

def start(name,script,args=(),remote=False,gpu=-1,weight=25):
    ns=dict(prior.__dict__,CYCLE=CYCLE)
    source=inspect.getsource(prior.start).replace('embodied-peer14-c00-','embodied-peer14-c06-')
    exec(compile(source,__file__+'::launch','exec'),ns)
    return ns['start'](name,script,args,remote,gpu,weight)

if __name__=='__main__':
    CYCLE.mkdir(exist_ok=False)
    atomic(CYCLE/'entry.json',dict(created_at=now(),script=ev(__file__),handoff=ev(AUTO/'current_handoff.md'),directive=ev(AUTO/'main_chat_directive.md')))
    atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
