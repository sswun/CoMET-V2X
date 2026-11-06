"""Additive peer14 midnight cycle, shared references and bounded launches."""
from peer14_cycle21_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE as C21,C15,C18,REMOTE_ROOT,atomic,ev,load,now,update,task
import subprocess,shlex
CYCLE=AUDIT/'cycle00_peer14_v1_20261103'

def record(name,values):
    tid='peer14_cycle00_'+name+'_v1_20261103'
    if str(ROOT).startswith('/home/user/'):
        p=CYCLE/'remote_queue_delta.json';d=load(p) if p.exists() else {};d[tid]=dict(id=tid,parent_task_id=task('K5'),**values);atomic(p,d)
    else:update(tid,task('K5'),values)

def register(name,script,prediction,cost,**extra):
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):(AUTO/n).read_text();refs[n]=ev(AUTO/n)
    p=CYCLE/(name+'_contract.json');atomic(p,dict(created_at=now(),script=ev(script),instructions=refs,prediction_before_result=prediction,cost=cost,official_main_unchanged=True,no_model_wakeup=True,**extra));record(name,dict(status='registered_ready',contract=ev(p)))

def start(name,script,args=(),remote=False,gpu=-1,weight=25):
    root=REMOTE_ROOT if remote else ROOT;unit='embodied-peer14-c00-'+name+'-v1-20261103'
    cmd=['systemd-run','--user','--unit='+unit,'--slice=embodied-research.slice','--property=Wants=embodied-thermal.service','--property=After=embodied-thermal.service','--property=CPUWeight='+str(weight),'--working-directory='+str(root),'--setenv=PYTHONPATH=preexperiment/research_autopilot:preexperiment/research_baselines:preexperiment','--setenv=OMP_NUM_THREADS='+('2' if gpu>=0 else '1'),'--setenv=MKL_NUM_THREADS='+('2' if gpu>=0 else '1'),'--setenv=OPENBLAS_NUM_THREADS=1',str(root/'.conda/opencood-linux/bin/python'),'preexperiment/innovation_decision_audit/pressure_guard.py','--config','preexperiment/research_autopilot/'+('peer14_4090_shared_disk_resource_config_v1_20261102.json' if remote else 'traf_native_cv_heading_directional_gate_resource_config_v1_20261031.json'),'run','--output-dir',str(CYCLE.relative_to(ROOT)/(name+'_resource')),'--gpu-index',str(gpu),'--','.conda/opencood-linux/bin/python','-u','preexperiment/research_autopilot/'+script,*args]
    r=subprocess.run(['ssh','remote-gpu',shlex.join(cmd)] if remote else cmd,capture_output=True,text=True,check=True);atomic(CYCLE/'launch_receipts'/(name+'.json'),dict(created_at=now(),unit=unit+'.service',remote=remote,args=cmd,stdout=r.stdout,stderr=r.stderr));return unit

if __name__=='__main__':
    CYCLE.mkdir(exist_ok=False);atomic(CYCLE/'entry.json',dict(created_at=now(),script=ev(__file__),prior_handoff=ev(AUTO/'current_handoff.md'),directive=ev(AUTO/'main_chat_directive.md')));atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
