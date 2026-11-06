"""Peer14 immutable registration and controlled cycle21 launch helpers."""
import argparse,json,shlex,subprocess
from pathlib import Path
from peer13_cycle18_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE as C18,C15,C09,atomic,ev,load,now,update,task
CYCLE=AUDIT/'cycle21_peer14_v1_20261102'
REMOTE_ROOT=Path('/path/to/remote_project')
RESOURCE='preexperiment/research_autopilot/peer14_4090_shared_disk_resource_config_v1_20261102.json'

def record(name,values):
    tid='peer14_'+name+'_v1_20261102'
    if str(ROOT).startswith('/home/user/'):
        p=CYCLE/'remote_queue_delta.json';d=load(p) if p.exists() else {};d[tid]=dict(id=tid,parent_task_id=task('K5'),**values);atomic(p,d)
    else:update(tid,task('K5'),values)

def register(name,path,prediction,cost):
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):(AUTO/n).read_text();refs[n]=ev(AUTO/n)
    value=dict(created_at=now(),script=ev(path),instructions=refs,prediction_before_result=prediction,cost=cost,
               official_main_unchanged=True,no_model_wakeup=True)
    atomic(CYCLE/(name+'_contract.json'),value)
    record(name,dict(status='registered_ready',contract=ev(CYCLE/(name+'_contract.json')),prediction_before_result=prediction,estimated_cost=cost))

def start(name,script,args=(),remote=False,gpu=-1,weight=25):
    root=REMOTE_ROOT if remote else ROOT
    unit='embodied-peer14-'+name+'-v1-20261102'
    command=['systemd-run','--user','--unit='+unit,'--slice=embodied-research.slice',
        '--property=Wants=embodied-thermal.service','--property=After=embodied-thermal.service',
        '--property=CPUWeight='+str(weight),'--working-directory='+str(root),
        '--setenv=PYTHONPATH=preexperiment/research_autopilot:preexperiment/research_baselines:preexperiment',
        '--setenv=OMP_NUM_THREADS='+('2' if gpu>=0 else '1'),'--setenv=MKL_NUM_THREADS='+('2' if gpu>=0 else '1'),
        '--setenv=OPENBLAS_NUM_THREADS=1',str(root/'.conda/opencood-linux/bin/python'),
        'preexperiment/innovation_decision_audit/pressure_guard.py','--config',RESOURCE if remote else
        'preexperiment/research_autopilot/traf_native_cv_heading_directional_gate_resource_config_v1_20261031.json',
        'run','--output-dir',str(CYCLE.relative_to(ROOT)/(name+'_resource')),'--gpu-index',str(gpu),'--',
        '.conda/opencood-linux/bin/python','-u','preexperiment/research_autopilot/'+script,*args]
    result=subprocess.run(['ssh','remote-gpu',shlex.join(command)] if remote else command,capture_output=True,text=True,check=True)
    atomic(CYCLE/'launch_receipts'/(name+'.json'),dict(created_at=now(),unit=unit+'.service',remote=remote,args=command,stdout=result.stdout,stderr=result.stderr))
    return unit

def prepare():
    CYCLE.mkdir(exist_ok=False)
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    atomic(CYCLE/'entry.json',dict(created_at=now(),instructions=refs,script=ev(__file__),no_model_wakeup=True))
    atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))

if __name__=='__main__':prepare()
