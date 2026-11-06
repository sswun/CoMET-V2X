"""Peer13 additive cycle: immutable contracts and small cross-host receipts."""
import json
import subprocess
from peer12_cycle12_v1_20261102 import ROOT,AUTO,AUDIT,C09,CYCLE as C12,atomic,ev,load,now,update,task
from close_peer12_cycle03_v1_20261102 import local_resource

CYCLE=AUDIT/'cycle15_peer13_v1_20261102'

def capture():
    refs={}
    for p in [AUTO/'current_handoff.md',AUTO/'main_chat_directive.md',
              ROOT/'docs/review_20261031/peer13_evidence_shift_and_no_regret_reallocation.md']:
        p.read_text();refs[str(p.relative_to(ROOT))]=ev(p)
    CYCLE.mkdir(exist_ok=False)
    atomic(CYCLE/'entry.json',dict(created_at=now(),instructions=refs,script=ev(__file__),no_model_wakeup=True))
    atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
    snapshot(CYCLE/'entry_execution')

def snapshot(out):
    out.mkdir(exist_ok=False)
    atomic(out/'local.json',local_resource())
    script='''
import json,subprocess
from peer10_tournament_v1_20261101 import ROOT,AUDIT,load,ev,now
files={}
for label in ('cycle09_peer12_v1_20261102','cycle12_peer12_v1_20261102','cycle15_peer13_v1_20261102'):
 for p in (AUDIT/label).rglob('*.json'):
  if any(x in p.parts for x in ('entry_execution','final_execution','final_execution_snapshot','final_execution_snapshot_recovery_v2')):continue
  if p.name not in ('terminal.json','status.json','panels.json','profile_terminal.json') and 'queue_delta' not in p.name:continue
  files[str(p.relative_to(ROOT))]=dict(identity=ev(p),value=load(p))
def output(cmd):return subprocess.check_output(cmd,text=True).strip()
print(json.dumps(dict(created_at=now(),files=files,thermal=load(ROOT/'artifacts/thermal_guard/latest.json'),gpu=output(['nvidia-smi','--query-gpu=index,uuid,power.limit,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits']),units=output(['systemctl','--user','list-units','embodied-peer1*','--all','--no-pager']),processes=output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']))))
'''
    result=subprocess.run(['ssh','remote-gpu','cd /path/to/remote_project && PYTHONPATH=preexperiment/research_autopilot .conda/opencood-linux/bin/python -'],input=script,text=True,capture_output=True,check=True)
    atomic(out/'remote.json',json.loads(result.stdout))
    refs=[]
    for base in (C09,C12):
        refs.extend(ev(p) for p in base.rglob('terminal.json'))
    atomic(out/'local_terminal_references.json',refs)

if __name__=='__main__':capture()
