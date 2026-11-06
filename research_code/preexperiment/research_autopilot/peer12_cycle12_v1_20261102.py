"""12h evidence capture: references locally, small remote receipts only."""
import json
import subprocess
from peer12_cycle06_v1_20261102 import ROOT, AUTO, AUDIT, C03, CYCLE as C06, atomic, ev, load, now, update, task
from close_peer12_cycle03_v1_20261102 import local_resource

C09 = AUDIT / 'cycle09_peer12_v1_20261102'
CYCLE = AUDIT / 'cycle12_peer12_v1_20261102'


def capture():
    instructions = {}
    for name in ('current_handoff.md', 'main_chat_directive.md'):
        (AUTO/name).read_text()
        instructions[name] = ev(AUTO/name)
    CYCLE.mkdir(exist_ok=False)
    atomic(CYCLE/'entry.json', dict(created_at=now(), instructions=instructions,
        script=ev(__file__), no_model_wakeup=True, official_main_unchanged=True))
    atomic(CYCLE/'storage_entry.json', load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
    atomic(CYCLE/'local_execution_snapshot.json', local_resource())
    references = []
    for base in (C03, C09):
        references += [ev(p) for p in base.rglob('terminal.json')]
    atomic(CYCLE/'local_terminal_references.json', references)
    script = '''
import json,subprocess
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,AUDIT,load,ev,now
files={}
for base in (AUDIT/'cycle03_peer12_v1_20261102',AUDIT/'cycle09_peer12_v1_20261102'):
 for p in base.rglob('*.json'):
  if any(part in ('final_execution_snapshot','snapshot_analysis','vit_native','vit_native_v2') for part in p.parts):continue
  if p.name not in ('terminal.json','status.json','panels.json','profile_terminal.json','gpu_terminal.json') and 'queue_delta' not in p.name:continue
  files[str(p.relative_to(ROOT))]=dict(identity=ev(p),value=load(p))
def output(cmd):return subprocess.check_output(cmd,text=True).strip()
models=[]
for line in output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']).splitlines():
 pid,uuid,mem=map(str.strip,line.split(','));proc=Path('/proc')/pid
 if not proc.exists():continue
 cmd=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 if 'preexperiment/research_autopilot/' not in cmd:continue
 env=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode().split('\\0') if '=' in x)
 models.append(dict(pid=int(pid),uuid=uuid,memory_MiB=mem,cmdline=cmd,cgroup=(proc/'cgroup').read_text(),OMP=env.get('OMP_NUM_THREADS'),MKL=env.get('MKL_NUM_THREADS'),pacing=load(ROOT/'artifacts/thermal_guard/pacing'/f'{pid}.json')))
units=['embodied-peer12-vit-native-remote-v3-20261102','embodied-peer12-vit-native-remote-source-v3-20261102','embodied-peer12-where-confirm-remote-v1-20261102','embodied-peer12-where-confirm-remote-sources-v1-20261102']
print(json.dumps(dict(created_at=now(),files=files,models=models,thermal=load(ROOT/'artifacts/thermal_guard/latest.json'),gpu=output(['nvidia-smi','--query-gpu=index,uuid,power.limit,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits']),units={u:output(['systemctl','--user','show',u+'.service','-p','ActiveState','-p','ControlGroup','-p','CPUWeight','-p','Wants']) for u in units})))
'''
    result = subprocess.run(['ssh', 'remote-gpu',
        'cd /path/to/remote_project && PYTHONPATH=preexperiment/research_autopilot .conda/opencood-linux/bin/python -'],
        input=script, text=True, capture_output=True, check=True)
    remote = json.loads(result.stdout)
    atomic(CYCLE/'remote_execution_snapshot.json', remote)
    print(dict(local_terminal_refs=len(references), remote_receipts=len(remote['files']),
               remote_models=remote['models'], remote_gpu=remote['gpu']))


if __name__ == '__main__':
    capture()
