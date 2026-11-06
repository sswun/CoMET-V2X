"""Additive 06h evidence capture. Never rewrites a prior experimental result."""
import argparse, json, subprocess
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT, AUTO, AUDIT, atomic, ev, load, now, task
import peer10_cycle09_v1_20261101 as registry

CYCLE = AUDIT / 'cycle06_peer12_v1_20261102'
C03 = AUDIT / 'cycle03_peer12_v1_20261102'

def update(tid, parent, values):
    key = 'cycle06_' + tid
    registry.JOBS[key] = (tid, parent)
    registry.update(key, values)

def capture():
    refs = {n: ev(AUTO/n) for n in ('current_handoff.md', 'main_chat_directive.md')}
    for n in refs:
        (AUTO/n).read_text()
    CYCLE.mkdir(exist_ok=False)
    atomic(CYCLE/'storage_entry.json', load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
    atomic(CYCLE/'entry.json', dict(created_at=now(), instructions=refs, script=ev(__file__),
        no_model_wakeup=True, official_main_unchanged=True, no_predictions_copied=True))
    from close_peer12_cycle03_v1_20261102 import local_resource
    atomic(CYCLE/'local_execution_snapshot.json', local_resource())
    script = '''
import json, subprocess
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,AUDIT,ev,load,now
c=AUDIT/'cycle03_peer12_v1_20261102'
files={}
for p in sorted(c.rglob('*.json')):
 if p.name not in ('terminal.json','status.json','panels.json','profile_terminal.json','annotation_reference_identity.json') and 'delta' not in p.name:continue
 files[str(p.relative_to(ROOT))]={'identity':ev(p),'value':load(p)}
def output(cmd):return subprocess.check_output(cmd,text=True).strip()
processes=[]
for line in output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']).splitlines():
 pid,uuid,mem=map(str.strip,line.split(','));p=Path('/proc')/pid
 if not p.exists():continue
 cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 processes.append(dict(pid=int(pid),uuid=uuid,memory_MiB=mem,cmdline=cmd,cgroup=(p/'cgroup').read_text(),project_model='python' in cmd and ('preexperiment/research_autopilot/' in cmd or str(ROOT) in cmd)))
units=['embodied-peer12-cycle03-remote-v2-20261102','embodied-peer12-factor-gap-remote-v1-20261102','embodied-peer12-factor-source-v1-20261102','embodied-peer12-cumulative-source-v1-20261102']
print(json.dumps(dict(created_at=now(),files=files,processes=processes,thermal=load(ROOT/'artifacts/thermal_guard/latest.json'),gpu=output(['nvidia-smi','--query-gpu=index,uuid,power.limit,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits']),units={u:output(['systemctl','--user','show',u+'.service','-p','ActiveState','-p','ControlGroup','-p','Wants','-p','CPUWeight']) for u in units},no_predictions_copied=True)))
'''
    r = subprocess.run(['ssh','remote-gpu','cd /path/to/remote_project && PYTHONPATH=preexperiment/research_autopilot .conda/opencood-linux/bin/python -'], input=script, text=True, capture_output=True, check=True)
    remote = json.loads(r.stdout)
    atomic(CYCLE/'remote_execution_snapshot.json', remote)
    local = {}
    for p in sorted(C03.rglob('*.json')):
        if p.name not in ('terminal.json','status.json','panels.json','profile_terminal.json','gpu_terminal.json','independent_review.json','manifest_v1.json'):
            continue
        local[str(p.relative_to(ROOT))] = dict(identity=ev(p), value=load(p))
    atomic(CYCLE/'local_output_snapshot.json', dict(created_at=now(), files=local))
    print(json.dumps(dict(local_files=len(local),remote_files=len(remote['files']),remote_gpu=remote['gpu'])))

if __name__ == '__main__':
    capture()
