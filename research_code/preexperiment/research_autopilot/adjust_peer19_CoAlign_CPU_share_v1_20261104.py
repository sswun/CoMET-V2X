"""Raise only R3's share inside our thermal-limited slice; unchanged science.

GPU1 is CPU-starved (~34% one CPU, GPU under10%) while many independent CPU
consumers share400%. CPUWeight25->100 changes intra-project sharing only;
no slice quota, temperature/power limit, other project or detector change.
"""
import subprocess
from peer19_v1_20261104 import *
if __name__=='__main__':
    name='coalign_H1_CPU_share';dest=BASE/name;unit='embodied-peer19-coalign-H1-gpu1-v1-20261104.service'
    register(name,__file__,__doc__,dict(GPU_forwards_added=0,updates=0,CPU_threads=2,disk_MiB='<1 resource receipt'),
        producer=ev(BASE/'coalign_H1/contract.json'),resource_change_only=True)
    before=subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUWeight','-p','ControlGroup','-p','ActiveState'],text=True)
    assert 'CPUWeight=25' in before and 'ActiveState=active' in before and 'embodied-research.slice' in before
    subprocess.run(['systemctl','--user','set-property','--runtime',unit,'CPUWeight=100'],check=True)
    after=subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUWeight','-p','ControlGroup','-p','ActiveState'],text=True)
    assert 'CPUWeight=100' in after
    atomic(dest/'terminal.json',dict(status='complete_light_checked',created_at=now(),before=before,after=after,
        other_projects_untouched=True,thermal_quota_unchanged=True,weights_inputs_readout_unchanged=True,
        interpretation='Actual throughput costs must include resource share and thermal waiting; not model-native latency.'))
    record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    record('coalign_H1',dict(runtime_CPUWeight=100,resource_adjustment=ev(dest/'terminal.json')))
