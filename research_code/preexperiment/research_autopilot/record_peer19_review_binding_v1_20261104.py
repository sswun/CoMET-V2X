"""Append actual reviewer execution identity; never rewrite frozen contract."""
import subprocess
from peer19_v1_20261104 import *

if __name__=='__main__':
    name='H1_assets_review_execution_binding';dest=BASE/name
    register(name,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,disk_MiB='<1'),
        original=ev(BASE/'H1_assets_independent/contract.json'),
        actual_review_script=ev(ROOT/'preexperiment/research_autopilot/review_peer19_H1_assets_v1_20261104.py'),
        limitation='Original contract script points to registration producer, NOT actual reviewer. Independent receipt corrects execution binding, no retrospective claim.')
    launch=BASE/'launch_receipts/H1-assets-independent.json';value=load(launch)
    script='preexperiment/research_autopilot/review_peer19_H1_assets_v1_20261104.py'
    assert script in value['args']
    unit=value['unit'];command=subprocess.check_output(['systemctl','--user','show',unit,'-p','ExecStart','-p','ActiveState','-p','MainPID'],text=True)
    assert script in command
    atomic(dest/'terminal.json',dict(status='complete_light_checked',created_at=now(),original_contract=ev(BASE/'H1_assets_independent/contract.json'),
        actual_script=ev(ROOT/script),launch=ev(launch),unit=unit,observed_command=command,
        original_contract_binding_incorrect=True,original_contract_unchanged=True,scientific_result_not_preaccepted=True))
    record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    record('H1_assets_independent',dict(execution_binding=ev(dest/'terminal.json'),actual_execution_script=ev(ROOT/script)))
