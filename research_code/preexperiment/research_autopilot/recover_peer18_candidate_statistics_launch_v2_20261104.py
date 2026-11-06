"""Recover zero-compute launcher path failure, unchanged frozen statistics code.

The first units passed an absolute script path to a basename-only launcher.
Keep their contracts and failed resource receipts. New independent task roots;
no statistics, model or hypothesis change.
"""
import argparse,inspect
import review_peer18_candidate_statistics_v1_20261104 as old
from peer18_v1_20261104 import *
def name_for(seed):return 'candidate_statistics_independent_launch_v2_'+str(seed)
def prepare(seed):
    failed=BASE/old.name_for(seed)
    atomic(failed/'terminal.json',dict(created_at=now(),status='failed_retained',
        reason='Launcher expected basename but received absolute path; Python could not open script; zero scientific compute.',GPU_forwards=0,updates=0))
    record(old.name_for(seed),dict(status='failed_retained',terminal=ev(failed/'terminal.json'),resource_owner=False,owner=None))
    register(name_for(seed),__file__,old.__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,panels=600,
        replicates=2000,wall_h='bounded24h',disk_MiB='<40 independent contrasts; shared inputs'),
        frozen_engine=ev(old.__file__),kernel=ev(old.independent.__file__),seed=seed,predecessor=ev(failed/'terminal.json'))
def run(seed):
    ns=dict(old.__dict__,__file__=__file__,name_for=name_for)
    assert load(BASE/name_for(seed)/'contract.json')['frozen_engine']==ev(old.__file__)
    exec(compile(inspect.getsource(old.run),__file__+'::frozen_statistics','exec'),ns);ns['run'](seed)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--seed',type=int,choices=(0,2023,2024,2025),required=True)
    a=p.parse_args();globals()[a.mode](a.seed)
