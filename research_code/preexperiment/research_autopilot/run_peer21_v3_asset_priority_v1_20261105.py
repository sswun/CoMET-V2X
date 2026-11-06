"""Advance two ready, lightweight v3 assets using the original DAG units.

Only CSV/JSON reductions and serialization review; no dataset materialization,
AP, bootstrap, fitting or forwards. Same units/receipts/canonical outputs let
the frozen DAG observe execution without duplicate launches or science edits.
"""
import argparse
import fcntl
import time
from peer21_v1_20261105 import *
import peer20_v1_20261105 as p20
import peer20_finite_resource_DAG_v1_20261105 as dag

NAME='v3_asset_priority'
OUT=BASE/NAME
KEYS=('R1-CI-attach','strict-cal4-attach')

def prepare():
    register(NAME,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        new_AP=0,new_bootstrap=0,disk_MiB='<45 table versions/CI CSV; no repeated candidate cache'),
        frozen_DAG=ev(dag.__file__),keys=list(KEYS),
        lightweight_only=True,max_additional_dataset_workers=0,
        source_consumers=[ev(ROOT/'preexperiment/research_autopilot'/n) for n in
            ('consume_peer19_R1_independent_CI_v1_20261104.py','consume_peer19_strict_cal4_tables_v1_20261104.py')])

def run():
    lock=(OUT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs={j['key']:j for j in dag.load(dag.OUT/'state.json')['jobs']}
    completed=[]
    for key in KEYS:
        j=jobs[key]
        assert all(dag.scientific_ready(p) for p in j['deps']),key
        if not dag.scientific_ready(j['output']):
            if not dag.active(j['unit']):
                assert not (P20/'launch_receipts'/('dag-'+key+'.json')).exists(),key
                unit=p20.start('dag-'+key,j['script'],j['args'])
                assert unit+'.service'==j['unit']
                p20.record(j['name'],dict(status='running_original_DAG_lightweight_asset_priority',
                    owner=j['unit'],resource_owner=True))
            while dag.active(j['unit']):
                atomic(OUT/'progress.json',dict(created_at=now(),current=key,completed=completed,
                    no_dataset_worker=True))
                time.sleep(10)
        assert dag.scientific_ready(j['output']),(key,j['output'])
        completed.append(dict(key=key,terminal=ev(j['output'])))
    atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',
        completed=completed,no_scientific_rerun=True,no_additional_dataset_worker=True))
    record(NAME,dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),owner=None,resource_owner=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));globals()[p.parse_args().mode]()
