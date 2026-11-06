"""Finite peer21 exception: one extra R7 statistics worker, same DAG units.

Only R7 AP/review/read may occupy the third heavy CPU slot. No retries, no
model wakeups, no changes to other DAG steps or to running scientific workers.
"""
import argparse, fcntl, time
from pathlib import Path
import peer20_finite_resource_DAG_v1_20261105 as dag
from peer20_v1_20261105 import BASE, ROOT, register, record, atomic, ev, now

NAME='R7_peer21_priority'; OUT=BASE/NAME

def prepare():
    register(NAME,__file__,__doc__,dict(CPU_threads=1,updates=0,GPU_forwards=0,
        disk_MiB='<2 coordinator; existing R7 outputs unchanged'),
        peer21=ev(ROOT/'docs/review_20261031/peer21_offset_grid_saturation_and_clean_native.md'),
        maximum_extra_R7_workers=1,launch_memory_GiB=28,no_retry=True)

def run():
    lock=(OUT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[j for j in dag.plan() if j['key'].startswith(('R7-AP-','R7-review-')) or j['key']=='R7-read']
    while True:
        live=[j for j in jobs if dag.active(j['unit'])]
        complete=[j['key'] for j in jobs if dag.scientific_ready(j['output']) and not dag.active(j['unit'])]
        if len(complete)==len(jobs):break
        failed=[j['key'] for j in jobs if (BASE/'launch_receipts'/('dag-'+j['key']+'.json')).exists()
            and not dag.active(j['unit']) and not dag.scientific_ready(j['output'])]
        if failed:
            atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',failed=failed,no_retry=True))
            record(NAME,dict(status='failed_retained',owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')));return
        available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
        if not live and available>=28*1024**3:
            for j in jobs:
                if j['key'] in complete or not all(dag.scientific_ready(p) for p in j['deps']):continue
                # The exact same systemd unit and immutable launch receipt prevent
                # a second scientific owner if the original DAG reaches this step.
                dag.launch(j);live=[j];break
        atomic(OUT/'progress.json',dict(created_at=now(),completed=complete,running=[j['key'] for j in live],available_bytes=available))
        time.sleep(30)
    atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',completed=complete,scientific_decision=str(BASE/'R7_preregistered_read/decision.json')))
    record(NAME,dict(status='complete_light_checked',owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));a=p.parse_args()
    prepare() if a.mode=='prepare' else run()
