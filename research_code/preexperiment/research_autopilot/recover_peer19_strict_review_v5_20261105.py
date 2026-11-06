"""Strict reviewer recovery: legacy source has two cal waits, not four.

The full four-wait cal check is a separate already accepted task. Do not
fabricate missing waits here or overwrite the v4 failed terminal.
"""
import argparse, inspect
from peer20_v1_20261105 import *
import recover_peer19_committed_review_v4_20261105 as recovery

NAME='R1_review_recovery_v5_strict60'
DEST=P19/'recipe_independent_resource_recovery_v5/strict60'

def prepare():
    old=recovery.destination('strict60')
    register(NAME,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<60'),
        failed_recovery=ev(old/'terminal.json'),kernel=ev(recovery.old.__file__),
        prefix={n:ev(old/n) for n in ('AP.json','bootstrap.json')},
        full_cal4_review=ev(P19/'strict_cal4_H1_independent/terminal.json'),new_output=str(DEST))
    DEST.mkdir(parents=True,exist_ok=False)
    for p in old.iterdir():
        if p.name!='terminal.json' and (p.suffix=='.npz' or p.name in ('AP.json','bootstrap.json')):
            (DEST/p.name).symlink_to(p.resolve())

def run():
    source=inspect.getsource(recovery.run)
    source=source.replace("name='R1_review_recovery_'+domain;dest=destination(domain)","name=NAME;dest=DEST")
    source=source.replace("assert len(existing)==4 and {r['wait_ms'] for r in existing}=={0,100,200,400}",
        "assert len(existing)==len(np.unique(m['wait_ms'])) and {r['wait_ms'] for r in existing}==set(map(int,np.unique(m['wait_ms'])))")
    ns=dict(recovery.__dict__,NAME=NAME,DEST=DEST)
    exec(compile(source,__file__+'::actual_legacy_wait_coverage','exec'),ns);ns['run']('strict60')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));globals()[p.parse_args().mode]()
