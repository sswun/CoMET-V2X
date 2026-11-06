"""Resolve existing canonical/mirrored native roots, no rerun or copied cache.

Author native already exists in the no-delete4090 mirror. strict60 canonical
root and ViT2025 reviewed summary were resolved in peer18; reuse these facts.
All v1 missing-path failures retained. Scientific adapter/statistics unchanged.
"""
import argparse,inspect
from peer20_v1_20261105 import *
import run_peer20_external_population_v1_20261105 as old
FAMILIES=('native_author','native_strict60','native_vit2025')
MIRROR=ROOT/'artifacts/remote4090_mirror/remote_project'

def prepare():
    for family in FAMILIES:
        for r in (False,True):
            register('R6bd_external_v2_'+family+('_independent' if r else ''),__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<250 compact outcomes only'),
                original=ev(old.__file__),adapter=ev(old.native.__file__),failure=ev(BASE/('R6bd_external_'+family)/'terminal.json'),science_unchanged=True)

def native_input(family,review):
    s=inspect.getsource(old.native_input)
    s=s.replace("ROOT/RUNS['traf']", "MIRROR/RUNS['traf']")
    s=s.replace("STRICT/f'strict{w:03d}/run'", "(AUTO/'status_audit_20261023_acceleration/traf_strict_training_fairness_v1/native_terminal_recovery_v3/evaluation_v1')/f'strict{w:03d}/run'")
    s=s.replace("status=ev(p/'status.json')", "status=ev(p/'status.json') if (p/'status.json').exists() else ev(AUTO/'status_audit_20261025_autonomy/v2xvit_seed2025_terminal_synthesis_v1/analysis_v1/summary.json')")
    ns=dict(old.__dict__,MIRROR=MIRROR);exec(compile(s,__file__+'::source_roots','exec'),ns)
    yield from ns['native_input'](family,review)

def run(family,review):
    source=inspect.getsource(old.run).replace("'R6bd_external_'+family", "'R6bd_external_v2_'+family")
    ns=dict(old.__dict__,native_input=native_input)
    body=inspect.getsource(old.inputs);exec(compile(body,__file__+'::same_inputs','exec'),ns)
    exec(compile(source,__file__+'::versioned_outputs','exec'),ns);ns['run'](family,review)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review'));p.add_argument('--family',choices=FAMILIES);a=p.parse_args()
    prepare() if a.mode=='prepare' else run(a.family,a.mode=='review')
