"""R7 engineering recovery only: existing author spconv2.1 environment.

Zero-update failed attempt is retained. No checkpoint permutation or scientific
change. Remote helper/source prerequisites are copied explicitly before launch.
"""
import argparse,inspect,os
from pathlib import Path
from peer20_v1_20261105 import *
import run_peer20_K4_decontamination_v1_20261105 as old
OUT=BASE/'R7_environment_v2'

def prepare():
    assert load(BASE/'R6a_raw_labels_independent_v2/terminal.json')['confirmed']
    for seed in old.SEEDS:
        spec=load(old.OUT/('seed'+str(seed)+'_contract.json'))
        spec['R7']['label_deltas']=ev(old.OUT/'fit24_label_deltas.json')
        atomic(OUT/('seed'+str(seed)+'_contract.json'),spec)
    atomic(OUT/'fit24_label_deltas.json',load(old.OUT/'fit24_label_deltas.json'))
    for lane in ('local','remote'):
        register('R7_environment_v2_'+lane,__file__,__doc__,dict(GPU_updates=4000 if lane=='local' else 2000,
            CPU_threads=2,disk_GiB='<2/seed',native_forwards=46032 if lane=='local' else 23016),
            original_science=ev(BASE/('R7_'+lane)/'contract.json'),
            prior_local_failure=ev(BASE/'R7_local/terminal.json'),
            environment='existing .conda/traf-align-spconv21; composed legacy aliases/thermal shim; no checkpoint conversion',
            sources=[ev(old.TRAIN),ev(old.EVAL),ev(Path(old.__file__)),ev(ROOT/'preexperiment/research_autopilot/compat_traf_numpy124_thermal_v4/sitecustomize.py')])

def main(a):
    ns=dict(old.__dict__,OUT=OUT,__file__=str(Path(__file__).resolve()))
    code=inspect.getsource(old.worker);exec(compile(code,__file__+'::same_worker','exec'),ns)
    code=inspect.getsource(old.lane).replace("name='R7_'+which","name='R7_environment_v2_'+which")
    exec(compile(code,__file__+'::same_lane_correct_environment','exec'),ns)
    inherited=[p for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p and 'traf-align-spconv21' not in p and 'compat_traf_numpy124' not in p]
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'preexperiment/research_autopilot/compat_traf_numpy124_thermal_v4'),str(ROOT/'.conda/traf-align-spconv21'),*inherited])
    if a.mode in ('local','remote'):ns['lane'](a.mode,a.gpu)
    else:ns['worker'](a.mode,a.seed,a.output,a.gpu)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','local','remote','train','val','hold'));p.add_argument('--seed',type=int);p.add_argument('--gpu',type=int,default=1);p.add_argument('--output',type=Path);a=p.parse_args()
    prepare() if a.mode=='prepare' else main(a)
