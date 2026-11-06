"""Exact frozen native evaluator, explicitly adapted to official train hold14.

Only role/input/dataset membership changes. Model, geometry, readouts, RNG,
native ROI, no-GT input, scoring and replay stay the original evaluator's.
The exact source and every single-occurrence substitution are in the contract.
"""
import argparse,sys
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,atomic,ev,load,sha
from peer10_cycle09_v1_20261101 import CYCLE

CHANGES=[
 ("('val21_material','val21_gate_material','val_conditions')", "('mechanism_holdout14_material','mechanism_holdout14_gate_material','train_conditions')"),
 ("refs['val21_material']", "refs['mechanism_holdout14_material']"),
 ("refs['val21_gate_material']", "refs['mechanism_holdout14_gate_material']"),
 ("refs['val_conditions']", "refs['train_conditions']"),
 ("assert len(cases)==13264", "assert len(cases)==9744 and all(r['role']=='mechanism_holdout14' for r in cases)"),
 ("ds=builder(copy.deepcopy(h),set='val');adapter.install", "ds=builder(copy.deepcopy(h),set='train');ds.train=False;adapter.install"),
 ("state={'status':'initializing'", "state={'role':'mechanism_holdout14','status':'initializing'"),
]

def adapted(root):
    spec=load(CYCLE/'gpu_contract_recovery_v2.json');path=root/spec['native_evaluator']['path']
    assert sha(path)==spec['native_evaluator']['sha256']
    source=path.read_text()
    for old,new in CHANGES:
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    return compile(source,str(Path(__file__).resolve())+'::frozen_native_hold14','exec')

def main():
    # Preserve the frozen evaluator's command-line surface, aside from role.
    code=adapted(ROOT);ns={'__name__':'peer10_hold14_adapted','__file__':__file__}
    exec(code,ns);ns['main']()

if __name__=='__main__':main()
