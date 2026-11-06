"""Additive bootstrap repair; scientific transforms unchanged from frozen v1."""
import argparse
from pathlib import Path
from peer14_cycle03_v1_20261103 import CYCLE, ROOT, atomic, ev, now, record, register as cycle_register
import run_peer14_coalign_physical_inputs_v1_20261103 as original


def register(name, script, prediction, cost, **extra):
    extra['bootstrap_repair'] = dict(original_script=ev(original.__file__),
        scope='correct fit-freeze provenance and homogeneous-matrix build assertion only; no scientific transform changes',
        prior_failed_units=['embodied-peer14-c03-physical-inputs-public-v1-20261103',
                            'embodied-peer14-c03-physical-inputs-online-v1-20261103'],
        scientific_forwards_in_failed_bootstrap=0)
    return cycle_register(name, script, prediction, cost, **extra)


source = Path(original.__file__).read_text()
replacements = {
    "coalign_physical_inputs_v1'": "coalign_physical_inputs_v2'",
    "coalign_physical_inputs_contract.json": "coalign_physical_inputs_recovery_v2_contract.json",
    "'coalign_physical_inputs'": "'coalign_physical_inputs_recovery_v2'",
    "'physical_inputs_'": "'physical_inputs_recovery_v2_'",
    "k2_k3_formal_resume_v3/fit_freeze.json": "k2_k3_cached_recovery_v2/fit_freeze.json",
    "all(np.allclose(a[1][3],[0,0,0,1]) for a in v.items())":
        "all(np.allclose(transforms[1][3],[0,0,0,1]) for transforms in v.values())",
}
for old, new in replacements.items():
    assert old in source, old
    source = source.replace(old, new)
ns = dict(__name__='frozen_physical_inputs_recovery_v2', __file__=str(Path(__file__).resolve()))
exec(compile(source, str(Path(__file__).resolve())+'::v1_bootstrap_repaired', 'exec'), ns)
ns['register'] = register

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'public', 'online'])
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    if args.mode == 'prepare':
        ns['prepare']()
        for arm in ('public', 'online'):
            resource = CYCLE / ('physical-inputs-'+arm+'_resource')
            atomic(CYCLE / ('bootstrap_failure_'+arm+'.json'), dict(created_at=now(),
                status='failed_retained', scientific_GPU_forwards=0,
                reason='prepare failed on nonexistent fit_freeze path; run failed before loading contract/model',
                command_log=ev(resource/'command.log'), recovery_script=ev(__file__)))
            record('physical_inputs_'+arm, dict(status='failed_retained',
                terminal=ev(CYCLE/('bootstrap_failure_'+arm+'.json'))))
    else:
        ns['run'](args.mode, args.gpu)
