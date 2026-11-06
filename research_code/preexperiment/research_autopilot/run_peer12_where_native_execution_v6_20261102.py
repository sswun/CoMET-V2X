"""Finite Where native lanes with explicit profile/evaluator namespaces.

Supersedes only unstarted waiters. Author code, paired sampling, own q005
inputs, CPU loader proof and original readout remain unchanged. No rerun of
previous native predictions or currently executing GPU work.
"""
import argparse
import inspect
import subprocess
import sys
import time
import traceback

import run_peer12_where_native_pair_v1_20261102 as old
from peer12_cycle06_v1_20261102 import ROOT, AUTO, AUDIT, CYCLE, atomic, ev, load, now, task, update

VERSION = 'v6_20261102'


def prepare():
    instructions = {}
    for name in ('current_handoff.md', 'main_chat_directive.md'):
        (AUTO/name).read_text()
        instructions[name] = ev(AUTO/name)
    contract = dict(load(CYCLE/'where_contract_recovery_v5.json'))
    contract.update(created_at=now(), script=ev(__file__), instructions=instructions,
                    predecessor_contract=ev(CYCLE/'where_contract_recovery_v5.json'),
                    CPU_loader_proof=ev(CYCLE/'where_qualification_v5/terminal.json'),
                    recovery='Static orchestration check found copied run_cell namespace references lag_inputs even for non-lag runs. Replace unstarted waiters with explicit native profile/evaluation functions, not a scientific change. No GPU training/evaluation has started in the superseded lanes.',
                    estimated_cost=dict(per_arm_updates=2000, per_arm_microbatches=8000,
                                        per_arm_formal_forwards=46016, per_arm_profile_forwards=272,
                                        plugin_formal_forwards=23008, plugin_profile_forwards=136,
                                        replay_forwards_per_paired_arm=16,
                                        plugin_replay_forwards=8,
                                        GPU_wall_h_per_paired_arm='3-7 under thermal quota',
                                        new_disk_GiB='<1.5 sole predictions/final/active rolling state'),
                    historical_budget_correction='Original 69048 per arm assumed three inputs; each paired arm has two inputs: (13264+9744)*2=46016 scientific forwards, plus 16 first-condition replays and 272 mandatory profile forwards. Plugin has one input across both roles:23008 scientific+8 replays+136 profile. Optional LRU replay profile, only if read>=30%, is additional.')
    atomic(CYCLE/'where_execution_contract_v6.json', contract)
    for arm in ('public', 'online_k1'):
        update('peer12_where_native_'+arm+'_'+VERSION, task('K4'),
               dict(status='registered_after_actual_CPU_loader_qualification',
                    contract=ev(CYCLE/'where_execution_contract_v6.json'),
                    prediction_before_result=contract['prediction_before_result'],
                    estimated_cost=contract['estimated_cost']))
    for role in ('val21', 'hold14'):
        update('peer12_where_native_plugin_'+role+'_'+VERSION, task('K1'),
               dict(status='registered_finite_successor', contract=ev(CYCLE/'where_execution_contract_v6.json'),
                    prediction_before_result='No-finetuning Where author weight plus own frozen q005 input; only missing ONLINE condition. Not same-budget K4 gain.',
                    estimated_cost=contract['estimated_cost']))
    update('peer12_where_native_source_aux_'+VERSION, task('K5'),
           dict(status='registered_finite_successor', contract=ev(CYCLE/'where_execution_contract_v6.json'),
                prediction_before_result='Native/common five-source official and two fixed auxiliary evaluations for paired arms and independent e60 plugin; sole prediction references only.',
                estimated_cost=dict(GPU_forwards=0, CPU_threads=1, CPU_wall_h='4-12 after finite producers', new_disk_GiB='<1 flags/metrics only')))


def setup():
    contract = load(CYCLE/'where_execution_contract_v6.json')
    assert ev(__file__) == contract['script']
    assert ev(old.WEIGHT) == contract['K4']['initial_weight']
    proof = load(CYCLE/'where_qualification_v5/terminal.json')
    assert proof['status'] == 'complete_light_checked' and proof['actual_batches'] == 752
    from run_peer12_where_native_pair_recovery_v4_20261102 import build
    old.build = build
    source = inspect.getsource(old.batch_audit)
    before = "batch['time_delays'][0].numpy()"
    assert source.count(before) == 1
    source = source.replace(before, "np.asarray(batch['time_delays'][0])")
    namespace = dict(old.__dict__)
    exec(compile(source, __file__+'::native1_age_layout', 'exec'), namespace)
    old.batch_audit = namespace['batch_audit']
    return contract


def wait_unit(unit, out):
    while subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ActiveState'], text=True).strip() in ('ActiveState=active', 'ActiveState=activating'):
        atomic(out/'status.json', dict(status='waiting_known_predecessor', unit=unit, created_at=now()))
        time.sleep(30)


def cell(weight, gpu, out, hold=False, arms='PUBLIC,ONLINE_K1'):
    import run_peer12_native_profile_v1_20261101 as profile
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    narm = len(arms.split(','))

    def run(is_profile, destination, cache):
        sys.argv = [__file__, '--workspace', str(ROOT), '--contract', str(CYCLE/'where_execution_contract_v6.json'),
                    '--output', str(destination), '--weight', str(weight), '--gpu', str(gpu), '--arms', arms]
        namespace = dict(__name__='where_native_explicit_evaluator', __file__=__file__,
                         install_reader=profile.install_reader,
                         attach_projection_timer=profile.attach_projection_timer,
                         cache_enabled=cache)
        exec(compile(old.source_for('where', hold, is_profile), __file__+'::native1_evaluator', 'exec'), namespace)
        namespace['main']()

    run(True, out/'profile_uncached', False)
    rows = load(out/'profile_uncached/profile_rows.json')
    assert len(rows) == 64*narm
    totals = {key: sum(row[key] for row in rows) for key in ('data_total_s', 'read_parse_s', 'projection_voxel_s', 'forward_s', 'AP_accounting_s')}
    fraction = totals['read_parse_s']/sum(totals[key] for key in ('data_total_s', 'forward_s', 'AP_accounting_s'))
    eligible = False
    replays = []
    if fraction >= .3:
        import numpy as np
        run(True, out/'profile_cached', True)
        for wait in (0, 100, 200, 400):
            with np.load(out/'profile_uncached'/f'wait{wait:03d}.npz') as left, np.load(out/'profile_cached'/f'wait{wait:03d}.npz') as right:
                equal = left.files == right.files and all(left[k].dtype == right[k].dtype and left[k].shape == right[k].shape and left[k].tobytes() == right[k].tobytes() for k in left.files)
                replays.append(dict(wait_ms=wait, bitwise_equal=equal))
        eligible = all(row['bitwise_equal'] for row in replays)
    atomic(out/'profile_terminal.json', dict(status='complete_light_checked', conditions=64, input_arms=narm,
            forwards=68*narm*(2 if fraction >= .3 else 1), timings=totals, read_parse_fraction=fraction,
            raw_cache_eligible=eligible, replays=replays, not_scientific_AP=True))
    run(False, out/'evaluation', eligible)
    terminal = load(out/'evaluation/terminal.json')
    assert terminal['status'] == 'native_AP_complete_light_checked' and terminal['cases'] == (9744 if hold else 13264)
    atomic(out/'terminal.json', dict(status='complete_light_checked', created_at=now(), weight=ev(weight),
            evaluation=ev(out/'evaluation/terminal.json'), profile=ev(out/'profile_terminal.json'),
            wall_s=time.monotonic()-started, no_model_wakeup=True))


def lane(arm, gpu):
    name = arm.lower()
    out = CYCLE/('where_'+name+'_v6')
    out.mkdir(exist_ok=False)
    tid = 'peer12_where_native_'+name+'_'+VERSION
    predecessor = 'embodied-peer12-factor-gap-gpu0-v1-20261102.service' if gpu == 0 else 'embodied-peer12-mixed-factor-gpu1-v1-20261102.service'
    wait_unit(predecessor, out)
    started = time.monotonic()
    try:
        setup()
        update(tid, task('K4'), dict(status='running_native_training_then_profiled_AP', unit='embodied-peer12-where-'+str(gpu)+'-v6-20261102.service'))
        sys.argv = [__file__, '--workspace', str(ROOT), '--contract', str(CYCLE/'where_execution_contract_v6.json'), '--output', str(out/'training'), '--arm', arm, '--gpu', str(gpu)]
        namespace = dict(__name__='where_native_paired_train_v6', __file__=__file__)
        exec(compile(old.training_source(), __file__+'::native1_train', 'exec'), namespace)
        namespace['main']()
        for role in ('val21', 'hold14'):
            cell(out/'training/final.pth', gpu, out/role, role == 'hold14')
        atomic(out/'terminal.json', dict(status='complete_light_checked', created_at=now(), actual_weight=ev(out/'training/final.pth'), wall_s=time.monotonic()-started, no_model_wakeup=True))
        update(tid, task('K4'), dict(status='complete_light_checked', terminal=ev(out/'terminal.json')))
    except Exception as error:
        atomic(out/'terminal.json', dict(status='failed_retained', created_at=now(), error=repr(error), traceback=traceback.format_exc(), wall_s=time.monotonic()-started))
        update(tid, task('K4'), dict(status='failed_retained', terminal=ev(out/'terminal.json')))
        raise


def plugin(gpu):
    role = 'val21' if gpu == 0 else 'hold14'
    out = CYCLE/('where_e60_'+role+'_v6')
    out.mkdir(exist_ok=False)
    tid = 'peer12_where_native_plugin_'+role+'_'+VERSION
    wait_unit('embodied-peer12-where-'+str(gpu)+'-v6-20261102.service', out)
    try:
        setup()
        update(tid, task('K1'), dict(status='running_native_plugin_profile_then_AP'))
        cell(old.WEIGHT, gpu, out/'native', gpu == 1, arms='ONLINE_K1')
        atomic(out/'terminal.json', dict(status='complete_light_checked', created_at=now(), separate_no_finetuning_scenario=True, evaluation=ev(out/'native/evaluation/terminal.json'), no_model_wakeup=True))
        update(tid, task('K1'), dict(status='complete_light_checked', terminal=ev(out/'terminal.json')))
    except Exception as error:
        atomic(out/'terminal.json', dict(status='failed_retained', created_at=now(), error=repr(error), traceback=traceback.format_exc()))
        update(tid, task('K1'), dict(status='failed_retained', terminal=ev(out/'terminal.json')))
        raise


def sources():
    from run_peer12_native_source_additions_v1_20261101 import compute_namespace
    namespace, contract = compute_namespace()
    namespace['CYCLE'] = CYCLE
    out = CYCLE/'where_sources_v6'
    out.mkdir(exist_ok=False)
    results = []
    producers = [(CYCLE/('where_'+arm+'_v6'), role+'/evaluation') for arm in ('public', 'online_k1') for role in ('val21', 'hold14')]
    producers += [(CYCLE/('where_e60_'+role+'_v6'), 'native/evaluation') for role in ('val21', 'hold14')]
    for parent, subpath in producers:
        producer = parent/subpath
        terminal = producer/'terminal.json'
        while not terminal.exists() and not (parent/'terminal.json').exists():
            atomic(out/'status.json', dict(status='waiting_known_Where_native_prediction', producer=str(producer), created_at=now()))
            time.sleep(30)
        try:
            assert terminal.exists() and load(terminal)['status'] == 'native_AP_complete_light_checked', ('producer failed', str(parent))
            name = str(producer.relative_to(AUDIT))
            namespace['compute'](name, contract)
            results.append(dict(producer=name, terminal=ev(CYCLE/'native_source_aux'/name/'terminal.json')))
        except Exception as error:
            results.append(dict(producer=str(producer), error=repr(error), traceback=traceback.format_exc()))
        atomic(out/'status.json', dict(status='finite_native_Where_sources', created_at=now(), results=results))
    status = 'complete_light_checked' if all('error' not in row for row in results) else 'failed_retained'
    atomic(out/'terminal.json', dict(status=status, created_at=now(), results=results, no_model_wakeup=True))
    update('peer12_where_native_source_aux_'+VERSION, task('K5'), dict(status=status, terminal=ev(out/'terminal.json')))


def preflight():
    # Compiles every role/profile without constructing a CUDA model. Actual
    # strict load/backward remains the first step on the assigned GPU lane.
    setup()
    checks = []
    for hold in (False, True):
        for profile in (False, True):
            compile(old.source_for('where', hold, profile), __file__+'::preflight', 'exec')
            checks.append(dict(hold=hold, profile=profile, compiled=True))
    compile(old.training_source(), __file__+'::train_preflight', 'exec')
    atomic(CYCLE/'where_execution_preflight_v6.json', dict(status='complete_light_checked', created_at=now(), checks=checks, CPU_loader_reference=ev(CYCLE/'where_qualification_v5/terminal.json'), GPU_forwards=0, updates=0, actual_GPU_qualification_pending=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'preflight', 'public', 'online', 'plugin0', 'plugin1', 'sources'])
    mode = parser.parse_args().mode
    if mode == 'prepare': prepare()
    elif mode == 'preflight': preflight()
    elif mode == 'public': lane('PUBLIC', 0)
    elif mode == 'online': lane('ONLINE_K1', 1)
    elif mode == 'plugin0': plugin(0)
    elif mode == 'plugin1': plugin(1)
    else: sources()
