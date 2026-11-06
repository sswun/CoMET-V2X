"""Finite replication and missing-factor lanes, never evaluation-selected variants."""
import argparse
import copy
import inspect
import subprocess
import sys
import time
import traceback
from pathlib import Path

from peer12_cycle06_v1_20261102 import ROOT, AUTO, AUDIT, CYCLE as C06, atomic, ev, load, now, task, update
import run_peer12_where_native_execution_v6_20261102 as execution
import run_peer12_where_native_pair_v1_20261102 as original

CYCLE = AUDIT/'cycle09_peer12_v1_20261102'
VERSION = 'v1_20261102'
REMOTE = str(ROOT).startswith('/home/user/')


def record(tid, values, candidate='K4'):
    if REMOTE:
        path = CYCLE/'where_remote_queue_delta.json'
        rows = load(path) if path.exists() else {}
        rows[tid] = dict(id=tid, parent_task_id=task(candidate), **values)
        atomic(path, rows)
    else:
        update(tid, task(candidate), values)


def prepare():
    refs = {}
    for name in ('current_handoff.md', 'main_chat_directive.md'):
        (AUTO/name).read_text()
        refs[name] = ev(AUTO/name)
    assert all(load(C06/name/'terminal.json')['status']=='complete_light_checked'
               for name in ('where_public_v6', 'where_online_k1_v6'))
    base = load(C06/'where_execution_contract_v6.json')
    dependencies = [ev(ROOT/'preexperiment/research_autopilot'/name) for name in (
        'run_peer12_where_confirmation_v1_20261102.py',
        'run_peer12_where_native_execution_v6_20261102.py',
        'run_peer12_where_native_pair_v1_20261102.py',
        'run_peer12_where_native_pair_recovery_v4_20261102.py',
        'run_peer10_k4_native_finetune_v1_20261101.py',
        'run_peer10_native_evaluate_v1_20261101.py',
        'run_peer12_native_profile_v1_20261101.py',
        'run_peer10_k4_hold14_native_v1_20261101.py')]
    common = dict(base, created_at=now(), script=ev(__file__), instructions=refs,
                  inherited_execution=ev(C06/'where_execution_contract_v6.json'), dependencies=dependencies,
                  scope='Own Where q005 translation only + native single-frame Where fusion/head/NMS. No heading object projection or learned U; not a test of the complete late-fusion main method.',
                  prediction_before_result='Seed1 fair val AP70 -0.025445 and hold -3.489200 pp. Two additional paired seeds test whether this negative/native transfer is stable, not a new estimator or evaluation-selected recipe. Missing OFFICIAL factors are annotation-assisted diagnostics only.',
                  seed_selection='Fixed 20261102 and 20261103; all signs retained, final checkpoint only, no evaluation-based schedule or model selection.',
                  no_model_wakeup=True,
                  storage='Reuse C06 input identities and author weight. One final weight and sole compressed predictions per new identity; no copied GT/features. Rolling resume has active consumer until lane closes.')
    common['estimated_cost'] = dict(local_per_lane_updates=4000, local_per_lane_microbatches=16000,
        local_per_lane_paired_scientific_forwards=92032,
        local_missing_official_forwards={'GPU0':92032, 'GPU1':69024},
        remote_updates=2000, remote_scientific_forwards=69024,
        profile_and_first_case_replays='Mandatory 64-condition profile and first-condition same-environment replay per cell, recorded separately.',
        wall_h_per_local_lane='3-5 measured from seed1 ~50min paired train+AP; OFFICIAL factors additional 1-2h',
        remote_wall_h='1.5-3 after current three-model hold slow-reference lane',
        new_disk_GiB='<4 paired final/active recovery and unique predictions; source deltas reference these')
    atomic(CYCLE/'where_confirmation_contract.json', common)
    for seed in (20261101, 20261102, 20261103):
        contract = copy.deepcopy(common)
        contract['K4']['seed'] = seed
        atomic(CYCLE/('where_seed'+str(seed)+'_contract.json'), contract)
    for lane in ('gpu0', 'gpu1', 'remote'):
        record('peer12_where_confirmation_'+lane+'_'+VERSION,
               dict(status='registered_finite_ready', contract=ev(CYCLE/'where_confirmation_contract.json'),
                    prediction_before_result=common['prediction_before_result'], estimated_cost=common['estimated_cost']))
    record('peer12_where_confirmation_sources_'+VERSION,
           dict(status='registered_finite_successor', contract=ev(CYCLE/'where_confirmation_contract.json'),
                estimated_cost=dict(GPU_forwards=0, CPU_threads=1, CPU_wall_h='4-12 shared quota', new_disk_GiB='<0.5 flags/metrics only')), 'K5')


def setup():
    execution.setup()
    c = load(CYCLE/'where_confirmation_contract.json')
    assert ev(__file__) == c['script']
    for reference in c['dependencies']:
        assert ev(ROOT/reference['path']) == reference
    for reference in c['inputs'].values():
        assert ev(ROOT/reference['path']) == reference
    return c


def cell(weight, gpu, out, hold=False, arms='PUBLIC,ONLINE_K1'):
    # Reuse v6's actual tested explicit namespace and 64-condition profile gate.
    source = inspect.getsource(execution.cell)
    old = "CYCLE/'where_execution_contract_v6.json'"
    assert source.count(old) == 1
    source = source.replace(old, "CYCLE/'where_confirmation_contract.json'")
    namespace = dict(execution.__dict__, CYCLE=CYCLE)
    exec(compile(source, __file__+'::same_v6_evaluator', 'exec'), namespace)
    namespace['cell'](weight, gpu, out, hold, arms)


def train(seed, arm, gpu, out):
    sys.argv = [__file__, '--workspace', str(ROOT), '--contract',
                str(CYCLE/('where_seed'+str(seed)+'_contract.json')),
                '--output', str(out), '--arm', arm, '--gpu', str(gpu)]
    namespace = dict(__name__='where_fixed_seed_replication', __file__=__file__)
    exec(compile(original.training_source(), __file__+'::unchanged_train', 'exec'), namespace)
    namespace['main']()


def attempt(out, function, *args):
    try:
        function(*args)
        return dict(status='complete_light_checked', path=str(out.relative_to(ROOT)), terminal=ev(out/'terminal.json'))
    except Exception as error:
        out.mkdir(parents=True, exist_ok=True)
        failure = dict(status='failed_retained', created_at=now(), error=repr(error), traceback=traceback.format_exc())
        # A saved successful producer never becomes a failed producer because
        # a downstream consumer failed. This out is the individual cell.
        if not (out/'terminal.json').exists():
            atomic(out/'terminal.json', failure)
        return dict(path=str(out.relative_to(ROOT)), **failure)


def pair(seed, arm, gpu, out):
    out.mkdir(exist_ok=False)
    train(seed, arm, gpu, out/'training')
    results = []
    for role in ('val21', 'hold14'):
        dest = out/role
        results.append(attempt(dest, cell, out/'training/final.pth', gpu, dest, role=='hold14'))
    status = 'complete_light_checked' if all(r['status']=='complete_light_checked' for r in results) else 'failed_retained'
    atomic(out/'terminal.json', dict(status=status, created_at=now(), seed=seed, arm=arm, results=results, no_model_wakeup=True))


def lane(gpu):
    setup()
    label = 'gpu'+str(gpu)
    out = CYCLE/('where_confirmation_'+label)
    out.mkdir(exist_ok=False)
    tid = 'peer12_where_confirmation_'+label+'_'+VERSION
    record(tid, dict(status='running_finite_paired_seed_replications_and_missing_factors', unit='embodied-peer12-where-confirm-'+label+'-v1-20261102.service'))
    results = []
    arm = 'PUBLIC' if gpu==0 else 'ONLINE_K1'
    for seed in (20261102, 20261103):
        dest = out/('seed'+str(seed))
        results.append(attempt(dest, pair, seed, arm, gpu, dest))
        atomic(out/'status.json', dict(status='finite_lane', created_at=now(), results=results))
    # Complete only genuinely missing OFFICIAL-input factors; never re-evaluate
    # already saved PUBLIC/ONLINE predictions or the e60 PUBLIC baseline.
    weights = [('seed20261101', C06/('where_public_v6' if gpu==0 else 'where_online_k1_v6')/'training/final.pth')]
    weights += [('seed'+str(seed), out/('seed'+str(seed))/'training/final.pth') for seed in (20261102, 20261103)]
    if gpu==0:
        weights.append(('author', original.WEIGHT))
    for name, weight in weights:
        for role in ('val21', 'hold14'):
            dest = out/('official_'+name+'_'+role)
            results.append(attempt(dest, cell, weight, gpu, dest, role=='hold14', 'OFFICIAL_REFERENCE'))
            atomic(out/'status.json', dict(status='finite_lane', created_at=now(), results=results))
    status = 'complete_light_checked' if all(r['status']=='complete_light_checked' for r in results) else 'failed_retained'
    atomic(out/'terminal.json', dict(status=status, created_at=now(), results=results, no_model_wakeup=True))
    record(tid, dict(status=status, terminal=ev(out/'terminal.json')))


def remote():
    out = CYCLE/'where_confirmation_remote'
    out.mkdir(exist_ok=False)
    predecessor = 'embodied-peer12-factor-gap-remote-v1-20261102.service'
    execution.wait_unit(predecessor, out)
    setup()
    tid = 'peer12_where_confirmation_remote_'+VERSION
    record(tid, dict(status='running_annotation_assisted_same_budget_official_training_factor', unit='embodied-peer12-where-confirm-remote-v1-20261102.service'))
    results = []
    training = out/'training'
    result = attempt(training, train, 20261101, 'OFFICIAL_REFERENCE', 0, training)
    results.append(result)
    if result['status']=='complete_light_checked':
        for role in ('val21', 'hold14'):
            dest = out/role
            results.append(attempt(dest, cell, training/'final.pth', 0, dest, role=='hold14', 'PUBLIC,ONLINE_K1,OFFICIAL_REFERENCE'))
            atomic(out/'status.json', dict(status='finite_remote', created_at=now(), results=results))
    status = 'complete_light_checked' if all(r['status']=='complete_light_checked' for r in results) else 'failed_retained'
    atomic(out/'terminal.json', dict(status=status, created_at=now(), results=results, annotation_assisted_not_deployable=True, no_model_wakeup=True))
    record(tid, dict(status=status, terminal=ev(out/'terminal.json')))


def sources():
    from run_peer12_native_source_additions_v1_20261101 import compute_namespace
    namespace, contract = compute_namespace()
    namespace['CYCLE'] = CYCLE
    out = CYCLE/'where_confirmation_sources'
    out.mkdir(exist_ok=False)
    results = []
    for gpu in (0, 1):
        parent = CYCLE/('where_confirmation_gpu'+str(gpu))
        producers = [parent/('seed'+str(seed))/role for seed in (20261102, 20261103) for role in ('val21','hold14')]
        producers += [parent/('official_seed'+str(seed)+'_'+role) for seed in (20261101,20261102,20261103) for role in ('val21','hold14')]
        if gpu==0:
            producers += [parent/('official_author_'+role) for role in ('val21','hold14')]
        for folder in producers:
            producer = folder/'evaluation'
            while not (folder/'terminal.json').exists() and not (parent/'terminal.json').exists():
                atomic(out/'status.json', dict(status='waiting_known_finite_producer', producer=str(producer), created_at=now(), results=results))
                time.sleep(30)
            try:
                assert load(producer/'terminal.json')['status']=='native_AP_complete_light_checked'
                name = str(producer.relative_to(AUDIT))
                namespace['compute'](name, contract)
                results.append(dict(status='complete_light_checked', producer=name, terminal=ev(CYCLE/'native_source_aux'/name/'terminal.json')))
            except Exception as error:
                results.append(dict(status='failed_retained', producer=str(producer), error=repr(error), traceback=traceback.format_exc()))
    status = 'complete_light_checked' if all(r['status']=='complete_light_checked' for r in results) else 'failed_retained'
    atomic(out/'terminal.json', dict(status=status, created_at=now(), results=results, no_model_wakeup=True))
    record('peer12_where_confirmation_sources_'+VERSION, dict(status=status, terminal=ev(out/'terminal.json')), 'K5')


def preflight():
    setup()
    compile(original.training_source(), __file__+'::train_preflight', 'exec')
    for hold in (False, True):
        for profile in (False, True):
            compile(original.source_for('where', hold, profile), __file__+'::eval_preflight', 'exec')
    atomic(CYCLE/('where_confirmation_preflight_'+('remote' if REMOTE else 'local')+'.json'),
           dict(status='complete_light_checked', created_at=now(), GPU_forwards=0, updates=0,
                deployment_not_full_heading_U=True, actual_GPU_qualification='first four backward updates of each fixed seed'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','preflight','gpu0','gpu1','remote','sources'])
    mode = parser.parse_args().mode
    if mode=='prepare': prepare()
    elif mode=='preflight': preflight()
    elif mode=='gpu0': lane(0)
    elif mode=='gpu1': lane(1)
    elif mode=='remote': remote()
    else: sources()
