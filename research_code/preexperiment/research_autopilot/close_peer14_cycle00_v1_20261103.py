"""Midnight exit: additive evidence assets, one mirror, queue/owner audit. No dispatch."""
import argparse,csv,json,shlex,subprocess,time
from pathlib import Path
from collections import defaultdict
from peer14_cycle00_v1_20261103 import ROOT,AUTO,AUDIT,CYCLE,C21,C18,C15,REMOTE_ROOT,atomic,ev,load,now,update,task
from peer10_tournament_v1_20261101 import QUEUES
import close_peer14_cycle21_v1_20261102 as prior
TERMINAL=prior.TERMINAL

def snapshot():
    d=prior.snapshot()
    for p in CYCLE.rglob('*.json'):
        if any(x in str(p) for x in ('exit_snapshot','publication','conflicts')):continue
        if p.name not in ('terminal.json','status.json','quality_frozen.json') and 'queue_delta' not in p.name:continue
        d['files'][str(p.relative_to(ROOT))]=dict(identity=ev(p),value=load(p))
    return d

def mirror():
    out=CYCLE/'mirror_incremental_v1';out.mkdir(exist_ok=False)
    backup=ROOT/'artifacts/remote4090_mirror/conflicts_cycle00_20261103'
    command=['rsync','-a','--stats','--backup','--backup-dir='+str(backup),
        '--exclude=.conda/','--exclude=.venv/','--exclude=artifacts/data/',
        '--exclude=artifacts/remote4090_mirror/','--exclude=__pycache__/',
        'remote-gpu:'+str(REMOTE_ROOT)+'/',str(ROOT/'artifacts/remote4090_mirror/remote_project')+'/']
    start=time.monotonic();r=subprocess.run(command,capture_output=True,text=True)
    atomic(out/'terminal.json',dict(status='complete_light_checked' if r.returncode==0 else 'failed_retained',
        created_at=now(),command=command,stdout=r.stdout,stderr=r.stderr,exit_code=r.returncode,
        wall_s=time.monotonic()-start,no_delete=True,old_changed_mirror_files_retained=str(backup.relative_to(ROOT))))
    assert r.returncode==0,r.stderr

def capture():
    out=CYCLE/'exit_snapshot_v1';out.mkdir(exist_ok=False);atomic(out/'local.json',snapshot())
    code='from close_peer14_cycle00_v1_20261103 import snapshot;import json;print(json.dumps(snapshot()))'
    raw=prior.output(['ssh','remote-gpu','cd '+str(REMOTE_ROOT)+' && PYTHONPATH=preexperiment/research_autopilot .conda/opencood-linux/bin/python -c '+shlex.quote(code)])
    atomic(out/'remote.json',json.loads(raw))

def reconcile():
    merged=[];skipped=[]
    for host in ('local','remote'):
        snap=load(CYCLE/'exit_snapshot_v1'/(host+'.json'))
        for p,ref in snap['files'].items():
            if 'queue_delta' not in Path(p).name and 'remote_delta' not in Path(p).name:continue
            for tid,v in ref['value'].items():
                if not isinstance(v,dict) or 'parent_task_id' not in v:continue
                rows={r['id']:r for r in load(QUEUES[0])['tasks']};current=rows.get(tid,{})
                if current.get('status') in TERMINAL and v.get('status') not in TERMINAL:skipped.append(tid);continue
                update(tid,v['parent_task_id'],{k:x for k,x in v.items() if k not in ('id','parent_task_id')});merged.append(tid)
    # Do not close a contract's scientific children or replace terminal evidence by old waiting states.
    atomic(CYCLE/'queue_reconciliation_v1.json',dict(created_at=now(),merged=merged,skipped_stale=skipped,no_dispatch=True))

def table(name,rows):
    assert rows
    with (CYCLE/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def means(rows,field,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r[field])
    return [dict(zip(keys,k),AP_mean=sum(v)/len(v),n_waits=len(v)) for k,v in groups.items()]

def assets():
    cached=load(C21/'registration_read_v1/V2/panels.json');native=load(CYCLE/'native_physical_dose_v1/V2_native_registration/panels.json')
    assert len(cached)==384 and len(native)==192
    table('V2_registration_cached_all384.csv',cached);table('V2_registration_native_all192.csv',native)
    cm=means(cached,'ap',('role','method','protocol','source_group','threshold'))
    nm=means(native,'AP',('role','method','protocol','source_group','threshold'))
    table('V2_registration_cached_means.csv',cm);table('V2_registration_native_means.csv',nm)
    def value(rows,role,method,protocol):
        rr=[r for r in rows if (r['role'],r['method'],r['protocol'],r['source_group'],r['threshold'])==(role,method,protocol,'all',70)]
        assert len(rr)==1 and rr[0]['n_waits']==2;return rr[0]['AP_mean']
    contrasts=[]
    for role in ('val21','mechanism_holdout14'):
        for protocol in ('official_quality_subset','registration_SE2_z_quality_subset'):
            k1=value(cm,role,'K1_exact',protocol);k2=value(cm,role,'K2_infra',protocol);stale=value(cm,role,'stale_same_U',protocol)
            public=value(nm,role,'public_trained__PUBLIC',protocol);online=value(nm,role,'online_trained__ONLINE_K1',protocol)
            contrasts.append(dict(role=role,protocol=protocol,K1=k1,K2=k2,stale=stale,K1_minus_stale_pp=k1-stale,K2_minus_K1_pp=k2-k1,
                native_PUBLIC_PUBLIC=public,native_ONLINE_ONLINE=online,native_paired_delta_pp=online-public,population='original saved quality-conditioned subset, not full main AP'))
    table('V2_registration_scoped_contrasts.csv',contrasts)
    refrows=[]
    for training in ('public','online'):
        for mode in ('ego','slow'):
            p=C21/'coalign_reference_completion_v1'/training/mode/'evaluation'
            assert load(p/'terminal.json')['status']=='native_AP_complete_light_checked'
            for r in load(p/'panels.json'):
                for arm,m in r['metrics'].items():
                    refrows.append(dict(training=training,arm=arm,wait_ms=r['wait_ms'],AP50=100*m['global_score_sort']['ap50'],AP70=100*m['global_score_sort']['ap70'],
                        scope='fixed fusion weight, no stage1/graph; shared TraF GT / own prediction ROI; source review pending unless exact source terminal'))
    table('CoAlign_fixed_weight_ego_slow_factor.csv',refrows)
    view=load(CYCLE/'k5_physical_delta_v1/current/view_manifest.json')
    synthesis=dict(created_at=now(),official_main_unchanged=True,no_new_microvariants=True,
        V2_cached_registration=dict(panels=384,independent=ev(CYCLE/'V2_registration_independent_v1/producer/terminal.json'),coverage=load(C21/'registration_read_v1/V2/population_coverage.json'),contrasts=contrasts,
            interpretation='V2-side preregistered condition satisfied only on quality-selected subset. No whole-population physical truth or SPD conclusion. K2 cannot return to method chapter until SPD val condition also known.'),
        V2_native_registration=dict(panels=192,terminal=ev(CYCLE/'native_physical_dose_v1/V2_native_registration/terminal.json'),
            independent_review='separate native192 task actually launched; accept only its own terminal',population='strict original native saved GT, missing35 GT not filled; original source membership unchanged',
            interpretation='val paired sign reverses, hold remains negative under registration proxy. Neither omitted. No generic cross-domain method improvement.'),
        V2_R2_R3_R4=dict(identity=ev(C21/'registration_read_v1/V2/R1_R4_analysis.json'),
            scope='R2/R4 eval accepted wait0 n433; R3 n976 includes fit/cal common groups. K2 yaw Pearson .050467 weak, median error does not establish strong framewise tracking.'),
        historical_V2_276=ev(C18/'v2_historical_full_independent_v1/terminal.json'),
        K5_current_view=ev(CYCLE/'k5_physical_delta_v1/current/view_manifest.json'),K5_current_rows=view['current_rows'],K5_delta_rows=view['delta_rows'],
        GPU_chain_contract=ev(CYCLE/'coalign_upper_contract.json'),GPU_parent_only_repair=ev(CYCLE/'coalign_upper_parent_recovery_contract.json'),
        remaining=['SPD raw registration / original-timeline R1 / cached and native registration subsets',
            'A native192 independent task terminal; additional historical/native registration coverage not implicitly done',
            'B SPD full cached + native18 + new Where/ViT dose curves and plot/peak/crossing synthesis',
            'CoAlign confirmation whole independent/source checks, author and new upper factors, ViT3600/TraF2880 independent terminals',
            'C full heading+U controlled arm absent; bounded estimator-only global flip failed, do not tune noise to rescue',
            'Paper main/source/factor/figure assets full coverage remains open'],whole_release_closed=False,next_calendar='2026-11-03T03:00:00+08:00')
    atomic(CYCLE/'scientific_synthesis.json',synthesis)
    atomic(CYCLE/'release_coverage_audit.json',dict(created_at=now(),prior=ev(C21/'release_coverage_audit.json'),whole_release_closed=False,
        covered=['V2 cached registration384 independent','native V2 registration192 light checked','historical V2 independent276','fixed-weight PUBLIC/ONLINE ego/slow original AP'],remaining=synthesis['remaining']))
    atomic(CYCLE/'finite_device_coverage.json',dict(created_at=now(),through='2026-11-03T03:00:00+08:00',calendar_only=True,
        GPU0=dict(unit='embodied-peer14-c00-upper-public-parent-v2-v1-20261103',scientific_forwards=99480,chain='first PUBLIC OFFICIAL -> each confirmation PUBLIC weight ego0/slow4wait/OFFICIAL4wait',estimated_h='3-5; real scientific forward already started'),
        GPU1=dict(unit='embodied-peer14-c00-upper-online-parent-v2-v1-20261103',scientific_forwards=99480,chain='first ONLINE OFFICIAL -> each confirmation ONLINE weight ego0/slow4wait/OFFICIAL4wait',estimated_h='3-5; real scientific forward already started'),
        remote=dict(unit='embodied-peer14-c00-upper-remote-parent-v2-v1-20261103',chain='original confirmation -> author slow -> author official -> one same-budget OFFICIAL-input fit -> three-input/ego/slow val',scientific_forwards=82900,training_updates=2000,microbatches=8000,estimated_h='3-5 after original sole owner; new chain waiting, not claimed trained'),
        CPU='A remote rawSPD+cached/original-timeline R1 and native consumers; B cached/native18/backbones; native192 independent; variable waits source; K5 finite deltas; collectors',
        reference_only_not_complete_system=True,source_predictions_shared=True,no_gt_future_in_deployable_inference=True,no_model_wakeup=True))

def audit():
    local,remote=(load(CYCLE/'exit_snapshot_v1'/(h+'.json')) for h in ('local','remote'))
    aa,bb=map(load,QUEUES);ar,br=({r['id']:r for r in q['tasks']} for q in (aa,bb))
    ids=[i for i,r in ar.items() if i.startswith(('peer10','peer11','peer12','peer13','peer14')) and r.get('parent_task_id')]
    lm=[r for r in local['processes'] if r['project_model']];rm=[r for r in remote['processes'] if r['project_model']]
    units={};receipts=list((CYCLE/'launch_receipts').glob('*.json'))
    for p in receipts:
        r=load(p);cmd=['systemctl','--user','show',r['unit'],'-p','ActiveState','-p','Result','-p','CPUWeight','-p','Environment']
        units[r['unit']]=prior.output(['ssh','remote-gpu',shlex.join(cmd)] if r['remote'] else cmd)
    failures=[p for p in (CYCLE/'coalign_upper_v1').glob('*/terminal.json') if load(p)['status']=='failed_retained']
    def healthy(unit,s):
        if 'ActiveState=active' in s:return True
        if 'ActiveState=inactive' in s and 'Result=success' in s:return True
        # Completed transient services may have been GC'd; launch identity plus terminal establishes outcome.
        if 'LoadState=not-found' in s or not s:
            return False
        return False
    known_bootstrap={f'embodied-peer14-c00-upper-{t}-v1-20261103.service' for t in ('public','online')}
    checks=dict(dual_queue_owners_equal=aa['autonomy_active_owners']==bb['autonomy_active_owners'],all_candidate_rows_equal=all(ar[i]==br[i] for i in ids),
        parent_child_links=all(i in ar[ar[i]['parent_task_id']].get('child_ids',[]) for i in ids),
        terminal_owners_released=all(ar[i]['status'] not in TERMINAL for i in aa['autonomy_active_owners']),standing_parent_open=ar['autonomous_research_continuation_20261024']['status']=='open_continuing',
        local_unique_project_model_each_GPU=len(lm)==2 and len({r['uuid'] for r in lm})==2,remote_unique_project_model=len(rm)==1,
        project_model_threads2_pacing=all(r['OMP']==r['MKL']=='2' and r['pacing'] for r in lm+rm),
        thermal_both_monitoring=local['thermal']['status']==remote['thermal']['status']=='monitoring',remote450W=float(remote['GPU'].splitlines()[0].split(',')[-1])==450,
        calendar_only='OnCalendar=' in local['timer'] and 'ActiveState=active' in local['timer'],event_disabled='UnitFileState=disabled' in local['event_timer'] and 'ActiveState=inactive' in local['event_timer'],
        pending_notifications_empty=not load(AUTO/'event_dispatch/state.json').get('pending'),
        new_contract_scripts_frozen=all(ev(ROOT/load(p)['script']['path'])==load(p)['script'] for p in CYCLE.glob('*_contract.json')),
        failed_bootstraps_preserved=len(failures)==2 and (CYCLE/'coalign_upper_parent_recovery_v2/bootstrap_repair.json').exists(),
        current_units_healthy=all(healthy(u,s) or u in known_bootstrap for u,s in units.items()),
        new_unit_weights_match=all('CPUWeight=25' in s or u in known_bootstrap or ('ActiveState=inactive' in s and 'Result=success' in s) for u,s in units.items()),
        mirror_once_success=load(CYCLE/'mirror_incremental_v1/terminal.json')['exit_code']==0,
        remote_new_guard2_1=load(ROOT/'preexperiment/research_autopilot/peer14_4090_shared_disk_resource_config_v1_20261102.json')['resource']['disk_free_hard_gib']==1)
    atomic(CYCLE/'execution_and_queue_audit.json',dict(created_at=now(),status='passed' if all(checks.values()) else 'needs_specific_review',checks=checks,units=units,local_models=lm,remote_models=rm,
        bootstrap_failures=[ev(p) for p in failures],no_dispatch=True,whole_release_closed=False))
    print(json.dumps(checks,indent=2))

def finalize():
    from peer14_cycle00_v1_20261103 import record
    # Retire unused registration placeholders, not the actual open scientific children.
    routes={'upper_remote':'parent_recovery_upper_remote','upper_sources':'variable_reference_sources','upper_collect':'upper_collection'}
    for placeholder,child in routes.items():
        record(placeholder,dict(status='complete_light_checked',administrative_routing_only=True,
            scientific_successor='peer14_cycle00_'+child+'_v1_20261103',scientific_completion_not_implied=True))
    scripts=sorted(ROOT.glob('preexperiment/research_autopilot/*20261103.py'))
    for p in scripts:compile(p.read_text(),str(p),'exec')
    contract=load(CYCLE/'coalign_upper_contract.json')
    native=load(CYCLE/'V2_native_registration_independent_v1/producer/terminal.json')
    cached=load(CYCLE/'V2_registration_independent_v1/producer/terminal.json')
    checks=dict(scripts_compile=True,planned_local_forwards_each=13264+2*(3316+26528+13264)==99480,
        planned_remote_forwards=13264+39792+3316+26528==82900,
        fixed_two_confirmation_weights_per_local=len(contract['plans']['public'])==len(contract['plans']['online'])==3,
        native192_independent=native['status']=='complete_independently_reviewed' and native['panels']==192,
        cached384_independent=cached['status']=='complete_independently_reviewed' and cached['panels']==384,
        native_counts_AP_exact=native['max_AP_difference']==0 and all(native['checks'].values()),
        cached_counts_AP_exact=cached['max_AP_difference']==0 and all(cached['checks'].values()))
    assert all(checks.values())
    atomic(CYCLE/'semantic_and_freeze_checks.json',dict(created_at=now(),checks=checks,script_identities=[ev(p) for p in scripts],
        scope='compile/finite planned budgets plus actual independent geometry/AP, no extra GPU forward; not model/ICP certification'))
    atomic(CYCLE/'scientific_publication_addendum.json',dict(created_at=now(),prior=ev(CYCLE/'scientific_synthesis.json'),
        native192_independent=ev(CYCLE/'V2_native_registration_independent_v1/producer/terminal.json'),
        cached384_independent=ev(CYCLE/'V2_registration_independent_v1/producer/terminal.json'),
        historical276_independent=ev(C18/'v2_historical_full_independent_v1/terminal.json'),
        native192_now_accepted=True,actual_native_review_cost=native['cost'],
        checks=ev(CYCLE/'semantic_and_freeze_checks.json'),administrative_routing=routes,
        original_release_closed=False,no_physical_truth_or_detector_claim=True,next_calendar='2026-11-03T03:00:00+08:00'))
    aa,bb=map(load,QUEUES);ar,br=({r['id']:r for r in q['tasks']} for q in (aa,bb))
    checks=dict(load(CYCLE/'execution_and_queue_audit.json')['checks'])
    checks.update(dual_queue_owners_equal=aa['autonomy_active_owners']==bb['autonomy_active_owners'],
        all_candidate_rows_equal=all(ar[i]==br[i] for i in ar if i.startswith(('peer10','peer11','peer12','peer13','peer14')) and ar[i].get('parent_task_id')),
        parent_child_links=all(i in ar[r['parent_task_id']].get('child_ids',[]) for i,r in ar.items() if i.startswith(('peer10','peer11','peer12','peer13','peer14')) and r.get('parent_task_id')),
        terminal_owners_released=all(ar[i]['status'] not in TERMINAL for i in aa['autonomy_active_owners']),
        unused_placeholders_routed=all(ar['peer14_cycle00_'+k+'_v1_20261103']['owner'] is None for k in routes))
    assert all(checks.values())
    atomic(CYCLE/'execution_publication_audit.json',dict(created_at=now(),status='passed',checks=checks,
        resource_snapshot=ev(CYCLE/'execution_and_queue_audit.json'),semantic=ev(CYCLE/'semantic_and_freeze_checks.json'),
        no_dispatch=True,no_model_wakeup=True,next_calendar='2026-11-03T03:00:00+08:00'))
    pub=CYCLE/'publication_v1';pub.mkdir(exist_ok=False)
    from run_cycle import prepare_handoff
    metadata=prepare_handoff(pub)
    atomic(AUDIT/'scoreboard_latest.json',dict(created_at=now(),cycle=str(CYCLE.relative_to(ROOT)),
        scientific_synthesis=ev(CYCLE/'scientific_synthesis.json'),scientific_completion=ev(CYCLE/'scientific_publication_addendum.json'),
        K5_current_view=ev(CYCLE/'k5_physical_delta_v1/current/view_manifest.json'),
        execution_audit=ev(CYCLE/'execution_publication_audit.json'),device_coverage=ev(CYCLE/'finite_device_coverage.json'),
        publication=ev(pub/'handoff_metadata.json'),original_release_closed=False,next_calendar='2026-11-03T03:00:00+08:00'))
    print(json.dumps(dict(status='published',publication=metadata['mode'],checks=checks),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mirror','capture','reconcile','assets','audit','finalize']);globals()[p.parse_args().mode]()
