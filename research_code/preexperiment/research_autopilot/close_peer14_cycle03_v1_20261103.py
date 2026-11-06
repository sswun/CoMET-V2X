"""03h additive scientific assets and exit audit; calendar only, no dispatch."""
import argparse, csv, inspect, json, shlex, subprocess, time
from pathlib import Path
from collections import defaultdict
from peer14_cycle03_v1_20261103 import ROOT,AUTO,AUDIT,CYCLE,C00,C21,C18,C15,REMOTE_ROOT,atomic,ev,load,now,record,update,task
from peer10_tournament_v1_20261101 import QUEUES
import close_peer14_cycle00_v1_20261103 as prior
TERMINAL=prior.TERMINAL

def table(name,rows):
    if not rows:return
    with (CYCLE/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def quality():
    raw=C21/'raw_registration_v1/producer';q=load(raw/'quality_frozen.json')['thresholds']['SPD'];accepted=[];spd=[]
    for r in map(json.loads,(raw/'registrations.jsonl').open()):
        if r['dataset']!='SPD':continue
        spd.append(r['index'])
        if not r.get('error') and r['correspondences']>=100 and r['fitness']>=q['fitness_min'] and r['inlier_RMSE']<=q['RMSE_max'] and r['planar_H_min_normalized']>=q['H_min'] and all(abs(x)<=2 for x in r['roll_pitch_deg']) and abs(r['delta_z'])<=1:accepted.append(r['index'])
    canonical=load(raw/'accepted_indices.json');expected=set(canonical)&set(spd)
    checks=dict(exact_SPD_set=set(accepted)==expected,canonical_has_no_extra_nonSPD=not(set(canonical)-set(spd)),no_duplicates=len(accepted)==len(set(accepted)))
    assert all(checks.values())
    atomic(CYCLE/'SPD_quality_set_independent_audit.json',dict(created_at=now(),checks=checks,accepted=len(accepted),raw=ev(raw/'terminal.json'),thresholds=ev(raw/'quality_frozen.json'),scope='supplements running independent reviewer with full canonical equality, no science script hot edit',GPU_forwards=0))

def assets():
    reg=load(C21/'registration_read_v1/SPD/panels.json');table('SPD_registration_cached_all2720.csv',reg)
    cm=prior.means(reg,'ap',('dataset','family','role','method','protocol','source_group','threshold'));table('SPD_registration_cached_means.csv',cm)
    comparisons=[]
    for dataset,path,field in [('SPD',C21/'registration_read_v1/SPD/panels.json','ap'),('V2',C21/'registration_read_v1/V2/panels.json','ap')]:
        rows=load(path)
        for role in ('val21','mechanism_holdout14'):
            for protocol in ('official_quality_subset','registration_SE2_z_quality_subset'):
                values={}
                for method in ('K1_exact','K2_infra'):
                    rr=[r[field] for r in rows if r['family']=='current' and (r['role'],r['protocol'],r['method'],r['source_group'],r['threshold'])==(role,protocol,method,'all',70)]
                    assert len(rr)==(4 if dataset=='SPD' else 2);values[method]=sum(rr)/len(rr)
                comparisons.append(dict(dataset=dataset,role=role,protocol=protocol,**values,K2_minus_K1_pp=values['K2_infra']-values['K1_exact'],scope='same fit-quality-conditioned population, not full main AP'))
    table('A_paired_quality_population_comparisons.csv',comparisons)
    vals=[r for r in comparisons if r['role']=='val21' and r['protocol']=='registration_SE2_z_quality_subset'];assert len(vals)==2
    branch=all(r['K2_minus_K1_pp']>0 for r in vals)
    r1=load(C21/'R1_original_time_v2/producer/R1_preregistered_read.json');assert not r1['point_gate_passed']
    dose=C21/'spd_dose_response_v1/producer';assert load(dose/'terminal.json')['status']=='complete_light_checked'
    rows=load(dose/'panels.json');table('SPD_fixed_prediction_all8400_dose_panels.csv',rows)
    groups=defaultdict(dict)
    for r in rows:
        if r['protocol'] not in ('alpha000','alpha025','alpha050','alpha075','official'):continue
        for th in (50,70):groups[r['family'],r['role'],r['method'],r['source_group'],th].setdefault(r['protocol'],[]).append(r['ap'+str(th)])
    curves=[];peaks=[];alpha=dict(alpha000=0.,alpha025=.25,alpha050=.5,alpha075=.75,official=1.)
    for key,protocols in groups.items():
        assert len(protocols)==5
        curve=[]
        for p,vs in protocols.items():
            assert len(vs)==4
            row=dict(zip(('family','role','method','source_group','threshold'),key),alpha=alpha[p],AP_mean=sum(vs)/len(vs),n_waits=4);curves.append(row);curve.append(row)
        best=max(r['AP_mean'] for r in curve)
        peaks.append(dict(zip(('family','role','method','source_group','threshold'),key),peak_alphas=[r['alpha'] for r in curve if abs(r['AP_mean']-best)<1e-10],peak_AP=best,AP_alpha0=next(r['AP_mean'] for r in curve if r['alpha']==0),AP_alpha1=next(r['AP_mean'] for r in curve if r['alpha']==1)))
    table('SPD_dose_mean_curve_figure_data.csv',curves);table('SPD_dose_mean_peaks.csv',peaks)
    crosses=load(dose/'crossings.json');table('SPD_dose_pairwise_crossings.csv',crosses)
    focus=[r for r in peaks if r['family']=='current' and r['source_group']=='all' and r['threshold']==70]
    factor=[]
    roots=[C15/'coalign_full_v1',C18/'coalign_confirmation_v1',C00/'coalign_upper_parent_recovery_v2',C21/'coalign_reference_completion_v1',C21/'coalign_author_reference_v1']
    for root in roots:
        for p in root.rglob('panels.json'):
            if any(x in str(p) for x in ('_profile','aborted','source','independent')):continue
            term=p.parent/'terminal.json'
            if not term.exists() or load(term).get('status')!='native_AP_complete_light_checked':continue
            for r in load(p):
                if 'metrics' not in r:continue
                for arm,m in r['metrics'].items():factor.append(dict(producer=str(p.parent.relative_to(AUDIT)),arm=arm,wait_ms=r['wait_ms'],AP50=100*m['global_score_sort']['ap50'],AP70=100*m['global_score_sort']['ap70'],review='light_checked; full independent status separate',scope='own CoAlign readout/shared TraF GT; FULL only named graph arms, annotation input reference nondeployable'))
    table('CoAlign_all_completed_weight_input_factors.csv',factor)
    ego=[];src=C00/'coalign_variable_reference_sources_v2/native_source_aux'
    for roi in ('native','common'):
        for threshold in (.5,.7):
            vals={}
            for t in ('public','online'):
                # The first own-weight actual-wait source consumers are complete.
                ep=src/(C21/'coalign_reference_completion_v1'/t/'ego/evaluation').relative_to(AUDIT)
                dp=C18/'coalign_sources_ROI_v2/native_source_aux'/(C15/'coalign_full_v1'/('adapted_evaluation' if t=='public' else 'online_adapted_evaluation')).relative_to(AUDIT)
                assert load(ep/'terminal.json')['status']==load(dp/'terminal.json')['status']=='complete_light_checked'
                def select(p,arm):
                    rr=[r for r in load(p/'panels.json') if (r['protocol'],r['roi'],r['source_group'],r['threshold'],r['wait_ms'],r['arm'])==('official',roi,'veh_coop',threshold,0,arm)];assert len(rr)==1;return rr[0]['ap']
                vals[t]=select(dp,'NO_GRAPH' if t=='public' else 'K1_INPUT')-select(ep,'EGO_ONLY_REFERENCE')
            diff=vals['online']-vals['public'];ego.append(dict(roi=roi,threshold=threshold,PUBLIC_drop_vc=vals['public'],ONLINE_drop_vc=vals['online'],difference_pp=diff,source='official veh_coop at wait0 only'))
    table('CoAlign_first_weight_ego_preregistered_source_read.csv',ego)
    native=[r for r in ego if r['roi']=='native'];support=next(r['difference_pp'] for r in native if r['threshold']==.7)>=.5 or (next(r['difference_pp'] for r in native if r['threshold']==.5)>=1 and next(r['difference_pp'] for r in native if r['threshold']==.7)>0)
    summary=dict(created_at=now(),A=dict(comparisons=comparisons,K2_existing_estimator_branch_light_passed=branch,scope='quality-selected val populations only; SPD independent pending, V2 passed; no new estimator variants',R1=dict(MA5=r1['original_time_MA5_correlation'],jitter=r1['original_time_jitter_correlation'],seqmean_descriptive_only=r1['seqmean_correlation'],point_gate_passed=False,interpretation='No physically-unrecoverable-jitter claim'),R2_R4=ev(C21/'registration_read_v1/SPD/R1_R4_analysis.json')),
        B=dict(panels=len(rows),official_endpoint_replay=True,curve_peak_focus=focus,comparisons=len(crosses),comparisons_with_rank_flip=sum(r['strict_rank_flips']>0 for r in crosses),total_strict_rank_flips=sum(r['strict_rank_flips'] for r in crosses),scope='fixed predictions, each own population; peaks descriptive not selected method; independent pending'),
        CoAlign=dict(completed_factor_rows=len(factor),ego_source=ego,shared_evidence_suppression_supported=support,scope='first-weight veh_coop only; no inf-only/global drop inference'),
        remaining=['SPD cached A independent terminal','native SPD registration and native18/backbone dose consumers','complete CoAlign/ViT/TraF/currentV2 independent release coverage','new fixed K2/ICP input GPU predictions and source/registration CPU consumers','OPV2V full heading+U controlled arm absent, estimator-only global flip failed; no noise chasing','whole paper main/source/factor/figure release, no final innovation choice'],whole_release_closed=False,next_calendar='2026-11-03T06:00:00+08:00')
    atomic(CYCLE/'scientific_synthesis.json',summary)
    atomic(CYCLE/'release_coverage_audit.json',dict(created_at=now(),prior=ev(C00/'release_coverage_audit.json'),remaining=summary['remaining'],whole_release_closed=False))
    print(json.dumps(dict(A=vals if False else comparisons,peaks=focus,ego=ego,rank_flip_comparisons=summary['B']['comparisons_with_rank_flip']),indent=2))

def mirror():
    ns=dict(prior.__dict__,CYCLE=CYCLE)
    s=inspect.getsource(prior.mirror).replace('conflicts_cycle00_20261103','conflicts_cycle03_20261103')
    exec(compile(s,__file__+'::mirror03','exec'),ns);ns['mirror']()

def snapshot():
    d=prior.snapshot()
    for p in CYCLE.rglob('*.json'):
        if any(x in str(p) for x in ('exit_snapshot','publication','conflicts')):continue
        if p.name not in ('terminal.json','status.json') and 'queue_delta' not in p.name:continue
        d['files'][str(p.relative_to(ROOT))]=dict(identity=ev(p),value=load(p))
    return d

def capture():
    out=CYCLE/'exit_snapshot_v1';out.mkdir(exist_ok=False);atomic(out/'local.json',snapshot())
    code='from close_peer14_cycle00_v1_20261103 import snapshot;import json;print(json.dumps(snapshot()))'
    raw=prior.prior.output(['ssh','remote-gpu','cd '+str(REMOTE_ROOT)+' && PYTHONPATH=preexperiment/research_autopilot .conda/opencood-linux/bin/python -c '+shlex.quote(code)])
    atomic(out/'remote.json',json.loads(raw))

def reconcile():
    ns=dict(prior.__dict__,CYCLE=CYCLE);exec(compile(inspect.getsource(prior.reconcile),__file__+'::reconcile03','exec'),ns);ns['reconcile']()

def audit():
    local,remote=(load(CYCLE/'exit_snapshot_v1'/(h+'.json')) for h in ('local','remote'))
    a,b=map(load,QUEUES);aa,bb=({r['id']:r for r in q['tasks']} for q in (a,b));ids=[k for k,r in aa.items() if k.startswith(('peer10','peer11','peer12','peer13','peer14')) and r.get('parent_task_id')]
    lm=[r for r in local['processes'] if r['project_model']];rm=[r for r in remote['processes'] if r['project_model']]
    checks=dict(dual_queue_owners_equal=a['autonomy_active_owners']==b['autonomy_active_owners'],candidate_rows_equal=all(aa[k]==bb[k] for k in ids),parent_child_links=all(k in aa[aa[k]['parent_task_id']].get('child_ids',[]) for k in ids),terminal_owners_released=all(aa[k]['status'] not in TERMINAL for k in a['autonomy_active_owners']),standing_parent_open=aa['autonomous_research_continuation_20261024']['status']=='open_continuing',
        two_local_unique_project_models=len(lm)==2 and len({r['uuid'] for r in lm})==2,one_remote_project_model=len(rm)==1,threads2_pacing=all(r['OMP']==r['MKL']=='2' and r['pacing'] for r in lm+rm),thermal_monitoring=local['thermal']['status']==remote['thermal']['status']=='monitoring',remote450W=float(remote['GPU'].splitlines()[0].split(',')[-1])==450,calendar_only='OnCalendar=' in local['timer'] and 'ActiveState=active' in local['timer'],event_disabled='UnitFileState=disabled' in local['event_timer'] and 'ActiveState=inactive' in local['event_timer'],notifications_empty=not load(AUTO/'event_dispatch/state.json').get('pending'),mirror_once_success=load(CYCLE/'mirror_incremental_v1/terminal.json')['exit_code']==0,new_contract_scripts_frozen=all(ev(ROOT/load(p)['script']['path'])==load(p)['script'] for p in CYCLE.glob('*_contract.json')),exact_SPD_quality_set=all(load(CYCLE/'SPD_quality_set_independent_audit.json')['checks'].values()))
    atomic(CYCLE/'execution_and_queue_audit.json',dict(created_at=now(),status='passed' if all(checks.values()) else 'needs_specific_review',checks=checks,local_models=lm,remote_models=rm,no_dispatch=True))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['quality','assets','mirror','capture','reconcile','audit']);globals()[p.parse_args().mode]()
