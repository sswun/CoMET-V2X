"""06h immutable registrations and finite CPU successors, never a model watcher."""
import fcntl
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,QUEUES,atomic,ev,load,now,task

CYCLE=AUDIT/'cycle06_v1'
JOBS={'zero_theta':('peer10_true_zero_theta_prior_control_v1_20261101',task('K2')),
      'fixed_population_aux':('peer10_fixed_population_all_cached_aux_v1_20261101',task('K5'))}

def update(name,values):
    tid,pid=JOBS[name];stamp=now()
    with (AUTO/'.queue_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        canonical=next((r for r in load(QUEUES[0])['tasks'] if r['id']==tid),None)
        for path in QUEUES:
            q=load(path);parent=next(r for r in q['tasks'] if r['id']==pid)
            found=[r for r in q['tasks'] if r['id']==tid];assert len(found)<=1
            row=found[0] if found else {'id':tid,'parent_task_id':pid,'created_at':stamp}
            if not found:q['tasks'].append(row)
            row['created_at']=canonical['created_at'] if canonical else stamp
            row.update(values,updated_at=stamp);finished=values.get('status') in ('complete_light_checked','failed_retained')
            row['owner']=None if finished else parent.get('owner') or 'research_autopilot_'+pid
            if finished:q.setdefault('autonomy_active_owners',{}).pop(tid,None)
            else:q.setdefault('autonomy_active_owners',{})[tid]=row['owner']
            if tid not in parent.setdefault('child_ids',[]):parent['child_ids'].append(tid)
            if finished and tid not in parent.setdefault('completed_child_ids',[]):parent['completed_child_ids'].append(tid)
            q['updated_at']=now();atomic(path,q)
        a,b=map(load,QUEUES);assert a['autonomy_active_owners']==b['autonomy_active_owners']
        assert next(r for r in a['tasks'] if r['id']==tid)==next(r for r in b['tasks'] if r['id']==tid)

def main():
    CYCLE.mkdir(exist_ok=False)
    instructions={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in instructions:(AUTO/n).read_text()
    contracts={k:ev(AUTO/(d+'/contract_v1.json')) for k,d in {
        'geometry':'status_audit_20261031_peer07_geometry_matched_source_ap_v1',
        'modules':'status_audit_20261031_peer07_roco_cbm_original_source_ap_v1',
        'ego':'status_audit_20261031_peer07_ego_original_source_ap_recovery_v2',
        'official_late':'status_audit_20261031_peer07_official_late_source_ap_v1'}.items()}
    c={'created_at':now(),'latest_instructions':instructions,'official_main_protocol_unchanged':True,
       'model_wakeup':'3h timer only; finite workers do not dispatch events or models',
       'shared_main_contract':ev(AUDIT/'contract_v1.json'),
       'frozen_fit':ev(AUDIT/'k2_k3_cached_recovery_v2/fit_freeze.json'),
       'cached_formal_terminal':ev(AUDIT/'k2_k3_formal_resume_v3/terminal.json'),
       'label_terminal':ev(AUDIT/'k5_labels_recovery_v2/terminal_v2.json'),
       'baseline_contracts':contracts,
       'zero_theta':{'prediction':'An exact improper zero-angle-information prior may behave differently from finite 3deg; retain all outcomes, no hyperparameter selection.',
           'state':'same causal observations, moving infra pivot and ego transport, Q/R as frozen K2; initial information diag(1,1,0), eta=0; exact canonical Gaussian process update; no finite diffuse proxy',
           'unobservable_policy':'if information rank deficient, theta=0 gauge and conditional translation solve for output only; retain singular canonical state, never feed gauge back',
           'evaluation':'fit/cal report without reselection; hold14/val21 official five-source AP, same frozen q005-U; no GPU',
           'estimated_cost':{'CPU_threads':1,'wall_hours':'1-4','new_disk_mib':80,'GPU_forwards':0,'updates':0}},
       'fixed_population_aux':{'prediction':'Label sensitivity can differ by geometry/readout; report all registered methods, not only favourable candidates.',
           'families':['current K1/K2/K3 controls + K2 own matched U','canonical stale/CV/KF/FreeAlign raw and matched U','original stale/KF/FreeAlign/RoCo/CBM raw and fresh39d','original ego','original official late no/time compensation'],
           'population':'per original method identity, freeze official GT membership and saved prediction IDs/order/scores/ROI; replace only inf box geometry; do NOT reapply auxiliary-centre ROI',
           'protocols':['official','interpolated','no_interpolation'],'groups':['all','veh','coop','inf','veh_coop'],
           'roles':['mechanism_holdout14','val21'],'waits_ms':[0,100,200,400],
           'checks':['all official saved TP/AP replay','unique token joins under 1mm','veh/coop unchanged','input SHA verification','no inference or auxiliary selection'],
           'boundary':'method populations and weights remain distinct; module baselines are not full-system reproductions; fixed population is a supplementary sensitivity not the older auxiliary-centre-ROI protocol',
           'estimated_cost':{'CPU_threads':1,'wall_hours':'4-12','new_disk_mib':300,'GPU_forwards':0,'updates':0}},
       'storage':'Reference shared material, saved predictions, geometry, labels and weight contracts. Persist only small metrics, per-sequence rows and compressed TP/FP increments. No duplicate pointcloud, GT or weights.',
       'scripts':{n:ev(ROOT/'preexperiment/research_autopilot'/n) for n in ('run_peer10_true_zero_theta_prior_v1_20261101.py','run_peer10_fixed_population_aux_v1_20261101.py')}}
    atomic(CYCLE/'contract.json',c)
    atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
    for name in JOBS:update(name,{'status':'registered_ready','contract':ev(CYCLE/'contract.json'),'prediction_before_result':c[name]['prediction'],'estimated_cost':c[name]['estimated_cost']})
    print(CYCLE)
if __name__=='__main__':main()
