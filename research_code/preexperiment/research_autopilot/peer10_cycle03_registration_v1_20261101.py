"""Register independent, reference-only successors under their existing owners."""
import fcntl
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,QUEUES,atomic,ev,load,now,task

CYCLE=AUDIT/'cycle03_v1'
JOBS={
 'native_source_aux':('peer10_native_source_aux_v1_20261101',task('K5')),
 'v2_transfer':('peer10_current_se2_expanded_v2v4real_v1_20261101',task('K2')),
 'ffnet_host_retry':('peer10_ffnet_official_host_retry_v1_20261101','ffnet_isolated_runtime_and_strict_adapter_v1_20261031_main'),
}

def update(name,values):
    tid,parentid=JOBS[name];stamp=now()
    with (AUTO/'.queue_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for path in QUEUES:
            q=load(path);parents=[r for r in q['tasks'] if r['id']==parentid];assert len(parents)==1
            found=[r for r in q['tasks'] if r['id']==tid];assert len(found)<=1
            row=found[0] if found else {'id':tid,'parent_task_id':parentid,'created_at':stamp}
            if not found:q['tasks'].append(row)
            owner=parents[0].get('owner') or 'research_autopilot_'+tid
            row.update(values,updated_at=stamp)
            finished=values.get('status') in ('complete_light_checked','failed_retained','download_failed_retained')
            row['owner']=None if finished else owner
            if finished:q.setdefault('autonomy_active_owners',{}).pop(tid,None)
            else:q.setdefault('autonomy_active_owners',{})[tid]=owner
            if tid not in parents[0].setdefault('child_ids',[]):parents[0]['child_ids'].append(tid)
            if finished and tid not in parents[0].setdefault('completed_child_ids',[]):parents[0]['completed_child_ids'].append(tid)
            q['updated_at']=stamp;atomic(path,q)
        a,b=map(load,QUEUES);assert a['autonomy_active_owners']==b['autonomy_active_owners']
        assert next(r for r in a['tasks'] if r['id']==tid)==next(r for r in b['tasks'] if r['id']==tid)

def main():
    CYCLE.mkdir(exist_ok=False)
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    old=AUTO/'status_audit_20261030_v2v4real_current_q005_heading_anchor_gate_factorial_v1'
    v2=load(old/'contract_recovery_v5.json')
    bound=[ev(old/'contract_recovery_v5.json'),ev(old/'terminal_recovery_v6.json')]
    for spec in v2['inputs'].values():
        if spec['path'].endswith(('.npz','.pth','.jsonl')):bound.append(spec)
    for role in ('fit24','calibration8','mechanism_holdout14','val21'):bound.append(ev(old/'producer_v1'/f'{role}_q005_state.npz'))
    bound.extend(ev(old/'producer_v1/q005_cv_matched39d_head'/f) for f in ('direct_u.pth','source_maps.json','training_schedule.npz'))
    contract={'created_at':now(),'latest_instructions':refs,'model_wakeup':'3h timer only; no terminal dispatch',
      'native_source_aux':{'inputs':'exact saved peer10 native shards/records/terminal + frozen K5 labels; dynamically bind each ready immutable producer before scoring',
        'methods':['k2_native','k4_eval_public','k4_eval_online_k1','k4_eval_official_reference','seed2_public/evaluation','seed2_online_k1/evaluation','seed2_official_reference/evaluation'],
        'protocols':['official','interpolated','no_interpolation'],'rois':['native','common'],'groups':['all','veh','coop','inf','veh_coop'],
        'native_eval_range':[0.,-40.,-3.,96.,40.,1.5],
        'auxiliary_dimension_conversion':'K5 cached l,w,h -> author native h,w,l via indices [5,4,3]; yaw unchanged',
        'auxiliary_eligibility':'freeze official native/common GT membership and prediction ROI/IDs/order/scores; only matched inf box geometry changes, not denominators',
        'prediction':'SE2 may lose inf but gain veh/coop; auxiliary labels may change source ordering. No prediction or NMS changes, no auxiliary selection.',
        'checks':['unique source joins under 1mm','all official native AP replay','veh/coop labels exactly unchanged','same frozen prediction IDs/order/scores for both auxiliary protocols','official original GT SHA before/after'],
        'estimated_cost':{'new_GPU_forwards':0,'updates':0,'CPU_threads':1,'wall_hours_per_method':'0.5-4','new_disk_mib_per_method':'20-100'},
        'storage':'reference native shards and label increments; save only outcomes/metrics, no duplicate GT/geometry'},
      'v2_transfer':{'bound_inputs':bound,'base_contract':ev(old/'contract_recovery_v5.json'),'base_producer':str((old/'producer_v1').relative_to(ROOT)),
        'prediction':'moving-peer pivot/ego transport may help or hurt; preserve all signs, do not substitute old anchor-gate recipe',
        'state':'reuse exact current K2 SE2 and K3 expanded kernel; actual source pose at source time, U96 state key, own V2 q005-U',
        'prior_degrees':.15,'K3_score_floor':0.,'fit_selection':'none; frozen main SPD fit parameters transferred',
        'calibration_fallback':'one global binary per candidate: choose candidate iff mean cal8 official AP70 >= current exact K1, else K1; report raw and fallback, never formal cell selection',
        'historical_protocol':'0/400ms V2 frozen candidate/GT population, no post-geometry ROI mask; not SPD four-wait main protocol',
        'estimated_cost':{'new_GPU_forwards':0,'new_updates':0,'CPU_threads':1,'wall_hours':'0.1-1','new_disk_mib':60}},
      'ffnet_host_retry':{'source':'author-linked same public Drive file 1k-WIcGvGY4YF02ITMTTIDdPblA2cjaji',
        'prediction':'official-host redirect correction may acquire weight; textual quota/error retained and not installed',
        'allowed_hosts':['drive.google.com','drive.usercontent.google.com'],'request_limit':4,'one_shot':True,'proxy':'http://127.0.0.1:7897',
        'no_bypass':'no mirrors, identities, quota bypass, global proxy changes or external contact',
        'stop_loss':'2026-11-03T24:00+08:00','estimated_cost':'single bounded download; CPU verification if acquired; no GPU until verified',
        'storage':'one author checkpoint plus hash, no duplicate target, error body only on failure'},
      'scripts':{n:ev(ROOT/'preexperiment/research_autopilot'/n) for n in ('run_peer10_native_source_aux_v1_20261101.py','run_peer10_v2_transfer_v1_20261101.py','retry_peer10_ffnet_official_host_v1_20261101.py')},
      'official_protocol_and_split_unchanged':True}
    atomic(CYCLE/'contract.json',contract)
    for name in JOBS:update(name,{'status':'registered_ready','contract':ev(CYCLE/'contract.json'),'prediction_before_result':contract[name]['prediction'],'estimated_cost':contract[name]['estimated_cost'],'next_action':'execute independent controlled worker; no model notification'})
    print(CYCLE)
if __name__=='__main__':main()
