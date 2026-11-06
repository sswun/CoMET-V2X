"""Peer10 immutable registration, shared references and non-model task chains."""
from __future__ import annotations
import argparse, datetime as dt, fcntl, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / 'artifacts/research_autopilot'
AUDIT = AUTO / 'status_audit_20261101_peer10_tournament_v1'
PARENT = 'autonomous_research_continuation_20261024'
QUEUES = [AUTO/'deadline_20261008/decision_queue.json', AUTO/'baseline_data_expansion_20261003/expansion_queue.json']
TASKS = {'K1':'identity_and_scoreboard', 'K2':'infra_se2_formal', 'K3':'expanded_anchor_formal', 'K4':'correction_aware_native_finetune', 'K5':'label_geometry_auxiliary'}
def now(): return dt.datetime.now().astimezone().isoformat(timespec='seconds')
def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): d.update(b)
    return d.hexdigest()
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.tmp.{os.getpid()}');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def load(p): return json.loads(Path(p).read_text())
def ev(p):
    p=Path(p);return {'path':str(p.relative_to(ROOT)), 'sha256':sha(p), 'bytes':p.stat().st_size}
def task(k): return f'peer10_{TASKS[k]}_v1_20261101'
def queues(updates):
    stamp=now()
    with (AUTO/'.queue_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for p in QUEUES:
            j=load(p);parent=next(x for x in j['tasks'] if x['id']==PARENT)
            for k,values in updates.items():
                tid=task(k); rows=[x for x in j['tasks'] if x['id']==tid]
                if not rows:
                    row={'id':tid,'parent_task_id':PARENT,'candidate':k,'created_at':stamp};j['tasks'].append(row)
                else:
                    assert len(rows)==1;row=rows[0]
                row.update(values,updated_at=stamp)
                if tid not in parent.setdefault('child_ids',[]): parent['child_ids'].append(tid)
                owner='research_autopilot_'+tid
                if values.get('status') in ('complete_light_checked','complete_independently_reviewed','failed_retained'):
                    row['owner']=None;j.setdefault('autonomy_active_owners',{}).pop(tid,None)
                    if tid not in parent.setdefault('completed_child_ids',[]): parent['completed_child_ids'].append(tid)
                else:
                    row['owner']=owner;j.setdefault('autonomy_active_owners',{})[tid]=owner
            parent.update(status='open_continuing',updated_at=stamp)
            j['latest_peer10_tournament']={'path':str(AUDIT.relative_to(ROOT)),'updated_at':now(),'model_wakeup':'3h_timer_only'}
            j['updated_at']=now();atomic(p,j)
        a,b=map(load,QUEUES)
        assert a['autonomy_active_owners']==b['autonomy_active_owners']
        for k in updates:
            ar=next(x for x in a['tasks'] if x['id']==task(k));br=next(x for x in b['tasks'] if x['id']==task(k));assert ar==br
        atomic(AUDIT/'queue_consistency_latest.json',{'status':'passed','time':now(),'task_ids':[task(k) for k in updates], 'owners':a['autonomy_active_owners'],'standing_parent_open':True})
def prepare():
    # Mandatory re-read at new-successor registration; old conflicts are history.
    refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in refs:(AUTO/n).read_text()
    if (AUDIT/'contract_v1.json').exists():raise FileExistsError(AUDIT)
    fair=load(AUTO/'status_audit_20261031_motion_direction_information_state_fair_controls_v1/contract_v1.json')
    fc=load(AUTO/'status_audit_20261031_motion_direction_information_state_actual_v1/fitcal_contract_recovery_v2.json')
    inputs=dict(fair['content_addressed_inputs'])
    for name,key in [('calibration8_material','calibration_material'),('fit_conditions','train_conditions'),('fit_q005_frame','fit_q005_frame')]:
        if key in fc['inputs']:inputs[name]=ev(ROOT/fc['inputs'][key]['path'])
    inputs['fit_q005_frame']=ev(AUTO/'status_audit_20261029_strict60_q005_task_refit_portability_v1/producer_v1/fit24_frame_and_residual.npz')
    inputs['head_maps']=ev(AUTO/'status_audit_20261029_strict60_q005_matched39d_task_opportunity_v1/producer_v1/q005_actual_matched39d_head/source_maps.json')
    contract={'schema':'peer10_tournament_v1','created_at':now(),'handoff_read':refs,'inputs':inputs,
      'official_protocol_unchanged':True,'model_wakeup':'3h_timer_only; terminal files only; no event dispatch',
      'K2':{'theta_prior_degrees':[.15,.3,.6],'fit_selection':'mean fit24 official AP70, pooled 0/400; no hold/val selection','calibration':'report selected vs baseline once; no sign gate','Q_xy_m':.05,'Q_theta_degrees':.01,'R_m':.6,'state':'t_x,t_y,theta; current ego transport and moving pivot Jacobian','geometry':'exact SE2 rotation of remote box centres and yaw; same rotation in native pointcloud plugin','controls':['ego_pivot','free_theta_prior','static_only_theta','fit_constant_theta'],'readout':'614b7e2c frozen matched q005 U first; own matched U separate successor'},
      'K3':{'hypothesis':'non-protected ego boxes add legal no-anchor measurements, unlike old weak-anchor gate','fit_variants':['expanded_all','expanded_score030'],'selection':'mean fit AP70 then report cal; freeze globally; no formal cell selection','combinations':['K1 translation','K2 SE2']},
      'K4':{'arms':['PUBLIC','ONLINE_K1','OFFICIAL_REFERENCE'],'seed':20261101,'training_roles':['fit24'],'training_waits_ms':[0,400],'evaluation_waits_ms':[0,100,200,400],'updates':2000,'gradient_accumulation':4,'lr':.0001,'optimizer':'AdamW default weight_decay .01','trainable':['fusion_net','aspp','head','shrink_conv_if_present'],'frozen':['reader','backbone (eval BN)'],'loss':'author anchor cls+box only, current cooperative GT targets; no history/remote GT, no trajectory loss','initial_weight':ev(ROOT/'artifacts/checkpoints/traf_align_v2xseq_0f6f98de/net_epoch60.pth'),'sampling':'same shuffled 7524 cached conditions for all three arms, repeated to 8000 microbatches; no val checkpoint selection','qualification':'first 4 updates actual backward and finite gradients; then same run continues, final checkpoint fixed','prediction':'online adaptation may reduce residual feature misregistration, or public adaptation may be as good; preserve all arms'},
      'K5':{'prediction':'inf-only public-label re-registration can separate annotation geometry from detection quality; official main AP unchanged','rules':'peer09 section4; label/token/calibration only; fit24 choose global estimator, cal8 report; only inf exclusive rewritten; <3 co-labelled matches fallback official; two time interpolation sensitivity arms'},
      'storage':{'shared_cache':'reference only; no copies of boxes/GT/features','expected_new_bytes':'CPU <1GiB; three rolling+final training checkpoints <1GiB total plus sharded native predictions','temporary':'only rolling optimizer state; retained while active','no_soft_budget_stop':True}}
    for k in TASKS:contract[k if k!='K1' else 'baseline_identity']=contract.get(k,{'recipe':'q005 KF + heading + matched39D U; exact saved base_delta, not anchor gate/direction'})
    atomic(AUDIT/'contract_v1.json',contract)
    predictions={'K1':'exact replay; no new effect expected','K2':'rotation lever arm can improve registration but hurt inf-labelled targets','K3':'new legal ego anchors may help unsupported frames or add wrong matches','K4':contract['K4']['prediction'],'K5':contract['K5']['prediction']}
    queues({k:{'status':'registered_ready','contract':ev(AUDIT/'contract_v1.json'),'prediction_before_result':predictions[k],'estimated_cost':contract.get(k,{}).get('updates','CPU cached AP; K2 also 13264 native GPU forwards'),'next_action':'execute producer, light self-check, retain terminal for next 3h cycle','release_coverage':'K1–K5 current tournament; prior releases/negative evidence not superseded'} for k in TASKS})
    print(AUDIT)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','audit']);a=p.parse_args()
    if a.command=='prepare':prepare()
    else:
        a,b=map(load,QUEUES);assert a['autonomy_active_owners']==b['autonomy_active_owners'];print(json.dumps(a['autonomy_active_owners']))
if __name__=='__main__': main()
