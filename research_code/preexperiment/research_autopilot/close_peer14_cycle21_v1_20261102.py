"""Small exit receipts, remote deltas and paper assets; no dispatch or du."""
import argparse,csv,json,shlex,subprocess
from pathlib import Path
from peer14_cycle21_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C15,C18,REMOTE_ROOT,atomic,ev,load,now,update,task,record
from peer10_tournament_v1_20261101 import QUEUES
TERMINAL={'complete_light_checked','complete_independently_reviewed','failed_retained'}

def output(cmd):return subprocess.check_output(cmd,text=True).strip()

def snapshot():
    files={}
    for root in (C15,C18,CYCLE):
        for p in root.rglob('*.json'):
            if any(s in str(p) for s in ('aborted_disk_guard','entry_execution','final_execution','exit_snapshot','conflicts')):continue
            if p.name not in ('terminal.json','status.json','quality_frozen.json') and 'queue_delta' not in p.name and 'remote_delta' not in p.name:continue
            files[str(p.relative_to(ROOT))]=dict(identity=ev(p),value=load(p))
    processes=[]
    for line in output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']).splitlines():
        pid,uuid,mem=map(str.strip,line.split(','));p=Path('/proc')/pid
        if not p.exists():continue
        cg=(p/'cgroup').read_text();row=dict(pid=int(pid),uuid=uuid,memory_MiB=mem,cgroup=cg,project_model='embodied-research.slice' in cg,cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
        if row['project_model']:
            env=dict(s.split('=',1) for s in (p/'environ').read_bytes().decode().split('\0') if '=' in s);row.update(OMP=env.get('OMP_NUM_THREADS'),MKL=env.get('MKL_NUM_THREADS'),threads=next(s for s in (p/'status').read_text().splitlines() if s.startswith('Threads:')));pace=ROOT/'artifacts/thermal_guard/pacing'/f'{pid}.json';row['pacing']=load(pace) if pace.exists() else None
        processes.append(row)
    return dict(created_at=now(),files=files,processes=processes,thermal=load(ROOT/'artifacts/thermal_guard/latest.json'),GPU=output(['nvidia-smi','--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,power.limit','--format=csv,noheader,nounits']),units=output(['systemctl','--user','list-units','--type=service','--state=running','embodied-peer1*','--no-pager']),timer=output(['systemctl','--user','show','embodied-cvpr2027.timer','-p','TimersCalendar','-p','ActiveState']),event_timer=output(['systemctl','--user','show','embodied-cvpr2027-events.timer','-p','ActiveState','-p','UnitFileState']))

def capture():
    out=CYCLE/'exit_snapshot_v1';out.mkdir(exist_ok=False);atomic(out/'local.json',snapshot())
    code='from close_peer14_cycle21_v1_20261102 import snapshot;import json;print(json.dumps(snapshot()))';raw=output(['ssh','remote-gpu','cd '+str(REMOTE_ROOT)+' && PYTHONPATH=preexperiment/research_autopilot .conda/opencood-linux/bin/python -c '+shlex.quote(code)]);atomic(out/'remote.json',json.loads(raw))

def reconcile():
    out=CYCLE/'exit_snapshot_v1';snapshots=[load(out/(h+'.json')) for h in ('local','remote')];rows={r['id']:r for r in load(QUEUES[0])['tasks']};merged=[];skipped=[]
    for snap in snapshots:
        for p,ref in snap['files'].items():
            if 'queue_delta' not in Path(p).name and 'remote_delta' not in Path(p).name:continue
            for tid,v in ref['value'].items():
                if not isinstance(v,dict) or 'parent_task_id' not in v:continue
                current=rows.get(tid,{})
                if current.get('status') in TERMINAL and v.get('status') not in TERMINAL:skipped.append(tid);continue
                update(tid,v['parent_task_id'],{k:x for k,x in v.items() if k not in ('id','parent_task_id')});merged.append(tid)
    # The aggregate registration task only registers a contract; science stays open in two children.
    for dataset in ('SPD','V2'):record('registration_read_'+dataset,dict(status='waiting_registration_terminal',contract=ev(CYCLE/'registration_read_contract.json')))
    record('registration_read',dict(status='complete_light_checked',administrative_contract_only=True,scientific_children=['peer14_registration_read_'+d+'_v1_20261102' for d in ('SPD','V2')]))
    # Native-style terminals use a different status string: release only the exact completed owner.
    author=C15/'coalign_full_v1/author_evaluation/terminal.json'
    if author.exists() and load(author)['status']=='native_AP_complete_light_checked':update('peer13_coalign_pipeline_author_v3_20261102',task('K5'),dict(status='complete_light_checked',terminal=ev(author),scope='Author pretrained fusion + fresh SPD uncertainty; shared TraF GT/CoAlign prediction ROI; not original all-training reproduction.'))
    atomic(CYCLE/'queue_reconciliation_v1.json',dict(created_at=now(),merged=merged,skipped_stale=skipped,no_dispatch=True))

def table(name,rows):
    if not rows:return
    with (CYCLE/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def assets():
    controlled=CYCLE/'opv2v_controlled_read_v1/producer';rows=load(controlled/'panels.json');table('OPV2V_controlled_all144_panels.csv',rows)
    contrasts=[]
    for noise in ('SPD_structured','iid_02m_02deg'):
        for split in ('validate','test'):
            for group in ('all','ego_visible','remote_exclusive'):
                for th in (.5,.7):
                    ds={}
                    for protocol in ('P1_simulation','P2_inherited'):
                        rr={r['method']:r for r in rows if (r['noise'],r['split'],r['source_group'],r['threshold'],r['protocol'])==(noise,split,group,th,protocol)}
                        ds[protocol]=rr['q005_KF_estimator_only']['ap']-rr['stale']['ap']
                    contrasts.append(dict(noise=noise,split=split,source_group=group,threshold=th,P1_delta_pp=ds['P1_simulation'],P2_delta_pp=ds['P2_inherited'],sign_flip=ds['P1_simulation']>0 and ds['P2_inherited']<0,scope='q005 estimator-only, fixed400sample; not fullK1/U'))
    table('OPV2V_controlled_preregistered_and_source_contrasts.csv',contrasts)
    author=C15/'coalign_full_v1/author_evaluation';rr=[]
    for p in load(author/'panels.json'):
        for arm,m in p['metrics'].items():rr.append(dict(training='unadapted_author_fusion_DAIR',input=arm,role='val21',wait_ms=p['wait_ms'],AP50=100*m['global_score_sort']['ap50'],AP70=100*m['global_score_sort']['ap70'],population='shared_TraF_GT__CoAlign_prediction_ROI'))
    table('CoAlign_author_fusion_three_arms_val.csv',rr)
    synthesis=dict(created_at=now(),peer14_scope='A geometric proxy/B label-pose intervention/C bounded simulation; official main and inference unchanged',OPV2V_frames=400,OPV2V_GPU_forwards=801,OPV2V_GPU_wall_s=load(CYCLE/'opv2v_inheritance_v1/producer/terminal.json')['wall_s'],OPV2V_actual_staged_input_bytes=load(CYCLE/'opv2v_input_staging.json'),OPV2V_global_sign_flip=load(controlled/'preregistered_read.json'),OPV2V_remote_source_test=[r for r in contrasts if r['split']=='test' and r['source_group']=='remote_exclusive' and r['threshold']==.7],OPV2V_boundary='q005 translation estimator-only, rawscore fixedNMS; 25scenarios central16frames (validate9/test16), not fullK1 heading+U/fullOPV2V. Global preregistered flip FAILS, do not change noise to chase it. Structured remote-only flip descriptive, iid remote-only no flip. Independent all144 geometry/AP/NMS/counts review complete; all AP differences0, estimator/noise not independently revalidated.',A_recovery='Remote V2 raw files absent -> Open3D empty-target normal errors. Not physical failure. Original remoteSPD continues. Local V2 same ICP with original manifest/full paths,48fit pilot valid and quality frozen before nonfit. No scientific script/output overwritten.',A_B_followups=['A R1-R4 and quality-conditioned cached label AP finite consumers active','Native/CoAlign registration labels remain next-batch gap','B SPD cached fivefamilies all current/control methods and V2cached8/native4 running','B native batch18 producers all3seedTraF/e60 + CoAlign3author/adapted/online +8confirm producers running','B Where/ViT/other saved native and V2 historical dose scans next-batch gap','B crosses/peak/rank plots after all matching population curves ready','C exact fullK1 heading+U not covered by estimator-only; preserve as scope gap, not implied done'],no_new_microvariants=True,K1_failed_gates_closed=True,K2_K3_contracted=True,whole_release_closed=False,next_calendar='2026-11-03T00:00:00+08:00',no_model_wakeup=True)
    atomic(CYCLE/'scientific_synthesis.json',synthesis)
    old=load(C18/'release_coverage_audit.json');atomic(CYCLE/'release_coverage_audit.json',dict(created_at=now(),prior=ev(C18/'release_coverage_audit.json'),prior_release_still_open=old,peer14_remaining=synthesis['A_B_followups'],not_all_K5_released=True))

def audit():
    out=CYCLE/'exit_snapshot_v1';local,remote=(load(out/(h+'.json')) for h in ('local','remote'));aa,bb=map(load,QUEUES);ar,br=({r['id']:r for r in q['tasks']} for q in (aa,bb));ids=[i for i,r in ar.items() if i.startswith(('peer10','peer11','peer12','peer13','peer14')) and r.get('parent_task_id')]
    lm=[r for r in local['processes'] if r['project_model']];rm=[r for r in remote['processes'] if r['project_model']];contracts=list(CYCLE.glob('*_contract.json'));event=load(AUTO/'event_dispatch/state.json')
    receipts=[load(p) for p in (CYCLE/'launch_receipts').glob('*.json')];props=['-p','ActiveState','-p','Result','-p','CPUWeight','-p','ControlGroup','-p','Environment'];units={}
    for r in receipts:
        cmd=['systemctl','--user','show',r['unit'],*props]
        units[r['unit']]=output(['ssh','remote-gpu',shlex.join(cmd)] if r['remote'] else cmd)
    def healthy(s):return 'ActiveState=active' in s or ('ActiveState=inactive' in s and 'Result=success' in s)
    checks=dict(dual_queue_owners_equal=aa['autonomy_active_owners']==bb['autonomy_active_owners'],all_candidate_rows_equal=all(ar[i]==br[i] for i in ids),parent_child_links=all(i in ar[ar[i]['parent_task_id']].get('child_ids',[]) for i in ids),terminal_owners_released=all(ar[i]['status'] not in TERMINAL for i in aa['autonomy_active_owners']),standing_parent_open=ar['autonomous_research_continuation_20261024']['status']=='open_continuing',local_one_project_model_per_GPU=len(lm)==2 and len({r['uuid'] for r in lm})==2,remote_model_unique_or_peer14C_only=(len(rm)==1 or (len(rm)==2 and sum('opv2v_single_sensor' in r['cmdline'] for r in rm)==1)),project_models_controlled_threads2_pacing=all(r['OMP']==r['MKL']=='2' and r['pacing'] for r in lm+rm),thermal_both_monitoring=local['thermal']['status']==remote['thermal']['status']=='monitoring',remote450W=float(remote['GPU'].splitlines()[0].split(',')[-1])==450,calendar_only='OnCalendar=' in local['timer'] and 'ActiveState=active' in local['timer'],event_disabled='UnitFileState=disabled' in local['event_timer'] and 'ActiveState=inactive' in local['event_timer'],pending_notifications_empty=not event.get('pending'),new_contract_scripts_frozen=all(ev(ROOT/load(p)['script']['path'])==load(p)['script'] for p in contracts),all_new_units_healthy=all(healthy(s) for s in units.values()),new_unit_weights_match=all(('CPUWeight=10' if 'CPUWeight=10' in ' '.join(r['args']) else 'CPUWeight=25') in units[r['unit']] for r in receipts),C_GPU_under2h=load(CYCLE/'opv2v_inheritance_v1/producer/terminal.json')['wall_s']<7200,remote_new_guard2_1=(load(ROOT/'preexperiment/research_autopilot/peer14_4090_shared_disk_resource_config_v1_20261102.json')['resource']['disk_free_hard_gib']==1),mirror_once_success=load(CYCLE/'mirror_incremental_v1/terminal.json')['exit_code']==0)
    atomic(CYCLE/'execution_and_queue_audit.json',dict(created_at=now(),status='passed' if all(checks.values()) else 'needs_specific_review',checks=checks,local_models=lm,remote_models=rm,units=units,contracts=[ev(p) for p in contracts],scientific_claims_only_from_complete_terminals=True,no_dispatch=True));print(json.dumps(checks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['capture','reconcile','assets','audit']);globals()[p.parse_args().mode]()
