"""Reconcile actual outputs, exact owners, costs and cross-calendar coverage."""
import json,subprocess,os
from pathlib import Path
from peer12_cycle03_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C00,atomic,ev,load,now,task
from peer10_tournament_v1_20261101 import QUEUES,queues
import peer10_cycle09_v1_20261101 as registry

def local_resource():
 def output(cmd):return subprocess.check_output(cmd,text=True).strip()
 compute=output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']);processes=[]
 for line in compute.splitlines():
  pid,uuid,mem=map(str.strip,line.split(','));p=Path('/proc')/pid
  if not p.exists():continue
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();processes.append(dict(pid=int(pid),uuid=uuid,memory_MiB=mem,cmdline=cmd,cgroup=(p/'cgroup').read_text(),threads=next(x for x in (p/'status').read_text().splitlines() if x.startswith('Threads:')),project_model=str(ROOT) in cmd or 'preexperiment/research_autopilot/run_peer12_cycle03' in cmd))
 units=['embodied-peer12-cycle03-gpu0-v2-20261102','embodied-peer12-cycle03-gpu1-v1-20261102','embodied-peer12-factor-gap-gpu0-v1-20261102','embodied-peer12-mixed-factor-gpu1-v1-20261102','embodied-peer12-hold-three-seed-independent-v1-20261102','embodied-peer12-cycle03-source-all-v1-20261102','embodied-peer12-factor-source-v1-20261102','embodied-peer12-mixed-factor-source-v1-20261102']
 return dict(created_at=now(),processes=processes,thermal=load(ROOT/'artifacts/thermal_guard/latest.json'),gpu=output(['nvidia-smi','--query-gpu=index,uuid,power.limit,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits']),units={u:output(['systemctl','--user','show',u+'.service','-p','ActiveState','-p','ControlGroup','-p','Wants','-p','CPUWeight']) for u in units},timer=output(['systemctl','--user','show','embodied-cvpr2027.timer','-p','TimersCalendar','-p','ActiveState']),event_timer=output(['systemctl','--user','show','embodied-cvpr2027-events.timer','-p','ActiveState','-p','UnitFileState']))

def remote_resource():
 script=r'''
import json,subprocess
from pathlib import Path
from peer12_cycle03_v1_20261102 import ROOT,AUDIT,CYCLE,C00,load,ev,now
def output(cmd):return subprocess.check_output(cmd,text=True).strip()
files={}
for base in [CYCLE,C00/'remote_hold_upper_v2',C00/'remote_hold_upper_source_v2']:
 for p in sorted(base.rglob('*.json')):
  if p.name not in ('terminal.json','status.json') and 'delta' not in p.name:continue
  files[str(p.relative_to(ROOT))]={'identity':ev(p),'value':load(p)}
compute=output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']);processes=[]
for line in compute.splitlines():
 pid,uuid,mem=map(str.strip,line.split(','));p=Path('/proc')/pid
 if not p.exists():continue
 cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();processes.append(dict(pid=int(pid),uuid=uuid,memory_MiB=mem,cmdline=cmd,cgroup=(p/'cgroup').read_text(),project_model=('python' in cmd and (str(ROOT) in cmd or 'preexperiment/research_autopilot/run_peer12_cycle03' in cmd))))
units=['embodied-peer12-cycle03-remote-v2-20261102','embodied-peer12-factor-gap-remote-v1-20261102','embodied-peer12-factor-source-v1-20261102','embodied-peer12-cumulative-source-v1-20261102']
print(json.dumps(dict(created_at=now(),files=files,processes=processes,thermal=load(ROOT/'artifacts/thermal_guard/latest.json'),gpu=output(['nvidia-smi','--query-gpu=index,uuid,power.limit,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits']),units={u:output(['systemctl','--user','show',u+'.service','-p','ActiveState','-p','ControlGroup','-p','Wants','-p','CPUWeight']) for u in units},no_predictions_copied=True)))
'''
 r=subprocess.run(['ssh','remote-gpu','cd /path/to/remote_project && PYTHONPATH=preexperiment/research_autopilot .conda/opencood-linux/bin/python -'],input=script,text=True,capture_output=True,check=True);return json.loads(r.stdout)

def bind_update(tid,pid,v):
 k='close03_'+tid;registry.JOBS[k]=(tid,pid);registry.update(k,v)

def main():
 from peer12_cycle00_v1_20261102 import JOBS as midnight
 from peer12_cycle21_v1_20261101 import JOBS as previous
 from peer11_cycle15_v1_20261101 import JOBS as seed3
 for mapping in (midnight,previous,seed3):registry.JOBS.update(mapping)
 completed={}
 for key,name in {'mixed00':'cycle00_peer12_v1_20261102/mixed00','hold3_online21':'cycle21_peer12_v1/hold3_online21','hold3_public21':'cycle21_peer12_v1/hold3_public21','source_successor':'cycle15_peer11_v1/source_successors'}.items():
  t=AUDIT/name/'terminal.json'
  if t.exists() and load(t)['status'] in ('complete_light_checked','complete_independently_reviewed') and key in registry.JOBS:
   registry.update(key,dict(status=load(t)['status'],terminal=ev(t),scientific_synthesis=ev(CYCLE/'scientific_synthesis.json')));completed[key]=ev(t)
 # Old full native replay is preserved failed; new composed independent review is separately complete.
 registry.JOBS.update(previous)
 registry.update('native_review21',dict(status='failed_retained',terminal=ev(AUDIT/'cycle21_peer12_v1/independent_native/terminal.json'),resolved_by=ev(CYCLE/'native_tie_review03/terminal.json'),scope='Original full flags/counts/labels passed, filtered tie-population AP comparison failed; no original result overwritten.'))
 bind_update('peer10_cached_hwl_full_independent_review_v1_20261101',task('K5'),dict(status='complete_independently_reviewed',terminal=ev(AUDIT/'cycle12_v1/independent_cached_hwl/terminal.json')))
 bind_update('peer10_fixed_population_hwl_independent_v1_20261101',task('K5'),dict(status='failed_retained',administrative_only_no_experiment=True,duplicate_registration_of='peer10_cached_hwl_full_independent_review_v1_20261101',GPU_forwards=0,cost=0,reason='Initial close03 bound a descriptive alias instead of the existing canonical task ID; canonical owner/result corrected, alias retained transparently.'))
 bind_update('peer12_ffnet_extension_bound_runtime_v5_20261101','autonomous_research_continuation_20261024',dict(status='failed_retained',terminal=ev(AUDIT/'cycle21_peer12_v1/ffnet_v5_resource/resource_summary.json'),committed_conditions=3584,recovery='peer12_ffnet_allocator03_v1_20261102',cause='23GiB GPU memory hard guard, not temperature, original runtime/error-slot qualification remains valid'))
 remote=remote_resource();local=local_resource();atomic(CYCLE/'remote_execution_snapshot.json',remote);atomic(CYCLE/'local_execution_snapshot.json',local)
 for path,row in remote['files'].items():
  if 'delta' in Path(path).name:
   for tid,v in row['value'].items():
    if isinstance(v,dict) and 'parent_task_id' in v:bind_update(tid,v['parent_task_id'],{k:x for k,x in v.items() if k not in ('id','parent_task_id')})
 # Terminal jobs never retain active owners, even administrative zero-forward failures.
 for key in ('lag_val_hold03','sources03'):
  t=CYCLE/key/'terminal.json'
  if t.exists():bind_update('peer12_'+key+'_v1_20261102',task('K5'),dict(status=load(t)['status'],terminal=ev(t),zero_GPU_namespace_failure=True,superseded_by='peer12_'+key+'_v2_v1_20261102'))
 for name,pid in [('remote_hold_upper_v2',task('K5')),('remote_hold_upper_source_v2',task('K5'))]:
  target=str((C00/name/'terminal.json').relative_to(ROOT));row=remote['files'].get(target)
  if row:
   tid='peer12_k5_native_hold14_slow_upper_recovery_v2_20261102' if name=='remote_hold_upper_v2' else 'peer12_k5_native_hold14_upper_source_v2_20261102';bind_update(tid,pid,dict(status=row['value']['status'],remote_terminal=row['identity'],actual_remote_snapshot=ev(CYCLE/'remote_execution_snapshot.json'),superseded_by='peer12_hold_upper_recovery03_v1_20261102'))
 for tid,pid,unit,script in [('peer12_lag_val_hold03_v2_v1_20261102',task('K5'),'embodied-peer12-cycle03-gpu0-v2-20261102.service','run_peer12_cycle03_recovery_v2_20261102.py'),('peer12_sources03_v2_v1_20261102',task('K5'),'embodied-peer12-cycle03-source-all-v1-20261102.service','run_peer12_sources03_all_v1_20261102.py')]:bind_update(tid,pid,dict(status='running_finite_worker' if 'lag_' in tid else 'waiting_registered_finite_predictions',unit=unit,actual_script=ev(ROOT/'preexperiment/research_autopilot'/script)))
 bind_update('peer12_remote_upper_cumulative03_v2_v1_20261102',task('K5'),dict(status='running_finite_worker',unit='embodied-peer12-cycle03-remote-v2-20261102.service',actual_script=ev(ROOT/'preexperiment/research_autopilot/run_peer12_cycle03_recovery_v2_20261102.py'),partial_upper_zero_forward_failures_retained=True,hold_materials_now_verified=True))
 for tid,pid,unit in [('peer12_official_train_hold03_v1_20261102',task('K4'),'embodied-peer12-factor-gap-gpu0-v1-20261102.service'),('peer12_hold_upper_recovery03_v1_20261102',task('K5'),'embodied-peer12-factor-gap-remote-v1-20261102.service'),('peer12_k4_mixed_official_factor_completion_v1_20261102',task('K4'),'embodied-peer12-mixed-factor-gpu1-v1-20261102.service')]:bind_update(tid,pid,dict(status='waiting_known_predecessor_finite_worker',unit=unit))
 bind_update('peer12_k4_hold_three_seed_full_independent_v1_20261102',task('K4'),dict(status='running_full_independent_native_replay',unit='embodied-peer12-hold-three-seed-independent-v1-20261102.service'))
 bind_update('peer12_k5_remote_cumulative_source_v1_20261102',task('K5'),dict(status='waiting_registered_finite_predictions',unit='embodied-peer12-cumulative-source-v1-20261102.service'))
 # Sole producers and CPU consumers; waiting successor is not a second GPU model.
 projects=local['processes'];rprojects=[r for r in remote['processes'] if r['project_model']]
 uuids=[line.split(',')[1].strip() for line in local['gpu'].splitlines()]
 checks={'local_one_actual_model_per_GPU':all(sum(r['uuid']==u for r in projects)==1 for u in uuids),'remote_one_actual_model_excluding_desktop':len(rprojects)==1,'all_project_models_controlled':all('embodied-research.slice' in r['cgroup'] for r in projects+rprojects),'thermal_both_monitoring':local['thermal']['status']==remote['thermal']['status']=='monitoring','remote450W':float(remote['gpu'].split(',')[2].strip())==450,'event_timer_disabled':'UnitFileState=disabled' in local['event_timer'],'calendar_only':'OnCalendar=' in local['timer'],'all_scheduled_local_units_active':all('ActiveState=active' in v for v in local['units'].values()),'all_remote_units_active':all('ActiveState=active' in v for v in remote['units'].values())}
 r=load(CYCLE/'scientific_synthesis.json');queues({'K4':dict(status='running_remaining_factor_and_hold_independent_review',latest_scientific_synthesis=ev(CYCLE/'scientific_synthesis.json'),hold_three_seed=r['hold_three_seed'],native_seed1_saved_AP_full_independent=ev(CYCLE/'native_tie_review03/terminal.json'),mixed_ablation_only_no_recipe_change=True,V2_negative_transfer_retained=True),'K5':dict(status='running_lag_cumulative_hold_upper_sources',latest_scientific_synthesis=ev(CYCLE/'scientific_synthesis.json'),native_val_all4_slow_reference_read=r['native_upper_reads']['all4waits'],cached_all3600_full_independent=ev(AUDIT/'cycle12_v1/independent_cached_hwl/terminal.json'),annotation_reference_not_physical_truth=True)})
 a,b=map(load,QUEUES);checks['dual_queue_owners_equal']=a['autonomy_active_owners']==b['autonomy_active_owners'];ids=[x['id'] for x in a['tasks'] if '20261102' in x['id'] and ('peer12' in x['id'])]
 checks['new_tasks_identical_both_queues']=all(next(x for x in a['tasks'] if x['id']==tid)==next(x for x in b['tasks'] if x['id']==tid) for tid in ids)
 checks['new_tasks_parent_links']=all(x['id'] in next(y for y in a['tasks'] if y['id']==x['parent_task_id']).get('child_ids',[]) for x in a['tasks'] if x['id'] in ids)
 checks['standing_parent_open']=next(x for x in a['tasks'] if x['id']=='autonomous_research_continuation_20261024')['status']=='open_continuing'
 contracts=['contract.json','recovery_v2_contract.json','factor_gaps_contract.json','mixed_official_factor_contract.json','hold_independent_contract.json'];checks['frozen_worker_SHA_intact']=all(ev(ROOT/load(CYCLE/c)['script']['path'])==load(CYCLE/c)['script'] for c in contracts)
 atomic(CYCLE/'execution_and_queue_audit.json',dict(status='passed' if all(checks.values()) else 'needs_specific_review',created_at=now(),checks=checks,contracts=[ev(CYCLE/c) for c in contracts],actual_completed=completed,local_snapshot=ev(CYCLE/'local_execution_snapshot.json'),remote_snapshot=ev(CYCLE/'remote_execution_snapshot.json'),new_task_ids=ids,no_model_wakeup=True))
 gaps=['hold14 full independent4320 panel review running; secondary-seed val independent still not claimed','lag/cumulative full3model val/hold plus source/aux','hold slow upper recovery and source, all3models; initial zero-forward failures preserved','OFFICIAL-trained hold3input and mixed50 official input factor/source completion','FFNet author checkpoint remaining39460 conditions and independent replay;11-03stop retained','all-external source/aux coverage including successful FFNet','complete CoAlign full system remains unmet, pose-graph module cannot replace it','Where native same-budget training/backbone only after actual loader/train qualification','11-06 objective effect/novelty comparison; no final core narrative selected']
 atomic(CYCLE/'release_coverage_audit.json',dict(created_at=now(),original_release_closed=False,standing_parent='open_continuing',remaining=gaps,not_a_promise_of_complete_full_system_coverage=True))
 atomic(CYCLE/'finite_device_coverage.json',dict(created_at=now(),target='2026-11-02T06:00:00+08:00 and beyond',GPU0='lag three models val+hold69048 forwards → first-seed OFFICIAL-trained hold three-input29244; finite, missing factors only',GPU1='FFNet remaining39460 author conditions + CPU independent review; on GPU producer release automatically mixed OFFICIAL_REFERENCE val+hold23016 forwards, parallel CPU review',remote4090='cumulative three models val+hold69048 → hold seqmean/MA5 three models58488; missing frozen materials and12388 raw/calibration files verified available',CPU='58464-condition independent4320 panels plus local/remote sources;CPUWeight25',no_padding_repeats=True,no_event_model_wakeup=True))
 outer=Path(load(AUTO/'latest.json')['cycle_dir']);atomic(CYCLE/'storage_exit_delegation.json',dict(created_at=now(),status='existing_outer_run_cycle_single_exit_measurement',expected=str(outer/'storage_end.json'),entry=ev(CYCLE/'storage_entry.json'),no_manual_duplicate_du=True,no_material_deleted_this_cycle=True,completed_mixed_rolling_retention_review='within48h after current final-weight consumers are bound; retain final/formal predictions/active recovery'))
 print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
