"""09h finite successors; immutable contracts and paired queue bookkeeping."""
import fcntl
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,QUEUES,atomic,ev,load,now,task

CYCLE=AUDIT/'cycle09_v1'
JOBS={'aux_remaining':('peer10_fixed_population_remaining_aux_v2_20261101',task('K5')),
      'ffnet_resume':('peer10_ffnet_cuda_diagnostic_resume_v2_20261101','autonomous_research_continuation_20261024'),
      'k4_hold14':('peer10_k4_current_recipe_hold14_native_v1_20261101',task('K4'))}

def update(name,values):
    tid,pid=JOBS[name];stamp=now()
    with (AUTO/'.queue_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        canonical=next((r for r in load(QUEUES[0])['tasks'] if r['id']==tid),None)
        for path in QUEUES:
            q=load(path);parent=next(r for r in q['tasks'] if r['id']==pid)
            rows=[r for r in q['tasks'] if r['id']==tid];assert len(rows)<=1
            row=rows[0] if rows else {'id':tid,'parent_task_id':pid,'created_at':stamp}
            if not rows:q['tasks'].append(row)
            row['created_at']=canonical['created_at'] if canonical else stamp
            row.update(values,updated_at=stamp)
            finished=values.get('status') in ('complete_light_checked','complete_independently_reviewed','failed_retained')
            row['owner']=None if finished else 'research_autopilot_'+tid
            if finished:q.setdefault('autonomy_active_owners',{}).pop(tid,None)
            else:q.setdefault('autonomy_active_owners',{})[tid]=row['owner']
            if tid not in parent.setdefault('child_ids',[]):parent['child_ids'].append(tid)
            if finished and tid not in parent.setdefault('completed_child_ids',[]):parent['completed_child_ids'].append(tid)
            atomic(path,q)
        a,b=map(load,QUEUES)
        assert a['autonomy_active_owners']==b['autonomy_active_owners']
        assert next(r for r in a['tasks'] if r['id']==tid)==next(r for r in b['tasks'] if r['id']==tid)

def prepare():
    CYCLE.mkdir(exist_ok=False)
    instructions={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in instructions:(AUTO/n).read_text()
    atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
    c={'created_at':now(),'latest_instructions':instructions,'official_main_protocol_unchanged':True,
       'model_wakeup':'3h calendar only; no event/dispatcher/model invocation',
       'original_aux_contract':ev(AUDIT/'cycle06_v1/contract.json'),
       'completed_aux_panels':ev(AUDIT/'cycle06_v1/registration_recovery_v2/fixed_population_aux/panels.json'),
       'completed_aux_terminal':ev(AUDIT/'cycle06_v1/registration_recovery_v2/fixed_population_aux/terminal.json'),
       'shared_main_contract':ev(AUDIT/'contract_v1.json'),
       'aux_remaining':{'prediction':'Missing module/ego/official-late label sensitivity may differ; retain every result, do not select methods on auxiliary AP.',
          'fix':'derive missing gt_offsets from selected metadata gt_start/gt_stop; verify lengths, official TP on first/last frames and full scoring; old scientific script and completed 2040 panels unchanged',
          'families':['modules','ego','official_late'],'estimated_cost':{'CPU_threads':1,'wall_hours':'3-8','new_disk_mib':200,'GPU_forwards':0,'updates':0}},
       'storage':'shared inputs, labels, predictions and completed prefixes referenced by SHA; save compressed outcomes/metrics only; no large duplicates; failed outputs retained pending consumers',
       'scripts':{n:ev(ROOT/'preexperiment/research_autopilot'/n) for n in ('peer10_cycle09_v1_20261101.py','run_peer10_aux_remaining_v2_20261101.py')}}
    atomic(CYCLE/'contract.json',c)
    update('aux_remaining',{'status':'registered_ready','contract':ev(CYCLE/'contract.json'),'prediction_before_result':c['aux_remaining']['prediction'],'estimated_cost':c['aux_remaining']['estimated_cost']})

if __name__=='__main__':prepare()
