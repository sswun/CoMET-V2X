"""Peer11 finite experiment registration; calendar-only research continuation."""
import sys
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,atomic,ev,load,now,task
import peer10_cycle09_v1_20261101 as registry
CYCLE=AUDIT/'cycle15_peer11_v1'
JOBS={
 'seed3_public':('peer11_k4_seed3_public_native_v1_20261101',task('K4')),
 'seed3_online':('peer11_k4_seed3_online_native_v1_20261101',task('K4')),
 'e60_val':('peer11_k4_e60_exact_val_native_v1_20261101',task('K4')),
 'ego_probe':('peer11_k4_ego_only_native_probe_v1_20261101',task('K4')),
 'upper_split':('peer11_k5_native_slow_vs_jitter_upper_v1_20261101',task('K5')),
 'v2_public':('peer11_k4_v2_public_actual_train_eval_v1_20261101',task('K4')),
 'v2_online':('peer11_k4_v2_online_actual_train_eval_v1_20261101',task('K4')),
 'hold2_recovery':('peer11_k4_seed2_public_hold_correct_environment_v1_20261101',task('K4')),
 'source_successor':('peer11_k5_native_all_new_rows_source_aux_v1_20261101',task('K5')),
 'ffnet_runtime':('peer11_ffnet_cuda_runtime_localization_v4_20261101','autonomous_research_continuation_20261024'),
}
registry.JOBS.update(JOBS)
def update(name,values): registry.update(name,values)
def prepare():
 CYCLE.mkdir(exist_ok=False)
 refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
 for n in refs:(AUTO/n).read_text()
 atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
 c=load(AUDIT/'seed20261102_contract_v1.json');c['K4']['seed']=20261103;c['K4']['arms']=['PUBLIC','ONLINE_K1'];c['created_at']=now()
 c['peer11_registration']={'instructions':refs,'peer_review':ev(ROOT/'docs/review_20261031/peer11_device_and_k4_fair_baseline.md'),'fair_comparison':'paired seed ONLINE/ONLINE minus PUBLIC/PUBLIC; no e60 substitution; no OFFICIAL seed3','threads':{'OMP_NUM_THREADS':2,'MKL_NUM_THREADS':2,'torch_runner_threads':1,'DataLoader_workers':0},'cost_per_arm':{'updates':2000,'microbatches':8000,'val_cases':13264,'val_forwards_with_replay':39804,'estimated_wall_hours':'4-8, re-estimate measured throughput'},'prediction':'A third paired seed may confirm or reverse the modest same-budget native benefit; all rows retained, no val/hold checkpoint selection.'}
 atomic(CYCLE/'seed20261103_contract.json',c)
 for name in ('seed3_public','seed3_online'):
  update(name,{'status':'registered_ready','contract':ev(CYCLE/'seed20261103_contract.json'),'prediction_before_result':c['peer11_registration']['prediction'],'estimated_cost':c['peer11_registration']['cost_per_arm'],'input_identity':c['inputs'],'initial_weight':c['K4']['initial_weight'],'threads':c['peer11_registration']['threads']})
if __name__=='__main__':prepare()
