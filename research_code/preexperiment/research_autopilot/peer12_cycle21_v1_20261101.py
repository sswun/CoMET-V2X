"""21h additive successors: one bounded causal hypothesis and missing holds."""
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,atomic,ev,load,now,task
import peer10_cycle09_v1_20261101 as registry
CYCLE=AUDIT/'cycle21_peer12_v1'
JOBS={
 'robust_history21':('peer12_k1_causal_geomedian_history_v1_20261101',task('K1')),
 'native_review21':('peer12_k4_seed1_full_independent_native_v1_20261101',task('K4')),
 'hold2_online21':('peer12_k4_seed2_online_hold14_v1_20261101',task('K4')),
 'hold3_public21':('peer12_k4_seed3_public_hold14_v1_20261101',task('K4')),
 'hold3_online21':('peer12_k4_seed3_online_hold14_v1_20261101',task('K4')),
}
registry.JOBS.update(JOBS)
def update(key,values):registry.update(key,values)
def prepare():
 CYCLE.mkdir(exist_ok=False)
 refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
 for n in refs:(AUTO/n).read_text()
 atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
 old=AUTO/'status_audit_20261029_peer_revision'
 c={'created_at':now(),'instructions':refs,'peer12':ev(ROOT/'docs/review_20261031/peer12_ceiling_split_and_preregistered_reads.md'),
 'official_protocol_unchanged':True,'shared_contract':ev(AUDIT/'contract_v1.json'),
 'trigger':ev(AUDIT/'cycle18_peer12_v1/ceiling_recovery_v2/preregistered_reads.json'),
 'old_answered_controls':{'terminal':ev(old/'correction_readout_z2_mean_q_controls_terminal_v1.json'),'implementation':ev(ROOT/'preexperiment/research_autopilot/run_crosscheckpoint_current_temporal_fixed_readout_controls_v1_20261029.py'),'distinction':'Old cumulative_anchor_mean is count-weighted pre-heading CV residual without ego transport; old q0/q005/q015/q030 are reused evidence, NOT rerun. New hypothesis is uniform-frame, heading-aligned robust spatial median in initial public ego yaw frame. Only ONE new variant, no Q grid.'},
 'robust_history':{'method':'causal_history_geomedian','frame_observation':'spatial median of K1 protected heading registration residuals; duplicate whole pair list has no effect','history':'equal weight per observed decision frame, per sequence/wait; transform displacement into first public ego yaw frame; spatial median of causal frame summaries; missing observation carries reference state; first unobserved output zero','solver':'deterministic smoothed Weiszfeld, max128 iterations, epsilon1e-8 m, stopping1e-10 m; numerical constants fixed, not tuned','selection':'No hyperparameter grid. fit24 then calibration8 only, globally freeze candidate and K1 fallback by calibration official mean AP70; preserve raw new hypothesis as a diagnostic even if not selected. val/hold never select.','native_promotion':'Only the cal-selected raw new hypothesis, if both complete val21/hold14 >=+0.5 AP70 vs exact current K1; log eligibility for next calendar, no automatic model wakeup.','prediction':'Robust equal-frame long history may recover slow shared residuals, or may underfit time-varying bias; both signs retained. This is a peer12 bounded followup, not evidence MA5 is physically observable.','cost':{'GPU_forwards':0,'updates':0,'CPU_threads':1,'estimated_wall_h':'0.3-3 under quota','new_disk_GiB':'<0.3; shared inputs/GT/U referenced'}},
 'native_review':{'producers':['k4_eval_public','k4_eval_online_k1'],'source_root':'cycle03_v1/native_source_aux','protocols':['official','interpolated','no_interpolation'],'rois':['native','common'],'groups':['all','veh','coop','inf','veh_coop'],'thresholds':[.5,.7],'prediction':'Independent numpy/shapely geometry, primary label source/HWL checks, target-first matching and VOC AP may confirm or expose existing scorer errors; discrepancies are retained, not repaired by selecting methods.','cost':{'GPU_forwards':0,'updates':0,'CPU_threads':1,'estimated_wall_h':'4-14','new_disk_GiB':'<0.05 metrics only; reference all predictions and labels'}},
 'holds':{'hold2_online21':{'weight':ev(AUDIT/'seed2_online_k1/training/final.pth'),'gpu':1},'hold3_public21':{'weight':ev(AUDIT/'cycle15_peer11_v1/seed3_public/training/final.pth'),'gpu':1},'hold3_online21':{'weight':ev(AUDIT/'cycle15_peer11_v1/seed3_online/training/final.pth'),'gpu':0}},
 'hold_prediction':'Complete missing same-budget paired-seed mechanism holdouts without substituting old branches or selecting checkpoints; hold14 is within official train, not new test data.',
 'hold_cost_each':{'conditions':9744,'forwards':29244,'updates':0,'profile_conditions':64,'profile_forwards':204,'optional_cached_profile_forwards':204,'estimated_wall_h':'1.5-4 under actual quota'},
 'native_evaluator':ev(ROOT/'preexperiment/research_autopilot/run_peer10_native_evaluate_v1_20261101.py'),
 'lanes':{'gpu0':{'after':'embodied-peer11-cycle15-gpu0-v1-20261101.service','jobs':['hold3_online21']},'gpu1':{'after':'embodied-peer11-cycle15-gpu1-v1-20261101.service','jobs':['hold2_online21','hold3_public21']},'cpu':{'after':None,'jobs':['robust_history21','native_review21']}},
 'resources':'Wants/After thermal, pressure_guard, OMP/MKL2 GPU workers0; CPUWeight25; no powers changed; remote existing OFFICIAL/profile/ego/upper/source chains unchanged',
 'coverage_until':'2026-11-02T00:00:00+08:00 and beyond; gpu0 existing V2ONLINE then seed3hold, gpu1 FFNet then two full holds, remote existing upper long chain, CPU robust then full native independent review',
 'storage':'No pointcloud/GT/feature copies. Existing final weights referenced. Compressed candidate IDs/score/TP/deltas necessary only. No new large deletions.',
 'scripts':{n:ev(ROOT/'preexperiment/research_autopilot'/n) for n in ('peer12_cycle21_v1_20261101.py','run_peer12_causal_history_v1_20261101.py','run_peer12_cycle21_lane_v1_20261101.py','run_peer12_hold_profiled_v1_20261101.py','review_peer12_native_independent_v1_20261101.py')}}
 atomic(CYCLE/'contract.json',c)
 update('robust_history21',{'status':'registered_ready','contract':ev(CYCLE/'contract.json'),'prediction_before_result':c['robust_history']['prediction'],'estimated_cost':c['robust_history']['cost']})
 update('native_review21',{'status':'registered_ready','contract':ev(CYCLE/'contract.json'),'prediction_before_result':c['native_review']['prediction'],'estimated_cost':c['native_review']['cost']})
 for k,v in c['holds'].items():update(k,{'status':'registered_finite_successor','contract':ev(CYCLE/'contract.json'),'weight_identity':v['weight'],'prediction_before_result':c['hold_prediction'],'estimated_cost':c['hold_cost_each'],'gpu':v['gpu']})
if __name__=='__main__':prepare()
