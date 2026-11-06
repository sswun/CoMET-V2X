"""Additive midnight successors. Frozen inputs, finite lanes, calendar-only."""
import copy
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,atomic,ev,load,now,task
import peer10_cycle09_v1_20261101 as registry
CYCLE=AUDIT/'cycle00_peer12_v1_20261102'
JOBS={
 'mixed00':('peer12_k4_mixed_pose_ablation_v1_20261102',task('K4')),
 'lag00':('peer12_k5_native_lag1_reference_v1_20261102',task('K5')),
 'v2review00':('peer12_k4_v2_paired_independent_AP_v1_20261102',task('K4')),
 'sources00':('peer12_k5_missing_hold_mixed_lag_sources_v1_20261102',task('K5')),
}
registry.JOBS.update(JOBS)
def update(key,values):
 registry.JOBS.update(JOBS);registry.update(key,values)

def prepare():
 # Mandatory re-read immediately before new successors, not an old snapshot.
 refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
 for n in refs:(AUTO/n).read_text()
 CYCLE.mkdir(exist_ok=False)
 atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
 train=copy.deepcopy(load(AUDIT/'contract_v1.json'))
 train['K4'].update(arms=['MIXED_PUBLIC_ONLINE'],seed=20261101)
 train['created_at']=now()
 train['midnight_ablation']={'sampling':'same fit24 condition permutation and microbatch RNG as seed1 controls; independent Bernoulli(0.5) input choice, fixed seed xor 0x50454552; no labels/future used in input choice','selection':'ablation only; cannot replace native recommendation without a preregistered fit/cal criterion; no val/hold selection'}
 atomic(CYCLE/'mixed_training_contract.json',train)
 files=['peer12_cycle00_v1_20261102.py','run_peer12_cycle00_v1_20261102.py','run_peer12_native_additions_v1_20261102.py','run_peer12_mixed_pose_v1_20261102.py','review_peer12_v2_paired_v1_20261102.py','run_peer12_sources00_v1_20261102.py']
 c={'created_at':now(),'instructions':refs,'peer12':ev(ROOT/'docs/review_20261031/peer12_ceiling_split_and_preregistered_reads.md'),'official_protocol_unchanged':True,'model_wakeup':'3h calendar only, no events/watchdog/agent',
 'training_source':ev(ROOT/'preexperiment/research_autopilot/run_peer10_k4_native_finetune_v1_20261101.py'),
 'evaluator_source':ev(ROOT/'preexperiment/research_autopilot/run_peer10_native_evaluate_v1_20261101.py'),
 'profile_source':ev(ROOT/'preexperiment/research_autopilot/run_peer12_native_profile_v1_20261101.py'),
 'hold_adapter':ev(ROOT/'preexperiment/research_autopilot/run_peer10_k4_hold14_native_v1_20261101.py'),
 'source_adapter':ev(ROOT/'preexperiment/research_autopilot/run_peer12_native_source_additions_v1_20261101.py'),
 'source_contract':ev(AUDIT/'cycle15_peer11_v1/contract.json'),
 'review_geometry':ev(ROOT/'preexperiment/research_autopilot/review_peer10_cached_hwl_independent_v1_20261101.py'),
 'mixed_contract':ev(CYCLE/'mixed_training_contract.json'),
 'lag_weights':load(AUDIT/'cycle15_peer11_v1/contract.json')['native_probe']['weights'],
 'v2_producers':{n:ev(AUDIT/('cycle15_peer11_v1/v2_'+n)/'terminal.json') for n in ('public','online')},
 'v2_config':ev(ROOT/'artifacts/checkpoints/traf_align_v2v4real_0f6f98de/config.yaml'),
 'predictions':{'mixed00':'Fixed 50/50 pose training may distinguish input distribution matching from lower training noise and reduce PUBLIC fallback loss; keep all signs, ablation only.', 'lag00':'Previous-frame annotation offsets separate temporal persistence from per-frame annotation coupling; diagnostic, not deployable or selected.', 'v2review00':'Independent HWL polygons, IoU matching and VOC AP can confirm or expose the completed V2 same-budget cross-domain result; no forwards.', 'sources00':'Complete missing actual seed2/seed3 hold and new mixed/lag source/auxiliary panels using unique producer shards; no inference or method selection.'},
 'costs':{'mixed00':{'updates':2000,'microbatches':8000,'val_conditions':13264,'hold_conditions':9744,'eval_inputs':2,'eval_forwards_including_replay':46032,'profile_forwards_each_role':136,'estimated_wall_h':'4-8 under thermal quota','new_disk_GiB':'<1.5; final/active rolling checkpoint + unique compressed shards'},'lag00':{'models':2,'conditions_each':13264,'forwards_each':13268,'profile_forwards_each':68,'estimated_wall_h':'2-6, after FFNet','new_disk_GiB':'<0.5 unique compressed diagnostic predictions'},'v2review00':{'GPU_forwards':0,'conditions_per_model':2644,'metric_panels':32,'CPU_threads':1,'estimated_wall_h':'0.1-1','new_disk_GiB':'<0.01 checks only'},'sources00':{'GPU_forwards':0,'CPU_threads':1,'estimated_wall_h':'6-18 plus finite producer dependencies','new_disk_GiB':'<0.8 compressed score/outcome deltas; no geometry/GT copies'}},
 'lanes':{'gpu0':{'after':['embodied-peer12-cycle21-gpu0-v1-20261101.service'],'jobs':['mixed00']},'gpu1':{'after':['embodied-peer12-cycle21-gpu1-v1-20261101.service','embodied-peer12-ffnet-chain-v5-20261101.service'],'jobs':['lag00']},'cpu':{'after':[],'jobs':['v2review00','sources00']}},
 'coverage_until':'2026-11-02T03:00:00+08:00 and beyond: GPU0 hold→mixed train/val/hold; GPU1 hold→FFNet→lag; 4090 existing ego→wait0-first upper→remaining; CPU independent→sources, old cached→new val sources unchanged',
 'resources':'controlled slice + Wants/After thermal + pressure_guard; GPU OMP/MKL/OpenBLAS2 and workers0; CPUWeight25; GPU80/CPU70 continuous pacing; remote450W unchanged',
 'storage':'reference shared data/weights/ledger; only unique final checkpoint and compressed new predictions; retain negatives; active resume retained; no manual duplicate exit du',
 'scripts':{n:ev(ROOT/'preexperiment/research_autopilot'/n) for n in files}}
 atomic(CYCLE/'contract.json',c)
 for key in JOBS:update(key,{'status':'registered_finite_successor','contract':ev(CYCLE/'contract.json'),'prediction_before_result':c['predictions'][key],'estimated_cost':c['costs'][key],'unique_output':str((CYCLE/key).relative_to(ROOT))})
if __name__=='__main__':prepare()
