"""Peer12 additive registration; old frozen experiments are not rewritten."""
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,atomic,ev,load,now,task
import peer10_cycle09_v1_20261101 as registry
CYCLE=AUDIT/'cycle18_peer12_v1'
JOBS={'ceiling':('peer12_k5_cached_observable_ceiling_v1_20261101',task('K5')),
      'profile':('peer12_native_64case_profile_v1_20261101',task('K4')),
      'upper':('peer12_k5_native_upper_wait0_first_v1_20261101',task('K5'))}
registry.JOBS.update(JOBS)
def update(key,values):registry.update(key,values)

def prepare():
 CYCLE.mkdir(exist_ok=False)
 refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
 for n in refs:(AUTO/n).read_text()
 atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
 old=load(AUDIT/'cycle15_peer11_v1/contract.json')
 c={'created_at':now(),'instructions':refs,'peer12':ev(ROOT/'docs/review_20261031/peer12_ceiling_split_and_preregistered_reads.md'),
 'shared_contract':ev(AUDIT/'contract_v1.json'),'native_contract':ev(AUDIT/'cycle15_peer11_v1/contract.json'),
 'weights':old['native_probe']['weights'],'official_protocol_unchanged':True,'model_wakeup':'3h calendar only; no agent or event',
 'offset_definition':'Import original peer11 upper_inputs for sequence mean and centred MA5 exactly; deduplicate decision times across waits; lag1 first frame zero; cummean includes current annotation. ALL annotation-assisted, NOT deployable, not method selection.',
 'read_rules':{'G_slow':'max(seqmean,MA5) minus K1, official all-GT AP70 four-wait means, both roles','small':'both <=0.5: no estimator branch','large':'val>=1.0 and hold>=1.0: at most two fit/cal-only causal variants; no model decision until next calendar','middle':'report only','ego':'veh_coop drop only; AP70 difference>=0.5 or same-sign AP50>=1.0; otherwise only inf localization hypothesis'},
 'storage':'Shared cached material/GT/head/labels referenced. Save only frame deltas, candidate IDs/scores/TP, compressed source outcomes, metrics. No pointcloud/GT/weight copies. Estimated <1GiB CPU, <6GiB native; discard no active snapshots.',
 'cost':{'ceiling':{'GPU_forwards':0,'updates':0,'CPU_threads':1,'wall_hours':'1-4 measured under shared thermal quota','roles':['val21','mechanism_holdout14'],'waits':[0,100,200,400],'source_groups':['all','veh','coop','inf','veh_coop']},'profile':{'cases':64,'three_inputs':True,'forwards':192,'GPU':'4090 after existing OFFICIAL seed2','purpose':'read/parse, projection+voxelization, forward, AP accounting; no scientific AP claim'},'upper':{'models':2,'inputs':2,'waits':[0,100,200,400],'forwards_with_replays':53072,'order':'both models wait0 first, then remaining waits','annotation_assisted':True}},
 'prediction':'K1 may already recover the slow observable reference, or a shared slow gap may remain; preserve either outcome and follow preregistered reads, no val selection.',
 'scripts':{n:ev(ROOT/'preexperiment/research_autopilot'/n) for n in ('peer12_cycle18_v1_20261101.py','run_peer12_cached_ceiling_v1_20261101.py','run_peer12_native_profile_v1_20261101.py','run_peer12_native_upper_v1_20261101.py','run_peer12_remote_lane_v1_20261101.py')}}
 atomic(CYCLE/'contract.json',c)
 for k in JOBS:update(k,{'status':'registered_ready','contract':ev(CYCLE/'contract.json'),'prediction_before_result':c['prediction'],'estimated_cost':c['cost'][k],'output':str((CYCLE/k).relative_to(ROOT))})
if __name__=='__main__':prepare()
