"""Preserve zero-forward namespace failures; supply AUDIT in a new version."""
import argparse,sys
from peer12_cycle03_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,atomic,ev,load,now,task
import peer10_cycle09_v1_20261101 as registry
def adapted():
 c=load(CYCLE/'contract.json');p=ROOT/c['script']['path'];assert ev(p)==c['script'];s=p.read_text()
 old="ns={'ROOT':ROOT,'CYCLE':C00";assert s.count(old)==1;s=s.replace(old,"ns={'ROOT':ROOT,'AUDIT':AUDIT,'CYCLE':C00")
 for old in ('lag_val_hold03','remote_upper_cumulative03','sources03'):s=s.replace(old,old+'_v2')
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gpu0','remote','cpu','source_remote']);a=p.parse_args()
 if a.mode=='prepare':
  refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
  for n in refs:(AUTO/n).read_text()
  compile(adapted(),__file__,'exec')
  c={'created_at':now(),'instructions':refs,'old_contract':ev(CYCLE/'contract.json'),'script':ev(__file__),'fix':'Add missing AUDIT global to copied helper namespace; no GPU calls in original failure, retain old outputs. New v2 paths and tasks. Include e60 reference with matched models explicitly; not a deployment method.','GPU_forwards_old_failure':0,'cost_multiplier_vs_two_weight_plan':1.5,'CPU_threads':1,'no_model_wakeup':True}
  atomic(CYCLE/'recovery_v2_contract.json',c)
  for old in ('lag_val_hold03','remote_upper_cumulative03','sources03'):
   registry.JOBS[old+'_v2']=('peer12_'+old+'_v2_v1_20261102',task('K5'));registry.update(old+'_v2',dict(status='registered_finite_successor',contract=ev(CYCLE/'recovery_v2_contract.json'),prediction_before_result=load(CYCLE/'contract.json')['jobs'][old]['prediction'],estimated_cost=load(CYCLE/'contract.json')['jobs'][old]['cost'],cost_multiplier_three_reference_weights=1.5))
  return
 ns={'__name__':'peer12_cycle03_namespace_recovery_v2','__file__':str(ROOT/'preexperiment/research_autopilot/peer12_cycle03_v1_20261102.py')};exec(compile(adapted(),__file__+'::namespace_only','exec'),ns)
 sys.argv=[__file__,a.mode];ns['main']()
if __name__=='__main__':main()
