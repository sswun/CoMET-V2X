"""New hold14 unit: profile64, optional replay-gated raw LRU, frozen scoring."""
import argparse,sys,time
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle21_v1_20261101 import CYCLE
import run_peer12_native_profile_v1_20261101 as profile
from run_peer10_k4_hold14_native_v1_20261101 import CHANGES

def replace_once(source,changes):
 for old,new in changes:assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
 return source

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c=load(CYCLE/'contract.json');spec=c['holds'][a.job];weight=ROOT/spec['weight']['path'];assert ev(weight)==spec['weight'];out=CYCLE/a.job;out.mkdir(exist_ok=False);start=time.monotonic()
 def run(source,destination,cache):
  sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(AUDIT/'contract_v1.json'),'--output',str(destination),'--weight',str(weight),'--gpu',str(spec['gpu'])]
  ns={'__name__':'peer12_hold_profiled','__file__':__file__,'install_reader':profile.install_reader,'attach_projection_timer':profile.attach_projection_timer,'cache_enabled':cache}
  exec(compile(source,__file__+'::frozen_hold','exec'),ns);ns['main']()
 source=replace_once(profile.adapted_source(),CHANGES)
 run(source,out/'profile_uncached',False);rows=load(out/'profile_uncached/profile_rows.json');assert len(rows)==192
 totals={k:sum(r[k] for r in rows) for k in ('data_total_s','read_parse_s','projection_voxel_s','forward_s','AP_accounting_s')};den=sum(totals[k] for k in ('data_total_s','forward_s','AP_accounting_s'));fraction=totals['read_parse_s']/den;eligible=False;comparisons=[]
 if fraction>=.3:
  run(source,out/'profile_cached',True)
  import numpy as np
  for w in (0,100,200,400):
   with np.load(out/'profile_uncached'/f'wait{w:03d}.npz') as x,np.load(out/'profile_cached'/f'wait{w:03d}.npz') as y:
    same=x.files==y.files and all(x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes() for k in x.files);comparisons.append({'wait_ms':w,'bitwise_equal':same})
  eligible=all(r['bitwise_equal'] for r in comparisons)
 atomic(out/'profile_terminal.json',{'status':'complete_light_checked','created_at':now(),'conditions':64,'forwards':204*(2 if fraction>=.3 else 1),'timing_s':totals,'read_parse_fraction':fraction,'raw_cache_eligible':eligible,'replays':comparisons,'no_scientific_AP_claim':True,'wall_s':time.monotonic()-start})
 path=ROOT/c['native_evaluator']['path'];assert ev(path)==c['native_evaluator'];source=replace_once(path.read_text(),CHANGES)
 source=replace_once(source,[('h,builder,roots,_,_=build(root);official=','h,builder,roots,_,_=build(root);reader=install_reader(cache_enabled);official=')])
 run(source,out/'evaluation',eligible)
 terminal=load(out/'evaluation/terminal.json');assert terminal['status']=='native_AP_complete_light_checked' and terminal['cases']==9744
 atomic(out/'terminal.json',{'status':'complete_light_checked','created_at':now(),'contract':ev(CYCLE/'contract.json'),'weight':spec['weight'],'profile':ev(out/'profile_terminal.json'),'full_evaluation':ev(out/'evaluation/terminal.json'),'wall_s_including_profile':time.monotonic()-start,'source_and_full_independent_review_pending':True,'no_model_wakeup':True})
if __name__=='__main__':main()
