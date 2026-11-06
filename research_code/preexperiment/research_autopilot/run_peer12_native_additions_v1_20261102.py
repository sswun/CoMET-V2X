"""New profiled native cells only; no modification of running frozen chains."""
import argparse,sys,time
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle00_v1_20261102 import CYCLE

def replace(source,pairs):
 for old,new in pairs:
  assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
 return source

def lag_inputs(cases,official,np):
 groups={}
 for r in cases:
  key=(str(r['sequence']),int(r['receiver_timestamp_us']));d=official[str(r['current_vehicle_frame'])]['system_error_offset'];v=np.asarray([d['delta_x'],d['delta_y']],np.float64)
  if key in groups:assert np.array_equal(groups[key],v)
  groups[key]=v
 result={}
 for seq in sorted({k[0] for k in groups}):
  times=sorted(t for s,t in groups if s==seq)
  for i,t in enumerate(times):result[seq,t]=groups[seq,times[i-1]].copy() if i else np.zeros(2)
 return result

def source_for(kind,hold,profile):
 c=load(CYCLE/'contract.json')
 for k in ('evaluator_source','profile_source','hold_adapter'):assert ev(ROOT/c[k]['path'])==c[k]
 import run_peer12_native_profile_v1_20261101 as prof
 from run_peer10_k4_hold14_native_v1_20261101 import CHANGES
 source=prof.adapted_source() if profile else (ROOT/c['evaluator_source']['path']).read_text()
 if hold:source=replace(source,CHANGES)
 if not profile:source=replace(source,[('h,builder,roots,_,_=build(root);official=','h,builder,roots,_,_=build(root);reader=install_reader(cache_enabled);official='),('if refresh.exists():subprocess.run','if False and refresh.exists():subprocess.run')])
 if kind=='lag':source=replace(source,[('maps={arm:{} for arm in arms}','lag=lag_inputs(cases,official,np);maps={arm:{} for arm in arms}'),("elif arm!='PUBLIC':raise ValueError(arm)","elif arm=='OFFICIAL_LAG1':matrix[:2,3]=lag[key[0],key[1]]\n                elif arm!='PUBLIC':raise ValueError(arm)")])
 return replace(source,[('torch.set_num_threads(1)','torch.set_num_threads(2)')])

def run_cell(kind,weight,gpu,out,hold=False):
 import run_peer12_native_profile_v1_20261101 as prof
 c=load(CYCLE/'contract.json');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();arms='OFFICIAL_LAG1' if kind=='lag' else 'PUBLIC,ONLINE_K1';narm=len(arms.split(','))
 def run(profile,destination,cache):
  sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(AUDIT/'contract_v1.json'),'--output',str(destination),'--weight',str(weight),'--gpu',str(gpu),'--arms',arms]
  ns={'__name__':'peer12_new_native_cell','__file__':__file__,'install_reader':prof.install_reader,'attach_projection_timer':prof.attach_projection_timer,'cache_enabled':cache,'lag_inputs':lag_inputs}
  exec(compile(source_for(kind,hold,profile),__file__+'::'+kind,'exec'),ns);ns['main']()
 run(True,out/'profile_uncached',False);rows=load(out/'profile_uncached/profile_rows.json');assert len(rows)==64*narm
 totals={k:sum(r[k] for r in rows) for k in ('data_total_s','read_parse_s','projection_voxel_s','forward_s','AP_accounting_s')};fraction=totals['read_parse_s']/sum(totals[k] for k in ('data_total_s','forward_s','AP_accounting_s'));eligible=False;replays=[]
 if fraction>=.3:
  import numpy as np
  run(True,out/'profile_cached',True)
  for w in (0,100,200,400):
   with np.load(out/'profile_uncached'/f'wait{w:03d}.npz') as x,np.load(out/'profile_cached'/f'wait{w:03d}.npz') as y:
    exact=x.files==y.files and all(x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes() for k in x.files);replays.append({'wait_ms':w,'bitwise_equal':exact})
  eligible=all(r['bitwise_equal'] for r in replays)
 atomic(out/'profile_terminal.json',{'status':'complete_light_checked','conditions':64,'input_arms':narm,'forwards':68*narm*(2 if fraction>=.3 else 1),'timings':totals,'read_parse_fraction':fraction,'raw_cache_eligible':eligible,'replays':replays,'not_scientific_AP':True})
 run(False,out/'evaluation',eligible);t=load(out/'evaluation/terminal.json');assert t['status']=='native_AP_complete_light_checked' and t['cases']==(9744 if hold else 13264)
 atomic(out/'terminal.json',{'status':'complete_light_checked','created_at':now(),'kind':kind,'weight':ev(weight),'evaluation':ev(out/'evaluation/terminal.json'),'profile':ev(out/'profile_terminal.json'),'wall_s':time.monotonic()-start,'annotation_assisted_not_deployable':kind=='lag','no_model_wakeup':True,'source_full_independent_pending':True})

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();c=load(CYCLE/'contract.json');out=CYCLE/'lag00';out.mkdir(exist_ok=False);results=[]
 for name in ('PUBLIC','ONLINE_K1'):
  ref=c['lag_weights'][name];assert ev(ROOT/ref['path'])==ref;run_cell('lag',ROOT/ref['path'],a.gpu,out/name.lower());results.append(ev(out/name.lower()/'terminal.json'))
 atomic(out/'terminal.json',{'status':'complete_light_checked','created_at':now(),'results':results,'annotation_assisted_not_deployable':True,'no_model_wakeup':True})
if __name__=='__main__':main()
