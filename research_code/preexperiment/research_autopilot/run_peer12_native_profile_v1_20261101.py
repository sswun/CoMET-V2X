"""64-condition native profile; optional raw-read-only LRU, replay-gated."""
import argparse,copy,sys,time,collections
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle18_v1_20261101 import CYCLE

class RawReader:
 def __init__(self,original,cache=False):
  self.original=original;self.enabled=cache;self.lru=collections.OrderedDict();self.bytes=0;self.read_s=0.;self.calls=0;self.hits=0
 def __call__(self,*args,**kwargs):
  tick=time.monotonic();key=(tuple(map(str,args)),tuple(sorted((k,str(v)) for k,v in kwargs.items())))
  if self.enabled and key in self.lru:
   self.hits+=1;value=self.lru.pop(key);self.lru[key]=value;result=value.copy()
  else:
   result=self.original(*args,**kwargs)
   if self.enabled and result.nbytes<=1<<30:
    value=result.copy();self.lru[key]=value;self.bytes+=value.nbytes
    while len(self.lru)>256 or self.bytes>1<<30:
     _,old=self.lru.popitem(last=False);self.bytes-=old.nbytes
  self.read_s+=time.monotonic()-tick;self.calls+=1;return result

def install_reader(cache=False):
 from utils import pcd_utils
 if isinstance(pcd_utils.pcd_to_np,RawReader):pcd_utils.pcd_to_np=pcd_utils.pcd_to_np.original
 reader=RawReader(pcd_utils.pcd_to_np,cache);pcd_utils.pcd_to_np=reader;return reader

def adapted_source():
 p=ROOT/'preexperiment/research_autopilot/run_peer10_native_evaluate_v1_20261101.py';ref=load(AUDIT/'cycle15_peer11_v1/contract.json')['native_probe']['source'];assert ev(p)==ref;source=p.read_text()
 changes=[("h,builder,roots,_,_=build(root);official=", "h,builder,roots,_,_=build(root);reader=install_reader(cache_enabled);profile_rows=[];official="),
 ("selected=[r for r in cases if int(r['wait_ms'])==wait]", "selected_all=[r for r in cases if int(r['wait_ms'])==wait];selected=[selected_all[int(j)] for j in np.linspace(0,len(selected_all)-1,16,dtype=int)]"),
 ("datasets[arm]=ds;stats[arm]=", "attach_projection_timer(ds);datasets[arm]=ds;stats[arm]="),
 ("ds=datasets[arm];np.random.seed(2023+index);batch=ds.collate_batch([ds[index]]);audit=ds.native4_audit", "ds=datasets[arm];np.random.seed(2023+index);tick_data=time.monotonic();read_before=reader.read_s;projection_before=ds.peer12_projection_s;batch=ds.collate_batch([ds[index]]);data_s=time.monotonic()-tick_data;read_s=reader.read_s-read_before;projection_s=ds.peer12_projection_s-projection_before;audit=ds.native4_audit"),
 ("stat,_=common.evaluate_frame", "tick_ap=time.monotonic();stat,_=common.evaluate_frame"),
 ("common.merge_stat(stats[arm],stat)", "common.merge_stat(stats[arm],stat);profile_rows.append({'wait_ms':wait,'index':index,'arm':arm,'data_total_s':data_s,'read_parse_s':read_s,'projection_voxel_s':projection_s,'forward_s':forward_ms[arm]/1000,'AP_accounting_s':time.monotonic()-tick_ap});atomic(out/'profile_rows.json',profile_rows)"),
 ("if refresh.exists():subprocess.run", "if False and refresh.exists():subprocess.run")]
 for old,new in changes:assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
 return source

def attach_projection_timer(ds):
 original=ds.get_item_single_car;ds.peer12_projection_s=0.
 def timed(*args,**kwargs):
  tick=time.monotonic()
  try:return original(*args,**kwargs)
  finally:ds.peer12_projection_s+=time.monotonic()-tick
 ds.get_item_single_car=timed

def run_profile(out,weight,gpu,cache):
 sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(AUDIT/'contract_v1.json'),'--output',str(out),'--weight',str(weight),'--gpu',str(gpu)]
 ns={'__name__':'peer12_profile_evaluator','__file__':__file__,'install_reader':install_reader,'attach_projection_timer':attach_projection_timer,'cache_enabled':cache}
 exec(compile(adapted_source(),__file__+'::profile64','exec'),ns);ns['main']()

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args();out=CYCLE/'profile';out.mkdir(exist_ok=False);c=load(CYCLE/'contract.json');weight=ROOT/c['weights']['e60']['path'];start=time.monotonic()
 run_profile(out/'uncached',weight,a.gpu,False);rows=load(out/'uncached/profile_rows.json');assert len(rows)==192
 totals={k:sum(r[k] for r in rows) for k in ('data_total_s','read_parse_s','projection_voxel_s','forward_s','AP_accounting_s')};denominator=sum(totals[k] for k in ('data_total_s','forward_s','AP_accounting_s'));fraction=totals['read_parse_s']/denominator;gate=False;comparison=[]
 # A cache may only replace raw file reads; transformed/voxelized values are
 # always recomputed. Every cache hit returns a copy because author code mutates.
 if fraction>=.3:
  run_profile(out/'cached',weight,a.gpu,True)
  import numpy as np
  for wait in (0,100,200,400):
   with np.load(out/'uncached'/f'wait{wait:03d}.npz') as left,np.load(out/'cached'/f'wait{wait:03d}.npz') as right:
    exact=left.files==right.files and all(left[k].dtype==right[k].dtype and left[k].shape==right[k].shape and left[k].tobytes()==right[k].tobytes() for k in left.files);comparison.append({'wait_ms':wait,'all_prediction_and_GT_arrays_bitwise_equal':exact})
  gate=all(r['all_prediction_and_GT_arrays_bitwise_equal'] for r in comparison)
 atomic(out/'terminal.json',{'status':'complete_light_checked','created_at':now(),'contract':ev(CYCLE/'contract.json'),'conditions':64,'input_arms':3,'timing_totals_s':totals,'other_data_s':totals['data_total_s']-totals['read_parse_s']-totals['projection_voxel_s'],'read_fraction':fraction,'cache_attempted':fraction>=.3,'cache_eligible':gate,'cache_replay':comparison,'cache_max_bytes':1<<30,'cache_max_frames':256,'wall_s':time.monotonic()-start,'scientific_AP_result':False,'no_model_wakeup':True})
if __name__=='__main__':main()
