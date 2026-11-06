"""Peer12 wait0-first native annotation-reference groups, frozen old source."""
import argparse,sys,subprocess,time
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle18_v1_20261101 import CYCLE
from run_peer11_native_probes_v1_20261101 import upper_inputs
from run_peer12_native_profile_v1_20261101 import install_reader

def evaluator_source(waits):
 ref=load(CYCLE/'contract.json');old=load(ROOT/ref['native_contract']['path']);p=ROOT/old['native_probe']['source']['path'];assert ev(p)==old['native_probe']['source'];s=p.read_text()
 changes=[('maps={arm:{} for arm in arms}','upper=upper_inputs(cases,official,np);maps={arm:{} for arm in arms}'),
 ("elif arm!='PUBLIC':raise ValueError(arm)","elif arm in ('OFFICIAL_SEQMEAN','OFFICIAL_MA5'):matrix[:2,3]=upper[arm,key[0],key[1]]\n                elif arm!='PUBLIC':raise ValueError(arm)"),
 ('for wait in (0,100,200,400):','for wait in '+str(tuple(waits))+':'),
 ('h,builder,roots,_,_=build(root);official=',"h,builder,roots,_,_=build(root);install_reader(cache_eligible);official="),
 ('if refresh.exists():subprocess.run','if False and refresh.exists():subprocess.run')]
 for a,b in changes:assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
 return s

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--weight-name',choices=['PUBLIC','ONLINE_K1']);p.add_argument('--phase',choices=['wait0','remaining']);a=p.parse_args();c=load(CYCLE/'contract.json');out=CYCLE/'upper'
 if a.weight_name:
  profile=load(CYCLE/'profile/terminal.json');cache=profile['cache_eligible'];ref=c['weights'][a.weight_name];assert ev(ROOT/ref['path'])==ref
  dest=out/a.phase/a.weight_name.lower();waits=[0] if a.phase=='wait0' else [100,200,400]
  sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(ROOT/c['native_contract']['path']),'--output',str(dest),'--weight',str(ROOT/ref['path']),'--gpu',str(a.gpu),'--arms','OFFICIAL_SEQMEAN,OFFICIAL_MA5']
  ns={'__name__':'peer12_wait0_first_upper','__file__':__file__,'upper_inputs':upper_inputs,'install_reader':install_reader,'cache_eligible':cache};exec(compile(evaluator_source(waits),__file__+'::upper','exec'),ns);ns['main']();return
 out.mkdir(exist_ok=False);results=[];start=time.monotonic()
 for phase in ('wait0','remaining'):
  (out/phase).mkdir()
  for name in ('PUBLIC','ONLINE_K1'):
   with (out/f'{phase}_{name}.log').open('x') as f:rc=subprocess.run([str(ROOT/'.conda/opencood-linux/bin/python'),'-u',__file__,'--gpu',str(a.gpu),'--weight-name',name,'--phase',phase],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
   term=out/phase/name.lower()/'terminal.json';assert rc==0 and load(term)['status']=='native_AP_complete_light_checked',(phase,name,rc)
   results.append({'phase':phase,'training':name,'terminal':ev(term)});atomic(out/'status.json',{'status':'running_wait0_first_reference','results':results,'wall_s':time.monotonic()-start})
 atomic(out/'terminal.json',{'status':'complete_light_checked','created_at':now(),'results':results,'wall_s':time.monotonic()-start,'all_annotation_assisted_not_deployable':True,'shared_constructor':'peer11.upper_inputs imported unchanged','source_AP_pending':True,'no_model_wakeup':True})
if __name__=='__main__':main()
