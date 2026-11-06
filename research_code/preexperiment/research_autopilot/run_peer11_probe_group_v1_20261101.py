"""Finite peer11 three-model ego probe or two-model slow/jitter reference."""
import argparse,subprocess,time
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,atomic,ev,load,now
from peer11_cycle15_v1_20261101 import CYCLE
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',choices=['ego','upper'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();c=load(CYCLE/'contract.json');out=CYCLE/('ego_probe' if a.group=='ego' else 'upper_split');out.mkdir(exist_ok=False);results=[];start=time.monotonic()
 names=['e60','PUBLIC','ONLINE_K1'] if a.group=='ego' else ['PUBLIC','ONLINE_K1']
 for name in names:
  ref=c['native_probe']['weights'][name];assert ev(ROOT/ref['path'])==ref;dest=out/name.lower()
  cmd=[ROOT/'.conda/opencood-linux/bin/python','-u',ROOT/'preexperiment/research_autopilot/run_peer11_native_probes_v1_20261101.py','--mode',a.group,'--workspace',ROOT,'--contract',CYCLE/'contract.json','--output',dest,'--weight',ROOT/ref['path'],'--gpu',str(a.gpu)]
  with (out/(name+'.log')).open('x') as log:rc=subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  results.append({'weight_name':name,'weight':ref,'returncode':rc,'terminal':ev(dest/'terminal.json') if (dest/'terminal.json').exists() else None});atomic(out/'status.json',{'status':'running_finite_group','results':results,'wall_s':time.monotonic()-start})
 atomic(out/'terminal.json',{'status':'complete_light_checked' if all(r['returncode']==0 for r in results) else 'failed_retained','created_at':now(),'results':results,'wall_s':time.monotonic()-start,'GT_inference_inputs':False,'upper_annotation_assisted':a.group=='upper','no_model_wakeup':True})
if __name__=='__main__':main()
