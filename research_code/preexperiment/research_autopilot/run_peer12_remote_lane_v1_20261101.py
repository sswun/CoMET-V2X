"""Finite remote continuation. Local queues authoritative; no event/model call."""
import subprocess,time,traceback
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle18_v1_20261101 import CYCLE,JOBS
def main():
 out=CYCLE/'remote_chain';out.mkdir(exist_ok=False);deltas={};results=[]
 old=AUDIT/'cycle15_peer11_v1';c=load(CYCLE/'contract.json')
 unit='embodied-peer10-k4-seed2-official-v1-20261101.service'
 while subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState'],text=True).strip() in ('ActiveState=active','ActiveState=activating'):
  atomic(out/'status.json',{'status':'waiting_registered_predecessor','unit':unit,'time':now()});time.sleep(30)
 jobs=[('profile','run_peer12_native_profile_v1_20261101.py',['--gpu','0'],CYCLE/'profile/terminal.json'),
       ('ego_probe','run_peer11_probe_group_v1_20261101.py',['--group','ego','--gpu','0'],old/'ego_probe/terminal.json'),
       ('upper','run_peer12_native_upper_v1_20261101.py',['--gpu','0'],CYCLE/'upper/terminal.json')]
 from peer11_cycle15_v1_20261101 import JOBS as OLDJOBS
 for key,script,args,terminal in jobs:
  tid,pid=(OLDJOBS if key=='ego_probe' else JOBS)[key]
  try:
   ref=(load(old/'contract.json')['scripts'] if key=='ego_probe' else c['scripts'])[script];assert ev(ROOT/ref['path'])==ref
   deltas[tid]={'id':tid,'parent_task_id':pid,'status':'running_remote_finite_worker','time':now()};atomic(out/'queue_deltas.json',deltas)
   with (out/(key+'.log')).open('x') as log:rc=subprocess.run([str(ROOT/'.conda/opencood-linux/bin/python'),'-u',str(ROOT/'preexperiment/research_autopilot'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
   assert rc==0 and load(terminal)['status']=='complete_light_checked',(key,rc)
   deltas[tid].update(status='complete_light_checked',terminal=ev(terminal),time=now());results.append({'key':key,'terminal':ev(terminal)})
  except Exception as e:
   failure=out/(key+'_failure.json');atomic(failure,{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc(),'time':now()});deltas[tid]={'id':tid,'parent_task_id':pid,'status':'failed_retained','terminal':ev(failure),'time':now()};results.append({'key':key,'failure':ev(failure)})
   # Profiling failure disables cache but must not discard the known scientific
   # successors: a small fallback profile record explicitly permits uncached.
   if key=='profile' and not terminal.exists():atomic(terminal,{'status':'failed_retained','cache_eligible':False,'failure':ev(failure)})
  atomic(out/'queue_deltas.json',deltas);atomic(out/'status.json',{'status':'finite_chain_running','completed':results,'time':now(),'no_model_wakeup':True})
 atomic(out/'terminal.json',{'status':'finite_chain_terminal','results':results,'created_at':now(),'no_model_wakeup':True,'local_queues_authoritative':True})
if __name__=='__main__':main()
