"""Missing holds/new cells only. No duplicate scoring of old identities."""
import time,traceback
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle00_v1_20261102 import CYCLE

def main():
 out=CYCLE/'sources00';out.mkdir(exist_ok=False);c=load(CYCLE/'contract.json');assert ev(ROOT/c['source_adapter']['path'])==c['source_adapter'];assert ev(ROOT/c['source_contract']['path'])==c['source_contract']
 from run_peer12_native_source_additions_v1_20261101 import compute_namespace
 ns,old=compute_namespace();ns['CYCLE']=CYCLE
 base=str(CYCLE.relative_to(AUDIT));jobs=[
  ('cycle15_peer11_v1/hold14_seed2_public_correct_env','seed2_public_hold'),
  ('cycle21_peer12_v1/hold2_online21/evaluation','seed2_online_hold'),
  ('cycle21_peer12_v1/hold3_public21/evaluation','seed3_public_hold'),
  ('cycle21_peer12_v1/hold3_online21/evaluation','seed3_online_hold'),
  (base+'/mixed00/val21/evaluation','mixed_val'),
  (base+'/mixed00/hold14/evaluation','mixed_hold'),
  (base+'/lag00/public/evaluation','lag_public'),
  (base+'/lag00/online_k1/evaluation','lag_online')];results=[]
 for name,label in jobs:
  try:
   terminal=AUDIT/name/'terminal.json'
   while not terminal.exists():
    lane='gpu0' if label.startswith('mixed') else 'gpu1';job='mixed00' if lane=='gpu0' else 'lag00'
    failure=CYCLE/(lane+'_chain')/(job+'_failure.json')
    if label.startswith(('mixed','lag')) and failure.exists():raise RuntimeError(('producer failed',str(failure)))
    outer=AUDIT/name.rsplit('/evaluation',1)[0]/'terminal.json'
    if outer.exists() and load(outer).get('status')=='failed_retained':raise RuntimeError(('failed producer',str(outer)))
    atomic(out/'status.json',{'status':'waiting_registered_producer','method':name,'time':now(),'zero_GPU_forwards':True});time.sleep(30)
   ns['compute'](name,old);results.append({'label':label,'method':name,'terminal':ev(CYCLE/'native_source_aux'/name/'terminal.json')})
  except Exception as e:results.append({'label':label,'method':name,'error':repr(e),'traceback':traceback.format_exc()})
  atomic(out/'status.json',{'status':'missing_source_scoring','results':results,'time':now(),'zero_GPU_forwards':True})
 atomic(out/'terminal.json',{'status':'complete_light_checked' if all('error' not in r for r in results) else 'failed_retained','created_at':now(),'results':results,'no_model_wakeup':True,'not_whole_K5_release':True})
if __name__=='__main__':main()
