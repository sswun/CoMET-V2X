"""Finite missing native source/aux panels using sole producer predictions."""
import argparse,subprocess,time,traceback
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle18_v1_20261101 import CYCLE

def compute_namespace():
 c=load(AUDIT/'cycle15_peer11_v1/contract.json');ref=c['source_successor']['source'];assert ev(ROOT/ref['path'])==ref;s=(ROOT/ref['path']).read_text()
 changes=[("frozen={'producer_terminal'","waits=sorted({r['wait_ms'] for r in load(producer/'records.json')});frozen={'producer_terminal'"),
 ("assert len(records)==13264 and len({(r['wait_ms'],r['global_index']) for r in records})==13264","assert len({(r['wait_ms'],r['global_index']) for r in records})==len(records)"),
 ('assert len(selected)==3316',"assert len(selected)==expected[wait]['cases']"),
 ('assert len(panels)==4*3*2*len(arms)*5*2','assert len(panels)==len(waits)*3*2*len(arms)*5*2')]
 for old,new in changes:assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
 assert s.count('(0,100,200,400)')==2;s=s.replace('(0,100,200,400)','waits')
 ns={'__name__':'peer12_source_additions','__file__':__file__};exec(compile(s,__file__+'::variable_roles_waits','exec'),ns);ns['CYCLE']=CYCLE;return ns,c

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['hold','remote'],required=True);a=p.parse_args();out=CYCLE/('source_'+a.mode);out.mkdir(exist_ok=False);ns,c=compute_namespace();results=[]
 names=['cycle09_v1/hold14_e60','cycle09_v1/hold14_public','cycle09_v1/hold14_online_k1'] if a.mode=='hold' else ['cycle15_peer11_v1/ego_probe/'+n for n in ('e60','public','online_k1')]+['cycle18_peer12_v1/upper/'+phase+'/'+n for phase in ('wait0','remaining') for n in ('public','online_k1')]
 for name in names:
  try:
   terminal=AUDIT/name/'terminal.json'
   while not terminal.exists():
    failed=CYCLE/'remote_chain'/('ego_probe_failure.json' if 'ego_probe' in name else 'upper_failure.json')
    if failed.exists():raise RuntimeError(('producer failure retained',str(failed)))
    atomic(out/'status.json',{'status':'waiting_known_prediction_identity','method':name,'time':now()});time.sleep(30)
   ns['compute'](name,c);results.append({'method':name,'terminal':ev(CYCLE/'native_source_aux'/name/'terminal.json')})
  except Exception as e:results.append({'method':name,'error':repr(e),'traceback':traceback.format_exc()})
  atomic(out/'status.json',{'status':'scoring_missing_native_panels','results':results,'time':now(),'zero_GPU_forwards':True})
 atomic(out/'terminal.json',{'status':'complete_light_checked' if all('error' not in r for r in results) else 'failed_retained','results':results,'created_at':now(),'no_model_wakeup':True,'full_K5_release_not_assumed':True})
if __name__=='__main__':main()
