"""03h additive finite lanes; no model/event dispatch and no old-result edits."""
import argparse,os,subprocess,sys,time,traceback
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,atomic,ev,load,now,task
import peer10_cycle09_v1_20261101 as registry
CYCLE=AUDIT/'cycle03_peer12_v1_20261102'
C00=AUDIT/'cycle00_peer12_v1_20261102'
JOBS={k:('peer12_'+k+'_v1_20261102',task('K5') if k!='ffnet_allocator03' else 'autonomous_research_continuation_20261024') for k in ('lag_val_hold03','remote_upper_cumulative03','sources03','native_tie_review03','ffnet_allocator03')}
def update(k,v):
 if os.environ.get('PEER12_REMOTE_ONLY')=='1':
  atomic(CYCLE/(k+'_queue_delta.json'),{JOBS[k][0]:dict(id=JOBS[k][0],parent_task_id=JOBS[k][1],time=now(),**v)});return
 registry.JOBS.update(JOBS);registry.update(k,v)
def prepare():
 refs={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
 for n in refs:(AUTO/n).read_text()
 CYCLE.mkdir(exist_ok=False)
 atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
 prefix=AUDIT/'cycle21_peer12_v1/ffnet_runtime_v5/resume/manifest_v1.json'
 d=load(prefix);assert sum(s['conditions'] for s in d['shards'])==3584
 for s in d['shards']:assert ev(ROOT/s['path'])=={k:s[k] for k in ('path','sha256','bytes')}
 c={'created_at':now(),'instructions':refs,'script':ev(__file__),'old00':ev(C00/'contract.json'),'old_remote_v2':ev(C00/'remote_hold_upper_recovery_v2_contract.json'),'ffnet_prefix':ev(prefix),'ffnet_prefix_conditions':3584,'ffnet_old_replay':ev(AUDIT/'cycle21_peer12_v1/ffnet_runtime_v5/probe_replay.json'),'weights':load(C00/'contract.json')['lag_weights'],'official_main_unchanged':True,'annotation_references_not_deployable':True,'no_model_wakeup':True,'coverage_target':'2026-11-02T06:00:00+08:00 and beyond; finite identities, no padding/repeats','jobs':{
 'lag_val_hold03':{'prediction':'Previous-frame annotation reference is diagnostic for temporal coupling; complete PUBLIC/ONLINE models on val21 and hold14 without changing deployment recipe.','cost':{'full_forwards':46032,'profile_forwards':272,'GPU_wall_h':'3-5, actual thermal quota','new_disk_GiB':'<0.7 unique compressed predictions'}},
 'remote_upper_cumulative03':{'prediction':'Recover unexecuted hold slow-reference cells by supplying missing frozen helper, then cumulative official reference on both roles/models; compare slow and per-frame coupling without treating labels as physical truth.','cost':{'hold_upper_full_forwards':38992,'cumulative_full_forwards':46032,'profile_forwards':816,'GPU_wall_h':'4-8','new_disk_GiB':'<1.3 unique shards'}},
 'sources03':{'prediction':'Complete source/aux panels for new sole producer shards, no extra forwards or formal-cell selection.','cost':{'GPU_forwards':0,'CPU_threads':1,'CPU_wall_h':'4-12 plus finite producer dependencies','new_disk_GiB':'<1.0 compressed score/flag deltas'}},
 'native_tie_review03':{'prediction':'Independent native flags/counts are exactly equal, AP tiny drift may be float32/float64 quicksort tie order. Diagnose on saved outcomes before deciding; retain original failed full review.','cost':{'GPU_forwards':0,'CPU_wall_h':'0.1-2','new_disk_GiB':'<0.02 summaries only'}},
 'ffnet_allocator03':{'prediction':'Release unreferenced CUDA allocator blocks each condition; exact saved eight-condition replay gates continuation from committed3584 without modifying author network, inputs or outputs. Keep23GiB guard.','cost':{'remaining_formal_conditions':39460,'safe_replay_forwards':9,'GPU_wall_h':'5-12','new_disk_GiB':'<0.8 unique compressed predictions','optimizer_updates':0}}},
 'resources':'controlled slice + Wants/After thermal + pressure_guard, GPU threads2 workers0, CPUWeight25, GPU80 CPU70 continuous pacing,4090 450W unchanged','storage':'shared references only; retain old failures/prefixes, final weights and formal predictions; new output identities, no second project du'}
 atomic(CYCLE/'contract.json',c)
 for k,v in c['jobs'].items():update(k,dict(status='registered_finite_successor',contract=ev(CYCLE/'contract.json'),prediction_before_result=v['prediction'],estimated_cost=v['cost']))
def cell_source(kind,hold,profile):
 from run_peer12_native_additions_v1_20261102 import source_for,replace
 s=source_for('lag' if kind=='lag' else 'mixed',hold,profile)
 if kind=='lag':return s
 marker='from __future__ import annotations'
 constructor='upper_inputs' if kind=='upper' else 'cumulative_inputs'
 imports='\nfrom run_peer11_native_probes_v1_20261101 import upper_inputs' if kind=='upper' else '\nfrom peer12_cycle03_v1_20261102 import cumulative_inputs'
 assert s.count(marker)==1;s=s.replace(marker,marker+imports)
 return replace(s,[('maps={arm:{} for arm in arms}',f'upper={constructor}(cases,official,np);maps={{arm:{{}} for arm in arms}}'),("elif arm!='PUBLIC':raise ValueError(arm)","elif arm in ('OFFICIAL_SEQMEAN','OFFICIAL_MA5','OFFICIAL_CUMMEAN'):matrix[:2,3]=upper[arm,key[0],key[1]]\n                elif arm!='PUBLIC':raise ValueError(arm)")])
def cumulative_inputs(cases,official,np):
 groups={}
 for r in cases:
  key=(str(r['sequence']),int(r['receiver_timestamp_us']));d=official[str(r['current_vehicle_frame'])]['system_error_offset'];v=np.asarray([d['delta_x'],d['delta_y']],np.float64)
  if key in groups:assert np.array_equal(groups[key],v)
  groups[key]=v
 result={}
 for seq in sorted({k[0] for k in groups}):
  total=np.zeros(2)
  for i,t in enumerate(sorted(t for s,t in groups if s==seq)):
   total+=groups[seq,t];result['OFFICIAL_CUMMEAN',seq,t]=total/(i+1)
 return result
def native_cell(kind,hold,weight,gpu,out):
 from run_peer12_native_additions_v1_20261102 import run_cell
 import inspect
 source=inspect.getsource(run_cell);old="arms='OFFICIAL_LAG1' if kind=='lag' else 'PUBLIC,ONLINE_K1'";assert source.count(old)==1
 arms={'lag':'OFFICIAL_LAG1','upper':'OFFICIAL_SEQMEAN,OFFICIAL_MA5','cumulative':'OFFICIAL_CUMMEAN'}[kind]
 source=source.replace(old,'arms='+repr(arms))
 ns={'ROOT':ROOT,'CYCLE':C00,'load':load,'atomic':atomic,'ev':ev,'now':now,'time':time,'sys':sys,'__file__':__file__,'source_for':cell_source}
 from run_peer12_native_additions_v1_20261102 import lag_inputs
 ns['lag_inputs']=lag_inputs;exec(compile(source,__file__+'::profiled_cell','exec'),ns)
 ns['run_cell'](kind,weight,gpu,out,hold)
 # Generic helper only knows lag; explicitly correct diagnostic metadata in a new sidecar.
 atomic(out/'annotation_reference_identity.json',{'kind':kind,'annotation_assisted_not_deployable':True,'cumulative_includes_current_annotation':kind=='cumulative','main_protocol_not_changed':True})
def gpu_lane(remote):
 k='remote_upper_cumulative03' if remote else 'lag_val_hold03';out=CYCLE/k;out.mkdir(exist_ok=False);c=load(CYCLE/'contract.json');results=[]
 cells=[('upper',True)] if remote else []
 cells += [('cumulative',False),('cumulative',True)] if remote else [('lag',False),('lag',True)]
 update(k,dict(status='running_finite_worker',host='remote-gpu' if remote else 'local',unit='embodied-peer12-cycle03-'+('remote' if remote else 'gpu0')+'-v1-20261102.service'))
 for kind,hold in cells:
  for name,ref in c['weights'].items():
   dest=out/(kind+'_'+('hold14' if hold else 'val21'))/name.lower()
   try:
    assert ev(ROOT/ref['path'])==ref;native_cell(kind,hold,ROOT/ref['path'],0,dest);results.append({'cell':str(dest.relative_to(CYCLE)),'terminal':ev(dest/'terminal.json')})
   except Exception as e:
    f=out/(kind+'_'+str(hold)+'_'+name+'_failure.json');atomic(f,dict(status='failed_retained',error=repr(e),traceback=traceback.format_exc(),created_at=now()));results.append({'failure':ev(f)})
   atomic(out/'status.json',dict(status='finite_chain_running',results=results,created_at=now()))
 status='complete_light_checked' if all('failure' not in r for r in results) else 'failed_retained'
 atomic(out/'terminal.json',dict(status=status,results=results,created_at=now(),annotation_assisted_not_deployable=True,no_model_wakeup=True));update(k,dict(status=status,terminal=ev(out/'terminal.json')))
def source_lane(remote):
 from run_peer12_native_source_additions_v1_20261101 import compute_namespace
 out=CYCLE/('source_remote' if remote else 'sources03');out.mkdir(exist_ok=False);ns,old=compute_namespace();ns['CYCLE']=CYCLE;results=[]
 k='remote_upper_cumulative03' if remote else 'lag_val_hold03'
 cells=[('upper',True),('cumulative',False),('cumulative',True)] if remote else [('lag',False),('lag',True)]
 for kind,hold in cells:
  for name in ('public','online_k1'):
   dest=CYCLE/k/(kind+'_'+('hold14' if hold else 'val21'))/name/'evaluation';t=dest/'terminal.json';chain=CYCLE/k/'terminal.json'
   while not t.exists() and not chain.exists():atomic(out/'status.json',dict(status='waiting_registered_producer',producer=str(dest),created_at=now()));time.sleep(30)
   try:
    assert t.exists(),('producer chain ended',str(dest));producer=str(dest.relative_to(AUDIT));ns['compute'](producer,old);results.append({'producer':producer,'terminal':ev(CYCLE/'native_source_aux'/producer/'terminal.json')})
   except Exception as e:results.append({'producer':str(dest),'error':repr(e),'traceback':traceback.format_exc()})
   atomic(out/'status.json',dict(status='source_scoring',results=results,created_at=now()))
 status='complete_light_checked' if all('error' not in r for r in results) else 'failed_retained';atomic(out/'terminal.json',dict(status=status,results=results,created_at=now(),no_model_wakeup=True))
 if not remote:update('sources03',dict(status=status,terminal=ev(out/'terminal.json')))
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gpu0','remote','cpu','source_remote']);a=p.parse_args()
 if a.mode=='prepare':return prepare()
 assert 'embodied-research.slice' in Path('/proc/self/cgroup').read_text()
 if a.mode in ('gpu0','remote'):return gpu_lane(a.mode=='remote')
 source_lane(a.mode=='source_remote')
if __name__=='__main__':main()
