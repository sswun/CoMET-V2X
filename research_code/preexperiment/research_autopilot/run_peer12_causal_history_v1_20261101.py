"""One bounded causal sequence hypothesis, frozen on fit/cal, no GT inference."""
import os,time,resource,traceback,statistics
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import run_peer10_se2_expanded_anchor_v1_20261101 as science
import run_peer10_fixed_population_aux_v1_20261101 as aux
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle21_v1_20261101 import CYCLE,update
core=science.core
NAME='causal_history_geomedian'

def median(points):
 points=np.asarray(points,np.float64)
 if not len(points):raise ValueError('empty spatial median')
 x=points.mean(axis=0)
 for _ in range(128):
  w=1/np.maximum(np.linalg.norm(points-x,axis=1),1e-8);new=(points*w[:,None]).sum(axis=0)/w.sum()
  if np.linalg.norm(new-x)<1e-10:return new
  x=new
 return x

def rotation(pose):
 yaw=np.arctan2(pose[1,0],pose[0,0]);c,s=np.cos(yaw),np.sin(yaw)
 return np.asarray([[c,-s],[s,c]])

def causal(rows,observations):
 states={};seen=set();out=[]
 for row,obs in zip(rows,observations):
  key=(str(row['sequence']),int(row['wait_ms']));stamp=int(row['receiver_timestamp_us']);assert (*key,stamp) not in seen;seen.add((*key,stamp))
  R=rotation(np.asarray(row['current_ego_pose_lidar_to_world'],np.float64))
  if key not in states:states[key]={'R0':R.copy(),'stamp':stamp-1,'history':[],'value':np.zeros(2)}
  st=states[key];assert stamp>st['stamp'];transport=st['R0'].T@R
  if len(obs):st['history'].append(transport@median(obs));st['value']=median(st['history'])
  out.append(transport.T@st['value']);st['stamp']=stamp
 return np.asarray(out)

def synthetic():
 rows=[];obs=[]
 for i in range(6):
  R=np.eye(4);a=.15*i;R[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
  rows.append({'sequence':'s','wait_ms':0,'receiver_timestamp_us':i,'current_ego_pose_lidar_to_world':R.tolist()});obs.append(np.asarray([[.2,-.3],[.3,-.2],[8,9]]) if i!=2 else np.empty((0,2)))
 full=causal(rows,obs);assert np.array_equal(full[:3],causal(rows[:3],obs[:3]));assert np.allclose(full,causal(rows,[np.tile(v,(3,1)) for v in obs]),atol=1e-8)
 assert np.allclose(median(np.asarray([[1,0],[-1,0],[0,2]])),[0,1/np.sqrt(3)],atol=1e-5)
 # A rigid rotation of every local coordinate and public yaw must rotate outputs.
 angle=.4;G=np.asarray([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 rr=[]
 for r in rows:
  r=dict(r);p=np.asarray(r['current_ego_pose_lidar_to_world']);p[:2,:2]=p[:2,:2]@G.T;r['current_ego_pose_lidar_to_world']=p.tolist();rr.append(r)
 assert np.allclose(causal(rr,[v@G.T for v in obs]),full@G.T,atol=1e-8)
 separate=[dict(r,sequence='other',wait_ms=400) for r in rows]
 assert np.array_equal(causal(rows+separate,obs+obs)[len(rows):],full)
 no=[np.empty((0,2))]*len(rows);assert np.array_equal(causal(rows,no),np.zeros((len(rows),2)))
 return {'status':'passed','future_prefix_invariant':True,'whole_pair_duplicate_invariant':True,'public_yaw_rotation_equivariant':True,'sequence_wait_isolation':True,'unobserved_zero_and_reference_carry':True,'not_AP':True}

def main():
 out=CYCLE/'robust_history';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();panels=[];seq=[]
 update('robust_history21',{'status':'running_fitcal_then_frozen_formal'})
 try:
  contract=load(CYCLE/'contract.json');c=load(aux.verified(contract['shared_contract']));atomic(out/'semantic_checks.json',synthetic());selected=None
  for role in ('fit24','calibration8','val21','mechanism_holdout14'):
   m=core.load_npz(aux.verified(c['inputs'][role+'_material']));ledger=core.load_jsonl(aux.verified(c['inputs']['fit_conditions' if role in ('fit24','calibration8') else 'val_conditions' if role=='val21' else 'train_conditions']));rows=core.conditions_for(m,ledger)
   obs=[core.frame_pairs(m,i)[0] for i in range(len(rows))];d=causal(rows,obs);assert np.array_equal(d,causal(rows,obs))
   basepath=aux.verified(c['inputs']['fit_q005_frame']) if role=='fit24' else ROOT/'artifacts/research_autopilot/status_audit_20261029_strict60_q005_task_refit_portability_v1/producer_v1/calibration8_frame_and_residual.npz' if role=='calibration8' else aux.verified(c['inputs'][role+'_gate_material'])
   base=core.load_npz(basepath)['q005_delta' if role in ('fit24','calibration8') else 'q005_base_delta'];atomic(out/(role+'_input_identity.json'),{'shared_material':c['inputs'][role+'_material'],'baseline_delta':ev(basepath),'observed_frames':sum(bool(len(v)) for v in obs),'conditions':len(rows),'annotation_input':False})
   deltas={NAME:d,'K1_exact':base};sub=out/role;sub.mkdir();core.save_npz(sub/'frame_deltas.npz',**deltas)
   for name,delta in deltas.items():
    b,f=core.build_heading(m,delta);scores=science.score(m,f,c);pp,pred=core.fixed.base.evaluate(m,{name:b},{name:scores})
    if role in ('fit24','calibration8'):panels.extend(dict(r,role=role,source_group='all') for r in pp)
    else:
     if name=='K1_exact':
      gold=core.load_npz(AUDIT/'k2_k3_formal_resume_v3'/(role+'_predictions.npz'))
      for k,v in pred.items():
       if k.startswith(name+'__'):assert np.array_equal(v,gold[k]),('current K1 replay',role,k)
     codes,mapping=aux.source.map_provenance(dict(m,dataset_index=np.arange(len(rows))),rows,role,aux.LABELROOT);assert mapping['unmatched']==mapping['ambiguous']==0
     before=aux.PROTOCOLS;aux.PROTOCOLS=('official',)
     try:mp,ms,arrays=aux.evaluate(m,pred,b,codes,role,{'official':m['gt_boxes']},aux.source.common_mask(m['gt_boxes']),name,sub)
     finally:aux.PROTOCOLS=before
     panels.extend(mp);seq.extend(ms)
     if name!='K1_exact':core.save_npz(sub/(name+'_predictions.npz'),**pred);core.save_npz(sub/(name+'_source_outcomes.npz'),**arrays)
    atomic(out/'panels.json',panels);atomic(out/'per_sequence_panels.json',seq)
   if role=='calibration8':
    means={n:statistics.mean(r['ap70'] for r in panels if r['role']==role and r['method']==n) for n in deltas};selected=NAME if means[NAME]>means['K1_exact'] else 'K1_exact'
    atomic(out/'fitcal_freeze.json',{'created_at':now(),'calibration_AP70_means':means,'selected':selected,'raw_hypothesis':NAME,'all_parameters_frozen_before_formal':True,'no_formal_selection':True})
   atomic(out/'status.json',{'status':'completed_role','role':role,'selected':selected,'wall_s':time.monotonic()-start})
  effects={role:{f'AP{t}':statistics.mean(r[f'ap{t}'] for r in panels if r['role']==role and r['source_group']=='all' and r['method']==NAME)-statistics.mean(r[f'ap{t}'] for r in panels if r['role']==role and r['source_group']=='all' and r['method']=='K1_exact') for t in (50,70)} for role in ('val21','mechanism_holdout14')}
  eligible=selected==NAME and all(v['AP70']>=.5 for v in effects.values())
  atomic(out/'preregistered_read.json',{'selected':selected,'raw_hypothesis_effect_pp':effects,'peer12_native_eligible':eligible,'action_next_calendar':'consider native only' if eligible else 'retain negative/mixed evidence; no recursive variants','official_protocol_unchanged':True})
  atomic(out/'terminal.json',{'status':'complete_light_checked','created_at':now(),'contract':ev(CYCLE/'contract.json'),'read':ev(out/'preregistered_read.json'),'panels':ev(out/'panels.json'),'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'GPU_forwards':0,'updates':0,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20},'no_model_wakeup':True});update('robust_history21',{'status':'complete_light_checked','terminal':ev(out/'terminal.json')})
 except Exception as e:
  atomic(out/'terminal.json',{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc(),'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu});update('robust_history21',{'status':'failed_retained','terminal':ev(out/'terminal.json')});raise
if __name__=='__main__':main()
