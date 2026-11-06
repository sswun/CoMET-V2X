"""Same K1 cached pipeline, annotation-assisted offset references; zero CUDA."""
import os,time,traceback,resource,statistics
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import run_peer10_se2_expanded_anchor_v1_20261101 as science
import run_peer10_fixed_population_aux_v1_20261101 as aux
from run_peer11_native_probes_v1_20261101 import upper_inputs
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle18_v1_20261101 import CYCLE,update
core=science.core

def reference_deltas(rows,official):
 upper=upper_inputs(rows,official,np);groups={}
 for r in rows:
  key=(str(r['sequence']),int(r['receiver_timestamp_us']));d=official[str(r['current_vehicle_frame'])]['system_error_offset'];v=np.asarray([d['delta_x'],d['delta_y']],np.float64)
  if key in groups:assert np.array_equal(groups[key],v)
  groups[key]=v
 causal={}
 for seq in sorted({k[0] for k in groups}):
  times=sorted(t for s,t in groups if s==seq);values=np.stack([groups[seq,t] for t in times]);cumulative=np.cumsum(values,axis=0)/np.arange(1,len(times)+1)[:,None]
  for i,t in enumerate(times):causal['OFFICIAL_LAG1',seq,t]=values[i-1] if i else np.zeros(2);causal['OFFICIAL_CUMMEAN',seq,t]=cumulative[i]
 names=('OFFICIAL_SEQMEAN','OFFICIAL_MA5','OFFICIAL_LAG1','OFFICIAL_CUMMEAN')
 result={n:np.stack([(upper if n in names[:2] else causal)[n,str(r['sequence']),int(r['receiver_timestamp_us'])] for r in rows]) for n in names}
 result['OFFICIAL']=np.stack([groups[str(r['sequence']),int(r['receiver_timestamp_us'])] for r in rows]);return result

def synthetic_checks():
 rows=[{'sequence':'s','receiver_timestamp_us':i,'current_vehicle_frame':str(i)} for i in (2,0,1,2)]
 official={str(i):{'system_error_offset':{'delta_x':i,'delta_y':-i}} for i in range(3)}
 d=reference_deltas(rows,official)
 assert np.array_equal(d['OFFICIAL_LAG1'],[[1,-1],[0,0],[0,0],[1,-1]])
 assert np.array_equal(d['OFFICIAL_CUMMEAN'],[[1,-1],[0,0],[.5,-.5],[1,-1]])
 assert np.array_equal(d['OFFICIAL_SEQMEAN'],np.tile([1,-1],(4,1)))
 assert np.array_equal(d['OFFICIAL_MA5'],d['OFFICIAL_SEQMEAN'])
 return {'status':'passed','deduplicate_waits':True,'sorted_real_frames':True,'lag1_first_zero':True,'cummean_includes_current_label':True,'shared_upper_constructor':True,'not_AP':True}

def main():
 out=CYCLE/'ceiling';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();panels=[];seq=[];update('ceiling',{'status':'running_cached_formal_reference'})
 try:
  contract=load(CYCLE/'contract.json');c=load(aux.verified(contract['shared_contract']));atomic(out/'semantic_checks.json',synthetic_checks())
  path=ROOT/'artifacts/data/v2x_seq_spd_traf_align_author21_v1/V2X-Seq-SPD/cooperative/data_info.json';official={str(r['vehicle_frame']):r for r in load(path)}
  atomic(out/'identity.json',{'contract':ev(CYCLE/'contract.json'),'official_offsets':ev(path),'head':c['inputs']['matched_q005_head'],'shared_constructor':ev(ROOT/'preexperiment/research_autopilot/run_peer11_native_probes_v1_20261101.py'),'GT_in_deployable_inference':False,'these_references_use_annotations':True})
  for role in ('val21','mechanism_holdout14'):
   sub=out/role;sub.mkdir();m=core.load_npz(aux.verified(c['inputs'][role+'_material']));ledger=core.load_jsonl(aux.verified(c['inputs']['val_conditions' if role=='val21' else 'train_conditions']));rows=core.conditions_for(m,ledger)
   delta=reference_deltas(rows,official);delta['PUBLIC']=np.zeros((len(rows),2));delta['K1_exact']=core.load_npz(aux.verified(c['inputs'][role+'_gate_material']))['q005_base_delta']
   core.save_npz(sub/'frame_deltas.npz',**delta)
   codes,mapping=aux.source.map_provenance(dict(m,dataset_index=np.arange(len(rows))),rows,role,aux.LABELROOT);assert mapping['unmatched']==mapping['ambiguous']==0;atomic(sub/'population_identity.json',mapping)
   # Score each method separately to bound resident memory. The K1 arm is a
   # same-environment replay against the saved current formal predictions.
   gold=core.load_npz(AUDIT/'k2_k3_formal_resume_v3'/(role+'_predictions.npz'))
   for name,d in delta.items():
    b,f=core.build_heading(m,d);s=science.score(m,f,c);pp,pred=core.fixed.base.evaluate(m,{name:b},{name:s})
    if name=='K1_exact':
     for k,v in pred.items():assert np.array_equal(v,gold[k]),('K1 replay drift',role,k)
    # Original official ROI and fixed readout. Target-first source ignore;
    # official TP/AP replayed by the common scorer on every method.
    before=aux.PROTOCOLS;aux.PROTOCOLS=('official',)
    try:mp,ms,arrays=aux.evaluate(m,pred,b,codes,role,{'official':m['gt_boxes']},aux.source.common_mask(m['gt_boxes']),name,sub)
    finally:aux.PROTOCOLS=before
    core.save_npz(sub/(name+'_predictions.npz'),**pred);core.save_npz(sub/(name+'_source_outcomes.npz'),**arrays)
    panels.extend(mp);seq.extend(ms);atomic(out/'panels.json',panels);atomic(out/'per_sequence_panels.json',seq);atomic(out/'status.json',{'status':'scoring_annotation_reference','role':role,'method':name,'completed_panels':len(panels),'wall_s':time.monotonic()-start})
   del gold,m
  reads={}
  for role in ('val21','mechanism_holdout14'):
   means={n:statistics.mean(r['ap70'] for r in panels if r['role']==role and r['source_group']=='all' and r['method']==n) for n in delta}
   slow=max(means['OFFICIAL_SEQMEAN'],means['OFFICIAL_MA5']);gap=slow-means['K1_exact'];full=means['OFFICIAL']-means['K1_exact']
   reads[role]={'AP70_means':means,'G_slow':gap,'G_full':full,'slow_recovery_ratio':(means['K1_exact']-means['PUBLIC'])/(slow-means['PUBLIC']) if slow!=means['PUBLIC'] else None,'full_recovery_ratio':(means['K1_exact']-means['PUBLIC'])/(means['OFFICIAL']-means['PUBLIC']) if means['OFFICIAL']!=means['PUBLIC'] else None,'annotation_coupled_remainder_fraction':(full-gap)/full if full else None}
  gaps=[v['G_slow'] for v in reads.values()];action='no_new_estimator_branch' if max(gaps)<=.5 else 'eligible_bounded_fitcal_only_max2_variants_next_calendar' if min(gaps)>=1 else 'report_only'
  atomic(out/'preregistered_reads.json',{'roles':reads,'action':action,'no_automatic_method_selection':True,'all_annotation_assisted_not_deployable':True})
  terminal={'status':'complete_light_checked','created_at':now(),'contract':ev(CYCLE/'contract.json'),'panels':ev(out/'panels.json'),'reads':ev(out/'preregistered_reads.json'),'K1_saved_prediction_exact_replay':True,'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'GPU_forwards':0,'updates':0,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20},'independent_review_pending_if_paper':True,'no_model_wakeup':True};atomic(out/'terminal.json',terminal);update('ceiling',{'status':'complete_light_checked','terminal':ev(out/'terminal.json')})
 except Exception as e:
  atomic(out/'terminal.json',{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc(),'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu},'partial_panels':len(panels)});update('ceiling',{'status':'failed_retained','terminal':ev(out/'terminal.json')});raise
if __name__=='__main__':main()
