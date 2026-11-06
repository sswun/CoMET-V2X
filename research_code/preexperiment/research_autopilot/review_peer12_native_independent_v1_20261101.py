"""Full seed1 native replay: independent HWL geometry, match, source and AP.

No author box/IoU/AP routines. Reads the sole saved native predictions and
primary label/token files; never performs forwards or duplicates outcomes.
"""
import os,time,traceback,resource
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,sha,now
from peer12_cycle21_v1_20261101 import CYCLE,update
from review_peer10_cached_hwl_independent_v1_20261101 import polygons,overlap_matrix,flags,metric
from audit_v2xseq_strict_common_roi_20261010 import mask as common_mask

def main():
 out=CYCLE/'independent_native';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();reports=[];checks=[];spec=load(CYCLE/'contract.json')['native_review'];panels=[];update('native_review21',{'status':'running_full_independent_native_replay'})
 try:
  lt=load(AUDIT/'k5_labels_recovery_v2/terminal_v2.json');replacements={p:{r['vehicle_frame']:{v['cooperative_token']:v['box'] for v in r['replacements']} for r in map(__import__('json').loads,(ROOT/lt['outputs'][p]['path']).read_text().splitlines())} for p in spec['protocols'][1:]}
  bounds=load(AUDIT/'cycle03_v1/contract.json')['native_source_aux']['native_eval_range'];labelroot=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label';labels={};hashes={}
  for name in spec['producers']:
   producer=AUDIT/name;terminal=load(producer/'terminal.json');assert terminal['status']=='native_AP_complete_light_checked' and terminal['cases']==13264 and terminal['weight_unchanged'] and not terminal['forward_GT_input']
   goldroot=AUDIT/spec['source_root']/name;goldterm=load(goldroot/'terminal.json');assert goldterm['status']=='complete_light_checked';goldpanels=load(goldroot/'panels.json');records=load(producer/'records.json');expected={(r['protocol'],r['roi'],r['arm'],r['source_group'],r['threshold'],r['wait_ms']):r for r in goldpanels};identities={'producer':ev(producer/'terminal.json'),'source':ev(goldroot/'terminal.json'),'records':ev(producer/'records.json'),'shards':{str(w):ev(producer/f'wait{w:03d}.npz') for w in (0,100,200,400)}}
   atomic(out/(name+'_identity.json'),identities)
   for wait in (0,100,200,400):
    selected=[r for r in records if r['wait_ms']==wait];assert len(selected)==3316;stats=defaultdict(lambda:{'flags':[],'scores':[],'gt':0});mismatches=0;details=[];dimchecks=0
    with np.load(producer/f'wait{wait:03d}.npz') as shard,np.load(goldroot/f'wait{wait:03d}_outcomes.npz') as gold:
     offsets=defaultdict(int)
     for ordinal,r in enumerate(selected):
      frame=r['current_vehicle_frame'];path=labelroot/(frame+'.json')
      if frame not in labels:labels[frame]=load(path);hashes[frame]=sha(path)
      objs=labels[frame];gt=shard[f"c{r['shard_case']}_gt"];centers=np.asarray([[o['3d_location']['x'],o['3d_location']['y']] for o in objs]);codes=np.empty(len(gt),np.int8)
      if len(gt):
       hits=cKDTree(centers).query_ball_point(gt[:,:2],r=.001)
       for j,h in enumerate(hits):
        assert len(h)==1;obj=objs[h[0]];codes[j]={'veh':0,'coop':1,'inf':2}[obj['from_side']];dims=obj['3d_dimensions'];assert np.allclose(gt[j,3:6],[dims['h'],dims['w'],dims['l']],rtol=0,atol=1e-5);dimchecks+=1
      targets={'official':gt[:,:7]}
      for p in spec['protocols'][1:]:
       target=gt[:,:7].copy()
       for j in np.flatnonzero(codes==2):target[j]=np.asarray(replacements[p][frame][str(objs[hits[j][0]]['token'])])[[0,1,2,5,4,3,6]]
       assert np.array_equal(target[codes!=2],gt[codes!=2,:7]);targets[p]=target
      for roi in spec['rois']:
       keepgt=gt[:,7]==0
       if roi=='common':keepgt &= common_mask(gt[:,:7])
       targetpoly={p:polygons(v[keepgt]) for p,v in targets.items()};side=codes[keepgt]
       for arm in terminal['arms']:
        idx=r['shard_case'];bb=shard[f'c{idx}_{arm}_box3d_lidar'];ss=shard[f'c{idx}_{arm}_scores'];cl=shard[f'c{idx}_{arm}_label_preds'];keep=(cl==0)&(bb[:,0]>=bounds[0])&(bb[:,0]<=bounds[3])&(bb[:,1]>=bounds[1])&(bb[:,1]<=bounds[4])
        if roi=='common':keep &= common_mask(bb)
        bb=bb[keep];ss=ss[keep];order=np.argsort(-ss);ss=ss[order].astype(np.float64);pp=polygons(bb[order])
        for p in spec['protocols']:
         iou=overlap_matrix(pp,targetpoly[p])
         for group in spec['groups']:
          for t in spec['thresholds']:
           f,count=flags(iou,side,group,t);key=(p,roi,arm,group,t);st=stats[key];st['flags'].append(f);st['scores'].append(ss);st['gt']+=count
           prefix=f'{p}_{roi}_{arm}_{group}_{round(100*t)}';a=offsets[prefix];b=a+len(ss)
           for j,field in enumerate(('tp','fp','ignored')):
            err=int(np.count_nonzero(f[j]!=gold[prefix+'_'+field][a:b]));mismatches+=err
            if err and len(details)<12:details.append({'condition':ordinal,'key':list(key),'field':field,'differences':err})
           assert np.array_equal(ss,gold[prefix+'_scores'][a:b]);offsets[prefix]=b
      if ordinal%128==0:atomic(out/'status.json',{'status':'independent_native_replay','producer':name,'wait_ms':wait,'conditions_in_wait':ordinal+1,'total_in_wait':3316,'completed_metric_checks':len(panels),'wall_s':time.monotonic()-start})
    for key,st in stats.items():
     p,roi,arm,group,t=key;f=np.concatenate(st['flags'],axis=1);ss=np.concatenate(st['scores']);valid=f[2]==0;ap=metric(f[0,valid],f[1,valid],ss[valid],st['gt']);old=expected[(*key,wait)]
     panels.append({'producer':name,'protocol':p,'roi':roi,'arm':arm,'source_group':group,'threshold':t,'wait_ms':wait,'AP':ap,'expected_AP':old['ap'],'absolute_AP_difference':abs(ap-old['ap']),'counts_equal':st['gt']==old['gt'] and int(f[0].sum())==old['tp'] and int(f[1].sum())==old['fp'] and int(f[2].sum())==old['ignored']})
    reports.append({'producer':name,'wait_ms':wait,'primary_HWL_checks':dimchecks,'flag_mismatches':mismatches,'first_differences':details});atomic(out/'panel_checks.json',panels);atomic(out/'flag_checks.json',reports)
  assert all(sha(labelroot/(frame+'.json'))==digest for frame,digest in hashes.items())
  checks={'all1440_metric_panels':len(panels)==1440,'all_flags_equal':all(r['flag_mismatches']==0 for r in reports),'all_counts_equal':all(r['counts_equal'] for r in panels),'all_AP_within1e8':max(r['absolute_AP_difference'] for r in panels)<1e-8,'primary_labels_unchanged':True};status='complete_independently_reviewed' if all(checks.values()) else 'failed_retained'
  atomic(out/'terminal.json',{'status':status,'created_at':now(),'contract':ev(CYCLE/'contract.json'),'checks':checks,'panel_checks':ev(out/'panel_checks.json'),'flag_checks':ev(out/'flag_checks.json'),'independent':'analytic float64 HWL corners, shapely overlap, primary source/token/HWL join, own target-first match+VOC AP; no producer scoring routines','cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'GPU_forwards':0,'updates':0,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20},'numerical_discrepancies_retained_not_method_sign_gate':True,'whole_release_not_closed':True,'no_model_wakeup':True});update('native_review21',{'status':status,'terminal':ev(out/'terminal.json')})
 except Exception as e:
  atomic(out/'terminal.json',{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc(),'completed_metric_checks':len(panels),'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu});update('native_review21',{'status':'failed_retained','terminal':ev(out/'terminal.json')});raise
if __name__=='__main__':main()
