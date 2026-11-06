"""Independent V2 paired all-GT AP; unique saved native shards, zero CUDA."""
import os,time,resource,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from collections import defaultdict
import numpy as np
import yaml
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now
from peer12_cycle00_v1_20261102 import CYCLE

def main():
 out=CYCLE/'v2review00';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();c=load(CYCLE/'contract.json');panels=[]
 try:
  for name in ('review_geometry','v2_config'):assert ev(ROOT/c[name]['path'])==c[name]
  from review_peer10_cached_hwl_independent_v1_20261101 import polygons,overlap_matrix,flags,metric
  h=yaml.safe_load((ROOT/c['v2_config']['path']).read_text());bounds=h['dataset']['eval_range'];producers={n:AUDIT/('cycle15_peer11_v1/v2_'+n) for n in ('public','online')}
  ids={n:load(p/'sampling_identity.json') for n,p in producers.items()};assert ids['public']==ids['online']
  records={n:load(p/'records.json') for n,p in producers.items()};assert records['public']==records['online'] and len(records['public'])==2644
  atomic(out/'paired_identity.json',{'created_at':now(),'same_training_schedule':True,'same_current_condition_records':True,'conditions_each_model':2644,'sampling':{n:ev(p/'sampling_identity.json') for n,p in producers.items()},'records':{n:ev(p/'records.json') for n,p in producers.items()},'evaluation_semantics':'author native HWL boxes, class0 and inclusive native XY ROI; saved GT class0, no auxiliary/ROI changes','review_geometry':c['review_geometry']})
  identity={};gt_count=0
  for train,producer in producers.items():
   t=load(producer/'terminal.json');assert ev(producer/'terminal.json')==c['v2_producers'][train] and t['status']=='complete_light_checked' and t['updates']==2000 and t['microbatches']==8000 and t['no_GT_forward_input']
   assert ev(ROOT/t['weight']['path'])==t['weight'];expected={(r['role'],r['wait_ms']):r for r in load(producer/'panels.json')};shards={}
   for (role,wait),old in expected.items():
    filename=f'{role}_wait{wait:03d}.npz';shards[filename]=ev(producer/filename);selected=[r for r in records[train] if r['role']==role and r['wait_ms']==wait];assert len(selected)==old['cases'];stats=defaultdict(lambda:{'flags':[],'scores':[],'gt':0})
    with np.load(producer/filename) as shard,np.load(producers['public']/filename) as targetref:
     for i,r in enumerate(selected):
      idx=r['shard_case'];gt=shard[f'c{idx}_gt'];assert np.array_equal(gt,targetref[f'c{idx}_gt']);assert np.isfinite(gt).all() and np.all(gt[:,3:6]>0);gt=gt[gt[:,7]==0,:7];gp=polygons(gt);gt_count+=len(gt)
      for arm in ('PUBLIC','ONLINE_K1'):
       bb=shard[f'c{idx}_{arm}_box3d_lidar'];ss=shard[f'c{idx}_{arm}_scores'];cl=shard[f'c{idx}_{arm}_label_preds'];keep=(cl==0)&(bb[:,0]>=bounds[0])&(bb[:,0]<=bounds[3])&(bb[:,1]>=bounds[1])&(bb[:,1]<=bounds[4]);bb=bb[keep];ss=ss[keep];order=np.argsort(-ss);ss=ss[order];iou=overlap_matrix(polygons(bb[order]),gp)
       for threshold in (.5,.7):
        f,count=flags(iou,np.zeros(len(gt),np.int8),'all',threshold);v=stats[arm,threshold];v['flags'].append(f);v['scores'].append(ss);v['gt']+=count
      if i%128==0:atomic(out/'status.json',{'status':'independent_v2_AP','training':train,'role':role,'wait_ms':wait,'conditions':i+1,'total':len(selected),'metric_checks':len(panels),'wall_s':time.monotonic()-start})
    for (arm,threshold),v in stats.items():
     f=np.concatenate(v['flags'],axis=1);ss=np.concatenate(v['scores']);ap=metric(f[0],f[1],ss,v['gt']);gold=100*old['metrics'][arm]['global_score_sort'][f'ap{round(100*threshold)}'];panels.append({'training':train,'role':role,'wait_ms':wait,'input':arm,'threshold':threshold,'AP':ap,'expected_AP':gold,'absolute_difference':abs(ap-gold),'gt':v['gt'],'tp':int(f[0].sum()),'fp':int(f[1].sum())})
    atomic(out/'panel_checks.json',panels)
   replays=load(producer/'replay.json');assert len(replays)==8 and all(r['shape_equal'] for r in replays)
   identity[train]={'terminal':ev(producer/'terminal.json'),'training_terminal':ev(producer/'training_terminal.json'),'environment':ev(producer/'environment.json'),'shards':shards,'replays':replays}
  checks={'all32_panels':len(panels)==32,'AP_within_1e8':max(r['absolute_difference'] for r in panels)<1e-8,'paired_current_GT_equal':True,'paired_training_schedule_equal':True,'actual_weights_verified':True}
  atomic(out/'input_identity.json',identity);atomic(out/'terminal.json',{'status':'complete_independently_reviewed' if all(checks.values()) else 'failed_retained','created_at':now(),'checks':checks,'panels':ev(out/'panel_checks.json'),'identity':ev(out/'input_identity.json'),'max_absolute_AP_difference':max(r['absolute_difference'] for r in panels),'scope':'all saved V2 native predictions and current GT, independent polygons/IoU/matching/VOC AP; not independent primary-label provenance/source audit or independent inference','cost':{'wall_s':time.monotonic()-start,'CPU_s':time.process_time()-cpu,'GPU_forwards':0,'updates':0,'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20},'negative_effect_retained':True,'no_model_wakeup':True})
 except Exception as e:
  atomic(out/'terminal.json',{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc(),'metric_checks':len(panels),'wall_s':time.monotonic()-start,'CPU_s':time.process_time()-cpu,'no_model_wakeup':True});raise
if __name__=='__main__':main()
