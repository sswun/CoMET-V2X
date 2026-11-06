"""Independent full cached-HWL replay, shared layout/geometry loaders only.

Primary labels are independently joined and dimensions checked for every GT.
IoU, target-first ignore and AP do not call the producer's scoring routines.
No duplicate prediction/GT/outcome cache is produced.
"""
import os,json,time,traceback,resource,math
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
import run_peer10_fixed_population_aux_v1_20261101 as layout
from run_peer10_aux_remaining_v2_20261101 import modules
from peer10_cycle12_v1_20261101 import CYCLE,update
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now,sha

GROUPS=('all','veh','coop','inf','veh_coop');PROTOCOLS=('official','interpolated','no_interpolation')

def polygons(boxes):
    result=[]
    for b in np.asarray(boxes,np.float64):
        c,s=math.cos(float(b[6])),math.sin(float(b[6]));l,w=float(b[5])/2,float(b[4])/2
        result.append(Polygon([(float(b[0])+c*x-s*y,float(b[1])+s*x+c*y) for x,y in ((l,-w),(l,w),(-l,w),(-l,-w))]))
    return result

def overlap_matrix(left,right):
    result=np.zeros((len(left),len(right)),np.float32)
    if not len(left) or not len(right):return result
    bounds=np.asarray([p.bounds for p in right])
    for i,p in enumerate(left):
        x0,y0,x1,y1=p.bounds
        for j in np.flatnonzero((bounds[:,0]<=x1)&(bounds[:,2]>=x0)&(bounds[:,1]<=y1)&(bounds[:,3]>=y0)):
            q=right[j];union=p.union(q).area;result[i,j]=p.intersection(q).area/union if union else 0.
    return result

def flags(iou,codes,group,threshold):
    selected=np.ones(len(codes),bool) if group=='all' else codes!=2 if group=='veh_coop' else codes=={'veh':0,'coop':1,'inf':2}[group]
    available=set(np.flatnonzero(selected).tolist());ignored=np.flatnonzero(~selected)
    result=np.zeros((3,len(iou)),np.int8)
    for i,row in enumerate(iou):
        choices=sorted(available);values=row[choices]
        if choices and values.max()>=threshold:
            available.remove(choices[int(values.argmax())]);result[0,i]=1
        elif len(ignored) and np.any(row[ignored]>=threshold):result[2,i]=1
        else:result[1,i]=1
    return result,int(selected.sum())

def metric(tp,fp,scores,gt):
    if not gt:return 0.
    order=np.argsort(-np.asarray(scores,np.float64));tp=np.cumsum(np.asarray(tp,np.int64)[order]);fp=np.cumsum(np.asarray(fp,np.int64)[order]);rec=tp/gt;pre=tp/np.maximum(tp+fp,1)
    r=np.r_[0.,rec,1.];p=np.r_[0.,pre,0.];p=np.maximum.accumulate(p[::-1])[::-1];idx=np.flatnonzero(r[1:]!=r[:-1])
    return float(100*np.sum((r[idx+1]-r[idx])*p[idx+1]))

def primary(m,rows,replacements):
    codes=np.empty(len(m['gt_boxes']),np.int8);targets={p:m['gt_boxes'].copy() for p in PROTOCOLS};cache={};hashes={};checks=0
    for i,row in enumerate(rows):
        frame=str(row['current_vehicle_frame']);path=layout.LABELROOT/(frame+'.json')
        if frame not in cache:
            objs=load(path);centers=np.asarray([[o['3d_location']['x'],o['3d_location']['y']] for o in objs],np.float64)
            cache[frame]=(objs,cKDTree(centers) if len(objs) else None);hashes[frame]=sha(path)
        objects,tree=cache[frame];a,b=map(int,m['gt_offsets'][i:i+2]);gt=m['gt_boxes'][a:b]
        if not len(gt):continue
        assert tree is not None
        hits=tree.query_ball_point(gt[:,:2].astype(np.float64),r=.001)
        for j,match in enumerate(hits):
            assert len(match)==1,('primary unique source',frame,j,match)
            o=objects[match[0]];side={'veh':0,'coop':1,'inf':2}[o['from_side']];codes[a+j]=side
            dims=o['3d_dimensions'];assert np.allclose(gt[j,3:6],[dims['h'],dims['w'],dims['l']],rtol=0,atol=1e-5),('primary HWL',frame,j)
            checks+=1
            if side==2:
                for p in PROTOCOLS[1:]:
                    box=replacements[p][frame][str(o['token'])]
                    targets[p][a+j]=[box[0],box[1],box[2],box[5],box[4],box[3],box[6]]
    assert all(np.array_equal(targets[p][codes!=2],m['gt_boxes'][codes!=2]) for p in PROTOCOLS)
    return codes,targets,hashes,checks

def main():
    out=CYCLE/'independent_cached_hwl';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();reports=[];panels=[]
    try:
        spec=load(CYCLE/'contract.json');terminal=load(layout.verified(spec['corrected_auxiliary_terminal']));assert terminal['status']=='complete_light_checked'
        original=load(layout.verified(load(layout.verified(spec['corrected_auxiliary_contract']))['original_contract']))
        labelterm=load(layout.verified(original['label_terminal']));replacements={p:{r['vehicle_frame']:{v['cooperative_token']:v['box'] for v in r['replacements']} for r in map(json.loads,layout.verified(labelterm['outputs'][p]).read_text().splitlines())} for p in PROTOCOLS[1:]}
        expected=load(layout.verified(terminal['panels']));assert len(expected)==3600
        expected={(r['family'],r['role'],r['method'],r['protocol'],r['source_group'],r['wait_ms']):r for r in expected}
        configs={'current':load(layout.verified(original['shared_main_contract'])),**{k:load(layout.verified(v)) for k,v in original['baseline_contracts'].items()}}
        update('independent_aux',{'status':'running_full_independent_replay','contract':ev(CYCLE/'contract.json')})
        for family,loader in [('current',layout.current),('geometry',layout.geometry),('modules',modules),('ego',layout.ego),('official_late',layout.official_late)]:
            for role in ('mechanism_holdout14','val21'):
                m,rows,pred,boxes,mask=loader(role,configs[family]);codes,targets,labelhash,dimchecks=primary(m,rows,replacements)
                for method,bb in boxes.items():
                    ref=AUDIT/'cycle09_v1/fixed_population_hwl_v3'/family/role/(method+'_outcomes.npz');identity=ev(ref)
                    with np.load(ref) as gold:
                        stats=defaultdict(lambda:{'scores':[],'gt':0,'flags':[]});mismatch=0;first_mismatch=[]
                        for i,wait in enumerate(m['wait_ms']):
                            ga,gb=map(int,m['gt_offsets'][i:i+2]);pa,pb=map(int,pred[method+'__offset'][i:i+2]);ids=pred[method+'__candidate'][pa:pb];scores=pred[method+'__score'][pa:pb]
                            cp=polygons(bb[ids]);side=codes[ga:gb][mask[ga:gb]]
                            for p in PROTOCOLS:
                                iou=overlap_matrix(cp,polygons(targets[p][ga:gb][mask[ga:gb]]))
                                for group in GROUPS:
                                    for t in (50,70):
                                        value,count=flags(iou,side,group,t/100)
                                        for k,field in enumerate(('tp','fp','ignore')):
                                            n=int(np.count_nonzero(value[k]!=gold[f'{p}__{group}__{field}{t}'][pa:pb]));mismatch+=n
                                            if n and len(first_mismatch)<16:first_mismatch.append({'condition':i,'protocol':p,'group':group,'threshold':t,'field':field,'differences':n})
                                        st=stats[p,group,int(wait),t];st['scores'].append(scores);st['flags'].append(value);st['gt']+=count
                            if i%256==0:atomic(out/'status.json',{'status':'independent_scoring','family':family,'role':role,'method':method,'conditions':i+1,'total':len(rows),'completed_panels':len(panels),'wall_s':time.monotonic()-start})
                    for (p,group,wait,t),v in sorted(stats.items()):
                        sc=np.concatenate(v['scores']);f=np.concatenate(v['flags'],axis=1);keep=f[2]==0;ap=metric(f[0,keep],f[1,keep],sc[keep],v['gt']);old=expected[family,role,method,p,group,wait]
                        panels.append({'family':family,'role':role,'method':method,'protocol':p,'source_group':group,'wait_ms':wait,'threshold':t,'AP':ap,'expected_AP':old[f'ap{t}'],'absolute_AP_diff':abs(ap-old[f'ap{t}']),'count_match':v['gt']==old['gt'] and int(f[0].sum())==old[f'tp{t}'] and int(f[1].sum())==old[f'fp{t}'] and int(f[2].sum())==old[f'ignored{t}']})
                    reports.append({'family':family,'role':role,'method':method,'outcome_reference':identity,'flag_mismatches':mismatch,'first_mismatches':first_mismatch,'primary_HWL_checks':dimchecks})
                    atomic(out/'method_checks.json',reports);atomic(out/'panel_checks.json',panels)
                assert all(sha(layout.LABELROOT/(f+'.json'))==s for f,s in labelhash.items()),'primary labels changed'
        checks={'all_3600_panels_both_thresholds':len(panels)==7200,'all_flags_equal':all(r['flag_mismatches']==0 for r in reports),'all_counts_equal':all(r['count_match'] for r in panels),'all_AP_equal_1e8':max(r['absolute_AP_diff'] for r in panels)<1e-8}
        status='complete_independently_reviewed' if all(checks.values()) else 'failed_retained'
        atomic(out/'terminal.json',{'status':status,'created_at':now(),'contract':ev(CYCLE/'contract.json'),'checks':checks,'panel_checks':ev(out/'panel_checks.json'),'method_checks':ev(out/'method_checks.json'),'shared_only_layout_and_geometry_not_scoring':True,'original_labels_unchanged':True,'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'GPU_forwards':0,'updates':0},'no_model_wakeup':True,'does_not_close_whole_K5_release':True});update('independent_aux',{'status':status,'terminal':ev(out/'terminal.json')})
    except Exception as e:
        atomic(out/'terminal.json',{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc(),'completed_metric_checks':len(panels),'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'no_model_wakeup':True});update('independent_aux',{'status':'failed_retained','terminal':ev(out/'terminal.json')});raise
if __name__=='__main__':main()
