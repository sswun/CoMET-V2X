"""Native/common source AP and fixed-population auxiliary AP, zero forwards."""
import argparse,os,time,traceback,resource
from collections import defaultdict
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now,sha
from peer10_cycle03_registration_v1_20261101 import CYCLE,update
import run_peer07_native_cvh_source_split_ap_v1_20261031 as base

def waitfor(path,terminal=None):
    while not path.exists():
        if terminal and terminal.exists() and load(terminal).get('status')=='failed_retained':raise RuntimeError(('failed dependency',str(terminal)))
        time.sleep(30)
    return path

def outcome(overlap,codes,group,threshold):
    target=np.ones(len(codes),bool) if group=='all' else codes!=2 if group=='veh_coop' else codes=={'veh':0,'coop':1,'inf':2}[group]
    remaining=np.flatnonzero(target).tolist();ignored=np.flatnonzero(~target)
    tp=np.zeros(len(overlap),np.int8);fp=tp.copy();ig=tp.copy()
    for i,row in enumerate(overlap):
        if remaining and row[remaining].max()>=threshold:remaining.pop(int(np.argmax(row[remaining])));tp[i]=1
        elif len(ignored) and row[ignored].max()>=threshold:ig[i]=1
        else:fp[i]=1
    return tp,fp,ig,int(target.sum())

def check_kernel():
    ov=np.asarray([[.8,.9],[.1,.95],[.1,.1]])
    tp,fp,ig,count=outcome(ov,np.asarray([0,2]),'veh',.7)
    assert tp.tolist()==[1,0,0] and fp.tolist()==[0,0,1] and ig.tolist()==[0,1,0] and count==1
    for group in ('all','veh_coop','inf'):
        assert all(np.array_equal(a,b) for a,b in zip(outcome(ov,np.asarray([0,2]),group,.7),base.outcomes(ov,np.asarray([0,2]),group,.7)))
    return {'target_first_ignore':True,'existing_kernel_three_groups_equal':True,'native_lwh_to_hwl':[5,4,3]}

def compute(name,c):
    producer=AUDIT/name;terminal=waitfor(producer/'terminal.json');t=load(terminal)
    if t['status']!='native_AP_complete_light_checked':raise RuntimeError(('native producer not accepted',name,t['status']))
    out=CYCLE/'native_source_aux'/name
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.monotonic();cpu=time.process_time();spec=c['native_source_aux']
    labeldir=AUDIT/'k5_labels_recovery_v2';lt=load(waitfor(labeldir/'terminal_v2.json'));assert lt['status']=='labels_built_frozen_AP_pending'
    frozen={'producer_terminal':ev(terminal),'records':ev(producer/'records.json'),'panels':ev(producer/'panels.json'),'shards':{str(w):ev(producer/f'wait{w:03d}.npz') for w in (0,100,200,400)},'label_terminal':ev(labeldir/'terminal_v2.json'),'labels':lt['outputs']}
    atomic(out/'input_identity_before_scoring.json',frozen);atomic(out/'kernel_checks.json',check_kernel())
    replacements={}
    for protocol in ('interpolated','no_interpolation'):
        s=lt['outputs'][protocol];assert sha(ROOT/s['path'])==s['sha256']
        replacements[protocol]={r['vehicle_frame']:{v['cooperative_token']:v['box'] for v in r['replacements']} for r in (loadline for loadline in map(__import__('json').loads,(ROOT/s['path']).read_text().splitlines()))}
    records=load(producer/'records.json');expected={r['wait_ms']:r for r in load(producer/'panels.json')};arms=t['arms'];panels=[];seqrows=[];cache={};maxerror=0.;checks=[];original_labels={}
    assert len(records)==13264 and len({(r['wait_ms'],r['global_index']) for r in records})==13264
    for wait in (0,100,200,400):
        stats=defaultdict(lambda:{'tp':[],'fp':[],'scores':[],'ignored':[],'gt':0});seqstats=defaultdict(lambda:{'tp':[],'fp':[],'scores':[],'gt':0})
        with np.load(producer/f'wait{wait:03d}.npz',allow_pickle=False) as shard:
            selected=[r for r in records if r['wait_ms']==wait];assert len(selected)==3316
            for ordinal,r in enumerate(selected):
                frame=str(r['current_vehicle_frame']);f=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label'/(frame+'.json')
                if frame not in cache:cache[frame]=load(f);original_labels[frame]=sha(f)
                objects=cache[frame];idx=r['shard_case'];gt=shard[f'c{idx}_gt'];codes,error=base.source_codes(gt,objects);maxerror=max(maxerror,error)
                gtby={'official':gt[:,:7]}
                for protocol in ('interpolated','no_interpolation'):
                    aux=gt[:,:7].copy();rep=replacements[protocol][frame]
                    for g in np.flatnonzero(codes==2):
                        match=[obj for obj in objects if obj['from_side']=='inf' and np.linalg.norm(gt[g,:2]-np.asarray([obj['3d_location']['x'],obj['3d_location']['y']]))<.001]
                        assert len(match)==1;b=np.asarray(rep[str(match[0]['token'])]);aux[g]=b[[0,1,2,5,4,3,6]]
                    assert np.array_equal(aux[codes!=2],gt[codes!=2,:7]);gtby[protocol]=aux
                for roi in ('native','common'):
                    keepgt=gt[:,7]==0
                    if roi=='common':keepgt &= base.common_mask(gt[:,:7])
                    sides=codes[keepgt];polys={p:base.polygons(b[keepgt]) for p,b in gtby.items()}
                    for arm in arms:
                        boxes=shard[f'c{idx}_{arm}_box3d_lidar'];scores=shard[f'c{idx}_{arm}_scores'];classes=shard[f'c{idx}_{arm}_label_preds'];bounds=spec['native_eval_range']
                        keep=(classes==0)&(boxes[:,0]>=bounds[0])&(boxes[:,0]<=bounds[3])&(boxes[:,1]>=bounds[1])&(boxes[:,1]<=bounds[4])
                        if roi=='common':keep &= base.common_mask(boxes)
                        bb=boxes[keep];ss=scores[keep];order=np.argsort(-ss);ss=ss[order].astype(np.float64);pp=base.polygons(bb[order])
                        for protocol in spec['protocols']:
                            overlap=base.matrix(pp,polys[protocol])
                            for group in spec['groups']:
                                for threshold in (.5,.7):
                                    tp,fp,ig,count=outcome(overlap,sides,group,threshold);key=(protocol,roi,arm,group,threshold);v=stats[key];v['gt']+=count
                                    for field,value in [('tp',tp),('fp',fp),('ignored',ig),('scores',ss)]:v[field].extend(value.tolist())
                                    v=seqstats[(r['sequence'],*key)];v['gt']+=count
                                    for field,value in [('tp',tp),('fp',fp),('scores',ss)]:v[field].extend(value.tolist())
                if ordinal%128==0:atomic(out/'status.json',{'status':'running','wait_ms':wait,'completed_in_wait':ordinal+1,'wall_s':time.monotonic()-start,'zero_GPU_forwards':True})
        arrays={}
        for (protocol,roi,arm,group,threshold),v in stats.items():
            m=base.metric(v['tp'],v['fp'],v['scores'],v['gt']);panels.append(dict(protocol=protocol,roi=roi,arm=arm,source_group=group,threshold=threshold,wait_ms=wait,ignored=sum(v['ignored']),**m))
            if protocol=='official' and roi=='native' and group=='all':checks.append(abs(m['ap']-100*expected[wait]['metrics'][arm]['global_score_sort'][f'ap{round(100*threshold)}'])<1e-8)
            prefix=f'{protocol}_{roi}_{arm}_{group}_{round(100*threshold)}'
            for k in ('tp','fp','ignored','scores'):arrays[prefix+'_'+k]=np.asarray(v[k],np.float64 if k=='scores' else np.int8)
        assert all(checks),('official native AP replay',wait)
        for (sequence,protocol,roi,arm,group,threshold),v in seqstats.items():seqrows.append(dict(sequence=sequence,protocol=protocol,roi=roi,arm=arm,source_group=group,threshold=threshold,wait_ms=wait,**base.metric(v['tp'],v['fp'],v['scores'],v['gt'])))
        np.savez_compressed(out/f'wait{wait:03d}_outcomes.npz',**arrays);atomic(out/'panels.json',panels);atomic(out/'per_sequence_panels.json',seqrows)
    assert len(panels)==4*3*2*len(arms)*5*2
    labelroot=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label';assert all(sha(labelroot/(f+'.json'))==s for f,s in original_labels.items())
    result={'status':'complete_light_checked','created_at':now(),'panels':ev(out/'panels.json'),'per_sequence':ev(out/'per_sequence_panels.json'),'input_identity':ev(out/'input_identity_before_scoring.json'),'checks':{'official_native_AP_replay_all':all(checks),'source_under_1mm':maxerror<.001,'veh_coop_labels_unchanged':True,'original_GT_files_unchanged':True,'auxiliary_fixed_population_and_predictions':True},'actual_cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'GPU_forwards':0,'updates':0},'not_full_K5_release_complete':True,'independent_review_if_top3_pending':True}
    atomic(out/'terminal.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',default='k2_native,k4_eval_public,k4_eval_online_k1,seed2_public/evaluation,seed2_online_k1/evaluation');p.add_argument('--after-unit');a=p.parse_args();c=load(CYCLE/'contract.json')
    if 'embodied-research.slice' not in Path('/proc/self/cgroup').read_text():raise RuntimeError('uncontrolled worker')
    if a.after_unit:
        import subprocess
        while subprocess.check_output(['systemctl','--user','show',a.after_unit,'-p','ActiveState'],text=True).strip() in ('ActiveState=active','ActiveState=activating'):time.sleep(30)
    for name in a.methods.split(','):
        try:compute(name,c)
        except Exception as exc:
            atomic(CYCLE/'native_source_aux'/name/'terminal.json',{'status':'failed_retained','error':repr(exc),'traceback':traceback.format_exc(),'no_model_wakeup':True})
    atomic(CYCLE/('native_source_aux_chain_'+('remote' if 'official_reference' in a.methods else 'local')+'.json'),{'status':'registered_method_chain_terminal','created_at':now(),'methods':a.methods,'no_model_wakeup':True,'full_external_K5_coverage_still_open':True})
if __name__=='__main__':main()
