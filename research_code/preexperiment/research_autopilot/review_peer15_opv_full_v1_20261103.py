"""Independent full-test raw-output geometry review plus canonical-input SHA check."""
import argparse,time,traceback
from pathlib import Path
import numpy as np
from peer15_v1_20261103 import ROOT,BASE,atomic,ev,load,now,register,record
import review_traf_v2xseq_adapted_fixed60_raw_20261006 as independent
from audit_v2xseq_strict_common_roi_20261010 import mask
OUT=BASE/'opv_full_independent';PRODUCER=BASE/'opv_full_read';INPUT=BASE/'opv_full_test_v2/producer'

def prepare():
    register('opv_full_independent',Path(__file__).resolve(),'Independently rederive corners/IoU/greedy matching/global VOC for all12 full-test panels; exact discrete counts and numerical AP differences retained. Verify every remote-used PCD/YAML against canonical3090 SHA, so staging scope/content cannot silently pass as full.',dict(GPU_forwards=0,CPU_threads=1,wall_h='finite24h wait +1-3h input hashes and independent geometry',disk_MiB='<5 identities/statistics only; no raw data/GT/pred copies'),producer_contract=ev(PRODUCER/'contract.json'),review_kernel=ev(independent.__file__),canonical=ev(BASE/'opv_full_test_v2/canonical_census.json'),scope='Reviewer shares saved detector boxes/noise and frozen population, independently reconstructs output geometry matching/AP and checks input contents; not independent detector or estimator retraining.')

def run():
    start=time.monotonic();cpu=time.process_time();record('opv_full_independent',dict(status='waiting_full_CPU_producer'))
    try:
        while not (PRODUCER/'terminal.json').exists():
            assert time.monotonic()-start<24*3600,'finite24h OPV read wait';time.sleep(30)
        assert load(PRODUCER/'terminal.json')['status']=='complete_light_checked';rows=load(INPUT/'records.json');assert len(rows)==2170
        seen={};hashchecks=0
        for i,r in enumerate(rows):
            for sensor in r['sensors']:
                for field in ('pcd','yaml'):
                    ref=sensor[field];p=ROOT/ref['path']
                    if ref['path'] not in seen:seen[ref['path']]=ev(p)
                    assert seen[ref['path']]==ref,('canonical-vs-actual-input',field,p);hashchecks+=1
            if i%128==0:atomic(OUT/'status.json',dict(created_at=now(),status='canonical_PCD_YAML_SHA',frames=i+1,unique_files=len(seen)))
        assert hashchecks==2*5985
        atomic(OUT/'canonical_input_check.json',dict(created_at=now(),unique_input_files=len(seen),sensor_input_checks=hashchecks,all_actual_remote_SHA_equal_canonical_3090=True,full_census_frames=2170,sensor_frames=5985))
        gold=load(PRODUCER/'panels.json');results=[];selfcheck=independent.self_test();assert selfcheck['status']=='passed'
        for family in ('SPD_structured','iid_02m_02deg'):
            for wait in (0,400):
                path=PRODUCER/(family+'_wait'+str(wait)+'_predictions.npz')
                with np.load(path) as z:
                    for method in ('stale_same_U','CoMET_heading_U','oracle_clean_heading_U'):
                        offset=z[method+'__offset'];boxes=z[method+'__boxes'];scores=z[method+'__scores'];tp={50:[],70:[]};rank=[];gtcount=0;discrete={50:0,70:0}
                        for i,r in enumerate(rows):
                            p=ROOT/r['geometry']['path'];assert ev(p)==r['geometry']
                            with np.load(p) as g:gt=g['gt_hwl'];gt=gt[mask(gt)]
                            lo,hi=map(int,offset[i:i+2]);b=boxes[lo:hi];s=scores[lo:hi]
                            assert len(gt)==z['GT_count'][i];gtcount+=len(gt)
                            for t in (50,70):
                                ss,tt,ff,ng=independent.match_frame(b,s,np.zeros(len(b)),gt,np.zeros(len(gt)),t/100);assert ng==len(gt);tp[t].extend(tt)
                                discrete[t]+=int(np.sum(np.asarray(tt)!=z[method+'__tp'+str(t)][lo:hi]))
                            rank.extend(ss)
                        expected=next(p for p in gold if p['noise']==family and p['wait_ms']==wait and p['method']==method)
                        entry=dict(noise=family,wait_ms=wait,method=method,frames=2170,predictions=ev(path),GT=gtcount,discrete_TP_mismatches=discrete)
                        for t in (50,70):
                            m=independent.voc_ap(tp[t],(1-np.asarray(tp[t])).tolist(),rank,gtcount,True);entry['AP'+str(t)]=m['ap']*100;entry['AP'+str(t)+'_error']=m['ap']*100-expected['AP'+str(t)];entry['TP'+str(t)]=m['tp'];entry['FP'+str(t)]=m['fp']
                        results.append(entry);atomic(OUT/'results.json',results)
        passed=all(all(abs(r['AP'+str(t)+'_error'])<1e-8 and r['discrete_TP_mismatches'][t]==0 for t in (50,70)) for r in results)
        status='complete_independently_reviewed' if passed else 'completed_discrepancy_retained';atomic(OUT/'terminal.json',dict(created_at=now(),status=status,results=ev(OUT/'results.json'),canonical_input_check=ev(OUT/'canonical_input_check.json'),reviewer_selftest=selfcheck,GPU_forwards=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,no_model_wakeup=True));record('opv_full_independent',dict(status=status,terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start));record('opv_full_independent',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();globals()[a.mode]()
