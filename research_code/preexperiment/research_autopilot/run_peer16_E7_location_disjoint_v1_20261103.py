"""Supplementary leave-one-intersection-out readout test, official split unchanged."""
import argparse
import time
import traceback
from collections import Counter
import numpy as np
from peer16_v1_20261103 import ROOT,BASE,atomic,ev,load,now,register,record
import run_peer16_E7_readout_v1_20261103 as old
OUT=BASE/'E7_location_disjoint'

def prepare():
    register('E7_location_disjoint',__file__,
        'All six named val intersections also occur in fit/cal, so one head disjoint from all val locations has no named fit population. Execute fixed metadata-only leave-one-intersection-out full39/no-position heads, one held site each, as supplementary leakage diagnostic, never a replacement official main recipe or evaluation-based selection.',
        dict(GPU_forwards=0,CPU_threads=2,updates='6sites×2heads×2000=24000',wall_h='1-6',disk_MiB='<80 predictions/heads; no copied features/GT'),
        original=ev(BASE/'E7_readout_v3/contract.json'),seed=2023,
        fit='Original fit24 minus held val site and unknown location, original fit weights renormalized;2000 batches of same width sampled with fixed seed20261103',
        calibration='Original calibration8 minus same held site and unknown; no val labels in training/calibration',
        evaluation='Original val21 named-site subsets; two frozen head kinds and all sites retained. Unknown20 conditions reported/excluded only in this supplementary diagnostic. Backbone has seen original official training data; not detector-level geography generalization.',
        no_single_global_disjoint_head_possible=True)

def candidate_mask(m,conditions):
    return np.repeat(np.asarray(conditions,bool),np.diff(m['condition_offsets']))

def train_arrays(m,keep):
    return {k:m[k][keep] for k in ('features','utility_y','quality_y','raw_scores','source','protected')}

def subset(m,conditions,b,f):
    ci=np.flatnonzero(conditions);pi=candidate_mask(m,conditions)
    gi=np.repeat(np.asarray(conditions,bool),np.diff(m['gt_offsets']))
    out={k:m[k][pi] for k in ('raw_scores','source','protected')}
    out.update({k:m[k][ci] for k in ('wait_ms','sequence','receiver_timestamp_us','dataset_index')})
    out.update(condition_offsets=np.r_[0,np.cumsum(np.diff(m['condition_offsets'])[ci])],
               gt_offsets=np.r_[0,np.cumsum(np.diff(m['gt_offsets'])[ci])],gt_boxes=m['gt_boxes'][gi])
    return out,b[pi],f[pi]

def run():
    start=time.monotonic();record('E7_location_disjoint',dict(status='running_metadata_only_geography_folds'))
    try:
        data,head,_=old.materials('author')
        info=load(ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/vehicle-side/data_info.json')
        lookup={x['frame_id']:x.get('intersection_loc','') for x in info}
        loc={role:np.asarray([lookup[str(r['current_vehicle_frame'])] for r in rows]) for role,(m,rows) in data.items()}
        sites=sorted(set(loc['val21'])-{''});assert len(sites)==6
        census={r:dict(Counter(x.tolist())) for r,x in loc.items()}
        global_fit=set(loc['fit24'])-set(loc['val21'])-{''}
        assert not global_fit
        atomic(OUT/'location_audit.json',dict(created_at=now(),census_conditions=census,sites=sites,single_global_disjoint_fit_sites=sorted(global_fit),single_global_head_status='no_named_training_population',supplementary_protocol='fixed leave-one-site-out; not new official split'))
        prepared={};geom={}
        for role in ('fit24','calibration8','val21'):
            m,_=data[role];b,f=old.geometry(m);geom[role]=(b,f)
            if role!='val21':prepared[role]=old.targets(m,b,f,.7)
        with np.load(head/'training_schedule.npz') as z:weights=z['weights'].copy();shape=z['batches'].shape
        assert shape[0]==2000
        results=[]
        for site in sites:
            fold=OUT/site;fold.mkdir(exist_ok=False)
            fitkeep=candidate_mask(data['fit24'][0],(loc['fit24']!=site)&(loc['fit24']!=''))
            calkeep=candidate_mask(data['calibration8'][0],(loc['calibration8']!=site)&(loc['calibration8']!=''))
            assert fitkeep.any() and calkeep.any()
            fit=train_arrays(prepared['fit24'],fitkeep);cal=train_arrays(prepared['calibration8'],calkeep)
            w=weights[fitkeep].astype(float);w/=w.sum();rng=np.random.default_rng(20261103)
            schedule=fold/'schedule.npz';np.savez_compressed(schedule,weights=w,batches=rng.choice(len(w),size=shape,p=w).astype(np.int32))
            identity=dict(held_site=site,fit_sites=sorted(set(loc['fit24'])-{site,''}),cal_sites=sorted(set(loc['calibration8'])-{site,''}),fit_candidates=len(w),cal_candidates=int(calkeep.sum()),fit_disjoint=site not in set(loc['fit24'][loc['fit24']!=site]),no_val_target_fit=True)
            atomic(fold/'identity.json',identity)
            vm=data['val21'][0];mm,bb,ff=subset(vm,loc['val21']==site,*geom['val21'])
            for kind in ('full39','no_position'):
                ft={k:v.copy() for k,v in fit.items()};ct={k:v.copy() for k,v in cal.items()};vf=ff.copy()
                if kind=='no_position':ft['features'][:,old.REMOVED]=0;ct['features'][:,old.REMOVED]=0;vf[:,old.REMOVED]=0
                dest=fold/kind;old.matched.train_head(ft,schedule,dest);raw=old.matched.score(ct,dest);old.matched.source_map(ct,raw,dest)
                name=site+'_'+kind;score=old.author.refit.score_with_head(mm,vf,dest)
                full=BASE/'E7_readout_v3'/(kind+'_seed2023');baseline=old.author.refit.score_with_head(mm,vf,full)
                pp,p=old.author.refit.fixed.base.evaluate(mm,{name:bb,name+'_allsite_head':bb},{name:score,name+'_allsite_head':baseline})
                results.extend(dict(x,held_site=site,kind=kind,role='val21_location_subset') for x in pp)
                np.savez_compressed(dest/'val_predictions.npz',**p,condition_sequence=mm['sequence'])
                atomic(OUT/'panels.json',results);atomic(OUT/'status.json',dict(created_at=now(),held_site=site,kind=kind,wall_s=time.monotonic()-start))
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',sites=6,heads=12,updates=24000,GPU_forwards=0,wall_s=time.monotonic()-start,scope='Supplementary location folds only; no single global-disjoint head possible; independent/bootstrap pending'))
        record('E7_location_disjoint',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start));record('E7_location_disjoint',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
