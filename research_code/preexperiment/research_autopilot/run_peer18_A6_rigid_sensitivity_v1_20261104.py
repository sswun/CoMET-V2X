"""A6 label-assisted fixed-score rigid-offset dose response, not deployment.

Use accepted author OFFICIAL-aligned boxes/current final protected prefix and
head2023 scores. Freeze scores, change only remote xy by 0:.25:1.5m along four
predeclared cardinal directions. Report every direction and their AP mean;
no best direction/dose selection, new head, detector forward or fitting.
Independent polygon AP and separate sequence-bootstrap arithmetic, shared NMS.
"""
import argparse,time,traceback
import numpy as np
from peer18_v1_20261104 import *
import run_peer17_final_recipe_v1_20261104 as final
from run_peer16_cached_methods_v1_20261103 import materials
from run_peer12_cached_ceiling_v1_20261101 import reference_deltas
from review_peer16_E4_E7_AP_v1_20261103 import check
from review_peer18_main_module_statistics_v1_20261104 import pack
import review_peer18_final_native_statistics_v1_20261104 as independent
OUT=BASE/'A6_rigid_sensitivity_v1'

def specs():
    return [('dose0',0.,'none',np.zeros(2))]+[(f'dose{k:02d}_{name}',k/4,name,k/4*np.asarray(v))
        for k in range(1,7) for name,v in [('xplus',(1.,0.)),('xminus',(-1.,0.)),('yplus',(0.,1.)),('yminus',(0.,-1.))]]
def prepare():
    register('A6_rigid_sensitivity_v1',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,conditions=25,
        replicates=2000,wall_h='bounded24h',disk_MiB='<180 compact scores/TP; shared GT/geometry and reconstructible offsets'),
        baseline=ev(final.OUT/'author/identity.json'),accepted_reference=ev(BASE/'A1/terminal.json'),
        seed=2023,doses=[0,.25,.5,.75,1,1.25,1.5],directions=['+x','-x','+y','-y'],
        selection='None. Four-direction mean and all direction rows, not max. Fixed scores; NMS geometry may change.',
        reuse='Shared accepted author material/GT; no pointcloud or training-tensor duplication. Formal compact needed for independent recomputation.')
def run():
    begin=time.monotonic();record('A6_rigid_sensitivity_v1',dict(status='running_frozen_score_dose_response',resource_owner=True))
    panels=[];reviews=[];statistics=[]
    try:
        m,rows=materials('author')[0]['val21']
        official={str(r['vehicle_frame']):r for r in load(ROOT/'artifacts/data/v2x_seq_spd_traf_align_author21_v1/V2X-Seq-SPD/cooperative/data_info.json')}
        delta=reference_deltas(rows,official)['OFFICIAL'];boxes,features=final.old.author.heading.build_heading(m,delta);features[:,final.REMOVED]=0
        scores=final.old.author.refit.score_with_head(m,features,final.head('author','no_position',2023))
        tau=load(final.OUT/'author/calibration.json')['selected_tau'];mm=final.protected(m,tau)
        seqs=sorted(set(m['sequence'].astype(str)));draw=np.random.Generator(np.random.PCG64(20261103)).integers(0,len(seqs),(2000,len(seqs)))
        counts=np.asarray([np.bincount(row,minlength=len(seqs)) for row in draw],np.int16)
        reference={};allcurves=[]
        for label,dose,direction,shift in specs():
            changed=boxes.copy();changed[m['source']==1,:2]+=shift
            pp,pred=final.old.author.refit.fixed.base.evaluate(mm,{label:changed},{label:scores})
            np.savez_compressed(OUT/(label+'_predictions.npz'),**pred)
            rr=check(mm,changed,pred,label,pp)
            assert all(abs(r['error'+str(th)])<1e-8 and r['TP_mismatch'+str(th)]==0 for r in rr for th in (50,70)),rr
            reviews.extend(dict(r,dose=dose,direction=direction) for r in rr)
            for r in pp:panels.append(dict(r,dose=dose,direction=direction))
            for th in (50,70):
                d=pack(mm,pred,label,th);dist={};point=[]
                for w in (0,100,200,400):
                    x=independent.panel(d,w,seqs);dist[w]=100*independent.empirical(*x,counts)
                    pt=100*float(independent.empirical(*x,np.ones((1,len(seqs)),np.int16))[0]);point.append(pt)
                    gold=next(r['ap'+str(th)] for r in pp if r['wait_ms']==w);assert abs(pt-gold)<1e-8,(label,th,w,pt,gold)
                dist['four_wait_mean']=np.mean(list(dist.values()),axis=0)
                if dose==0:reference[th]=dict(point=point+[float(np.mean(point))],dist=dist)
                np.savez_compressed(OUT/(label+'_AP'+str(th)+'_draws.npz'),**{'wait'+str(k):v for k,v in dist.items()})
                for j,w in enumerate((0,100,200,400,'four_wait_mean')):
                    pt=(point+[float(np.mean(point))])[j];dd=dist[w]-reference[th]['dist'][w]
                    statistics.append(dict(method=label,dose=dose,direction=direction,threshold=th,wait_ms=w,
                        AP=pt,AP_CI95=np.percentile(dist[w],[2.5,97.5]).tolist(),
                        delta_from_dose0_pp=pt-reference[th]['point'][j],delta_CI95=np.percentile(dd,[2.5,97.5]).tolist()))
            atomic(OUT/'panels.json',panels);atomic(OUT/'independent_AP.json',reviews);atomic(OUT/'statistics.json',statistics)
            atomic(OUT/'progress.json',dict(created_at=now(),completed=len(panels)//4,total=25,dose=dose,direction=direction))
            assert time.monotonic()-begin<24*3600
        for dose in (0.,.25,.5,.75,1.,1.25,1.5):
            chosen=[r for r in specs() if r[1]==dose]
            for th in (50,70):
                for w in (0,100,200,400,'four_wait_mean'):
                    values=[r for r in statistics if r['dose']==dose and r['threshold']==th and r['wait_ms']==w]
                    draws=[]
                    for label,_,_,_ in chosen:
                        with np.load(OUT/(label+'_AP'+str(th)+'_draws.npz')) as z:draws.append(z['wait'+str(w)])
                    avg=np.mean(draws,axis=0);difference=avg-reference[th]['dist'][w]
                    allcurves.append(dict(dose=dose,threshold=th,wait_ms=w,mean_direction_AP=float(np.mean([r['AP'] for r in values])),
                        AP_CI95=np.percentile(avg,[2.5,97.5]).tolist(),delta_from_dose0_pp=float(np.mean([r['delta_from_dose0_pp'] for r in values])),
                        delta_CI95=np.percentile(difference,[2.5,97.5]).tolist(),directions=len(chosen)))
        atomic(OUT/'direction_mean_curves.json',allcurves)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',panels=ev(OUT/'panels.json'),
            curves=ev(OUT/'direction_mean_curves.json'),independent_polygon_AP=True,independent_statistics=True,
            shared_NMS_and_reference=True,diagnostic_label_assisted_not_deployable=True,GPU_forwards=0,updates=0,wall_s=time.monotonic()-begin))
        record('A6_rigid_sensitivity_v1',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record('A6_rigid_sensitivity_v1',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
