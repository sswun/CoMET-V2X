"""Actual final late3head cooperation-retention ratios with official/MA5 references.

Same-weight PUBLIC-trained native pose controls, retrained OFFICIAL-vs-PUBLIC,
and final late readout are different estimands. Missing weights/ego/MA5 must
remain explicit; this late result cannot establish a median across backbones.
"""
import argparse,time,traceback
import numpy as np
from peer18_v1_20261104 import *
import run_peer17_final_recipe_v1_20261104 as final
from run_peer16_cached_methods_v1_20261103 import materials,observation
from run_peer12_cached_ceiling_v1_20261101 import reference_deltas
import bootstrap_peer15_paper_v1_20261103 as boot
import review_traf_v2xseq_adapted_fixed60_raw_20261006 as ind
OUT=BASE/'A1'
def prepare():
    register('A1',__file__,__doc__,dict(detector_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<80 compact predictions/draws'),
        baseline=ev(final.OUT/'author/identity.json'),ego=ev(P16/'E5_single_car_table_v2/panels.json'),
        ratio='(method_AP-ego_AP)/(reference_AP-ego_AP); both OFFICIAL and centredMA5; report poles/negative values, never clip ratios.',
        remaining='Native OFFICIAL-trained vs PUBLIC-trained retention requires matching ego/MA5 for those actual checkpoints; same-PUBLIC-weight offset ablations cannot substitute. No new training authorized by A1.')
def ego_predictions(m):
    scores=[];ids=[];tp50=[];tp70=[];offset=[0]
    for i in range(len(m['wait_ms'])):
        a,b=map(int,m['condition_offsets'][i:i+2]);ga,gb=map(int,m['gt_offsets'][i:i+2]);candidate=a+np.flatnonzero(m['source'][a:b]==0)
        candidate=candidate[final.old.author.common_mask(m['cv_boxes'][candidate])];candidate=candidate[np.argsort(-m['raw_scores'][candidate],kind='stable')]
        gt=m['gt_boxes'][ga:gb];gt=gt[final.old.author.common_mask(gt)];s=m['raw_scores'][candidate]
        for th,out in ((50,tp50),(70,tp70)):
            ss,t,_,_=ind.match_frame(m['cv_boxes'][candidate],s,np.zeros(len(s)),gt,np.zeros(len(gt)),th/100);assert np.array_equal(ss,s);out.extend(t)
        scores.extend(s);ids.extend(candidate);offset.append(len(scores))
    return {'EGO__score':np.asarray(scores),'EGO__candidate':np.asarray(ids),'EGO__offset':np.asarray(offset),
        'EGO__tp50':np.asarray(tp50),'EGO__tp70':np.asarray(tp70)}
def run():
    starttime=time.monotonic();record('A1',dict(status='running_current_late_retention_and_gap_audit',resource_owner=True))
    try:
        m,rows=materials('author')[0]['val21'];seqs=sorted(set(m['sequence'].astype(str)));counts=boot.counts(seqs)
        ego=ego_predictions(m);np.savez_compressed(OUT/'ego_predictions.npz',**ego)
        official={str(r['vehicle_frame']):r for r in load(ROOT/'artifacts/data/v2x_seq_spd_traf_align_author21_v1/V2X-Seq-SPD/cooperative/data_info.json')}
        refs=reference_deltas(rows,official);panels=[];ratios=[];tau=load(final.OUT/'author/calibration.json')['selected_tau']
        obs=observation(m);n=np.array([len(z) for _,z in obs]);z=np.array([x.mean(0) if len(x) else np.zeros(2) for _,x in obs]);co=final.old.author.kf(m,n,z,.15,.6)[0]
        for seed in final.SEEDS:
            pred=dict(ego);states={'PUBLIC':np.zeros_like(co),'OFFICIAL':refs['OFFICIAL'],'MA5':refs['OFFICIAL_MA5']}
            for name,delta in states.items():
                b,f=final.old.author.heading.build_heading(m,delta);f[:,final.REMOVED]=0
                score=final.old.author.refit.score_with_head(m,f,final.head('author','no_position',seed))
                pp,p=final.old.author.refit.fixed.base.evaluate(final.protected(m,tau),{name:b},{name:score});panels.extend(dict(r,seed=seed) for r in pp);pred.update(p)
            with np.load(final.OUT/'author'/('val21_seed'+str(seed)+'_predictions.npz')) as p:
                for k in p.files:
                    if k.startswith('CoMET_seed'+str(seed)+'__'):pred[k.replace('CoMET_seed'+str(seed),'CoMET',1)]=p[k]
            np.savez_compressed(OUT/('seed'+str(seed)+'_predictions.npz'),**pred)
            for th in (50,70):
                points={};samples={}
                for name in ('EGO','PUBLIC','CoMET','OFFICIAL','MA5'):
                    d=boot.from_pred(m,pred,name,th);points[name]=[];samples[name]=[]
                    for w in (0,100,200,400):
                        x=boot.panel(d,w,seqs);points[name].append(100*boot.voc_ap(x[1],x[0],int(x[3].sum())))
                        samples[name].append(100*boot.weighted_ap_batch(*x,counts,batch_size=8))
                    points[name].append(float(np.mean(points[name])));samples[name].append(np.mean(samples[name],axis=0))
                for method in ('PUBLIC','CoMET'):
                    for reference in ('OFFICIAL','MA5'):
                        for j,w in enumerate((0,100,200,400,'four_wait_mean')):
                            den=points[reference][j]-points['EGO'][j];bn=samples[method][j]-samples['EGO'][j];bd=samples[reference][j]-samples['EGO'][j]
                            valid=np.abs(bd)>1e-12;values=bn[valid]/bd[valid]
                            ratios.append(dict(family='late_final_author',seed=seed,threshold=th,wait_ms=w,method=method,reference=reference,
                                AP={k:points[k][j] for k in points},retention=None if abs(den)<=1e-12 else (points[method][j]-points['EGO'][j])/den,
                                CI95=None if not len(values) else np.quantile(values,[.025,.975]).tolist(),
                                denominator_CI95=np.quantile(bd,[.025,.975]).tolist(),bootstrap_denominator_positive_fraction=float((bd>0).mean()),
                                bootstrap_valid_fraction=float(valid.mean()),estimand='Ratio of pooledAP gains, four_wait_mean uses meanAP before division;3head seeds are not3detector seeds.'))
                atomic(OUT/'retention.json',ratios);atomic(OUT/'reference_panels.json',panels)
        missing={name:dict(status='actual_checkpoint_identity_or_MA5_not_joined',
            required='Match PUBLIC/OFFICIAL training weight, own ego checkpoint, commonROI/currentGT and real MA5 input; do not replace by strict-PUBLIC-weight offset arm.') for name in ('FCooper','ViT_seed2023','ViT_seed2024','ViT_seed2025','Where2comm','TraF_R_adapted','FFNet','CoAlign')}
        atomic(OUT/'coverage_gaps.json',dict(native=missing,source_ego=ev(P16/'E5_single_car_table_v2/panels.json'),
            factors=ev(BASE/'A2/existing_factors.json'),D1_median_claim_allowed=False,
            planned_next_calendar='06h inspect exact existing native ego/ceiling weights and schedule only missing inference, no new training.'))
        atomic(OUT/'terminal.json',dict(created_at=now(),status='partial_explicit_pending',
            actual_late_retention_complete=True,native_retention_incomplete=True,independent_pending=True,
            results=ev(OUT/'retention.json'),gaps=ev(OUT/'coverage_gaps.json'),wall_s=time.monotonic()-starttime,GPU_forwards=0))
        record('A1',dict(status='partial_explicit_pending',terminal=ev(OUT/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
