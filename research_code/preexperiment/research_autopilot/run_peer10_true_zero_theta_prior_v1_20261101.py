"""Exact zero initial theta information control, not a finite diffuse prior."""
import os,time,traceback,resource
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now,sha
from peer10_cycle06_v1_20261101 import CYCLE,update
import peer10_public_pivot_recovery_v2_20261101 as recovery
science=recovery.science;core=science.core;source=science.source

def predict(L,e,A,Q):
    invA=np.linalg.inv(A);L=invA.T@L@invA;e=invA.T@e
    # Woodbury in canonical form remains valid for singular PSD information.
    M=np.linalg.inv(Q)+L
    newL=L-L@np.linalg.solve(M,L);newe=e-L@np.linalg.solve(M,e)
    return (newL+newL.T)/2,newe

def state(m,rows):
    n=len(rows);means=np.zeros((n,3));infos=np.zeros((n,3,3));pivots=np.zeros((n,2));ranks=np.zeros(n,np.int8);counts=np.zeros(n,np.int16);states={};seen=set()
    Q=np.diag([.05**2,.05**2,np.deg2rad(.01)**2])
    for i,row in enumerate(recovery.rows_with_public_pivot(rows)):
        key=(str(m['sequence'][i]),int(m['wait_ms'][i]));stamp=int(m['receiver_timestamp_us'][i]);assert (*key,stamp) not in seen;seen.add((*key,stamp));pose=np.asarray(row['current_ego_pose_lidar_to_world'],np.float64)
        if key in states:
            L,e,oldpose,oldpivot,oldstamp=states[key];assert stamp>oldstamp
            relative=np.linalg.solve(pose,oldpose);R=relative[:2,:2];u=relative[:2,3]
        else:L=np.diag([1.,1.,0.]);e=np.zeros(3);R=np.eye(2);u=np.zeros(2);oldpivot=None
        if row.get('source') is not None:pivot=np.asarray(row['source']['source_to_current_ego'],np.float64)[:2,3]
        else:pivot=R@oldpivot+u if oldpivot is not None else np.zeros(2)
        A=np.eye(3);A[:2,:2]=R
        if oldpivot is not None:A[:2,2]=science.J@(pivot-u-R@oldpivot)
        L,e=predict(L,e,A,Q)
        points,residual,_,_=science.observations(m,i);counts[i]=len(points)
        if len(points):
            H=np.zeros((len(points),2,3));H[:,:,:2]=np.eye(2);H[:,:,2]=(points-pivot)@science.J.T
            L+=np.einsum('nji,njk->ik',H,H)/.6**2;e+=np.einsum('nji,nj->i',H,residual)/.6**2
        L=(L+L.T)/2;rank=np.linalg.matrix_rank(L);ranks[i]=rank
        if rank==3:mean=np.linalg.solve(L,e)
        else:mean=np.r_[np.linalg.solve(L[:2,:2],e[:2]),0.]
        assert np.isfinite(mean).all() and np.linalg.eigvalsh(L).min()>-1e-8
        means[i]=mean;infos[i]=L;pivots[i]=pivot;states[key]=(L.copy(),e.copy(),pose.copy(),pivot.copy(),stamp)
    return {'mean':means,'information':infos,'information_rank':ranks,'pivot':pivots,'protected_pair_count':counts,'used_pair_count':counts.copy()}

def checks():
    rng=np.random.default_rng(6);B=rng.normal(size=(3,3));P=B@B.T+np.eye(3);L=np.linalg.inv(P);e=L@rng.normal(size=3);A=np.eye(3);A[:2,2]=[.4,-.2];Q=np.diag([.0025,.0025,1e-5]);nl,ne=predict(L,e,A,Q);expected=np.linalg.inv(A@P@A.T+Q)
    assert np.allclose(nl,expected,atol=1e-9) and np.allclose(np.linalg.solve(nl,ne),A@np.linalg.solve(L,e),atol=1e-9)
    zl,ze=predict(np.diag([1.,1.,0.]),np.zeros(3),np.eye(3),Q);assert zl[2,2]==0 and np.array_equal(ze,np.zeros(3))
    return {'proper_covariance_equivalence':True,'exact_zero_angle_information_retained_without_observation':True,'no_finite_angle_initialization':True}

def main():
    out=CYCLE/'zero_theta';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time()
    try:
        contract=load(CYCLE/'contract.json');c=load(AUDIT/'contract_v1.json');atomic(out/'kernel_checks.json',checks());panels=[];sequences=[]
        for role in ('fit24','calibration8','mechanism_holdout14','val21'):
            material=c['inputs'][role+'_material'];ledger=c['inputs']['fit_conditions' if role in ('fit24','calibration8') else 'train_conditions' if role=='mechanism_holdout14' else 'val_conditions']
            assert sha(ROOT/material['path'])==material['sha256'] and sha(ROOT/ledger['path'])==ledger['sha256']
            m=core.load_npz(ROOT/material['path']);allrows=core.load_jsonl(ROOT/ledger['path']);rows=core.conditions_for(m,allrows);s=state(m,rows)
            if role=='fit24':
                repeat=state(m,rows);atomic(out/'same_environment_replay.json',{'mean_max_error':float(np.max(np.abs(s['mean']-repeat['mean']))),'information_max_error':float(np.max(np.abs(s['information']-repeat['information'])))});assert np.array_equal(s['mean'],repeat['mean'])
            name='K2_true_zero_theta_information';p,pred,boxes,_=science.evaluate(m,{name:s},c)
            if role in ('fit24','calibration8'):panels.extend(dict(r,role=role,source_group='all') for r in p)
            else:
                codes,mapping=source.map_provenance(m,allrows,role,ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label');assert mapping['unmatched']==mapping['ambiguous']==0;source.METHODS=(name,);source.GROUPS=('all','veh','coop','inf','veh_coop');source.masks=science.extend_masks
                ps,ss,oc,exact=source.evaluate_role(m,pred,boxes,codes,role);assert all(exact.values());panels.extend(ps);sequences.extend(ss);core.save_npz(out/(role+'_outcomes.npz'),**oc);core.save_npz(out/(role+'_predictions.npz'),**pred)
            core.save_npz(out/(role+'_state.npz'),**s);atomic(out/'panels.json',panels);atomic(out/'per_sequence_panels.json',sequences);atomic(out/'status.json',{'status':'running','completed_role':role,'wall_s':time.monotonic()-start,'unobservable_conditions':int((s['information_rank']<3).sum()),'theta_range_rad':[float(s['mean'][:,2].min()),float(s['mean'][:,2].max())]})
        result={'status':'complete_light_checked','created_at':now(),'contract':ev(CYCLE/'contract.json'),'panels':ev(out/'panels.json'),'per_sequence':ev(out/'per_sequence_panels.json'),'no_selection':True,'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'GPU_forwards':0,'updates':0},'not_finite_3deg_prior':True,'independent_review_if_top3_pending':True};atomic(out/'terminal.json',result);update('zero_theta',{'status':'complete_light_checked','terminal':ev(out/'terminal.json')})
    except Exception as e:atomic(out/'terminal.json',{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc()});update('zero_theta',{'status':'failed_retained','terminal':ev(out/'terminal.json')});raise
if __name__=='__main__':main()
