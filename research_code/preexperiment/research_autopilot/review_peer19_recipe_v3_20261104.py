"""Independent polygon/AP, H1 recurrence and whole-sequence2000CI review.

Shared input boxes, trained scores and NMS/candidate decisions. Primary H1
measurements and KF states are separately reconstructed; all saved predictions
are independently polygon-matched. Statistics use the separately implemented
empirical-AP kernel, not production AP/bootstrap. Not independent head training.
"""
import argparse,time,traceback
import numpy as np
from scipy.optimize import linear_sum_assignment
from peer19_v1_20261104 import *
import run_peer19_recipe_v3_20261104 as prod
from review_peer16_E4_E7_AP_v1_20261103 import check,geometry
from review_peer18_main_module_statistics_v1_20261104 import pack
import review_peer18_final_native_statistics_v1_20261104 as stats
OUT=BASE/'recipe_independent'

def primary_state(m,q):
    states={};out=[];counts=[];means=[]
    for a0,b0,s,w in zip(m['condition_offsets'][:-1],m['condition_offsets'][1:],m['sequence'],m['wait_ms']):
        a,b=int(a0),int(b0);remote=a+np.where(m['source'][a:b]==1)[0];ego=a+np.where(m['source'][a:b]==0)[0]
        n=0;y=np.zeros(2)
        if len(remote) and len(ego):
            distances=np.sqrt(np.sum((m['cv_boxes'][remote,None,:2]-m['cv_boxes'][None,ego,:2])**2,axis=2))
            ii,jj=linear_sum_assignment(distances);valid=(distances[ii,jj]<=2.5)&(np.argmin(distances,axis=1)[ii]==jj)&(np.argmin(distances,axis=0)[jj]==ii)
            ii,jj=ii[valid],jj[valid];n=len(ii)
            if n:y=np.mean(m['cv_boxes'][ego[jj],:2].astype(np.float64)-m['cv_boxes'][remote[ii],:2],axis=0)
        key=str(s),int(w);x,var=states.get(key,(np.zeros(2),1.));prior=var+q*q
        if n:
            k=prior/(prior+.36/n);x=x+k*(y-x);var=prior*(1-k)
        else:var=prior
        states[key]=(x.copy(),var);out.append(x.copy());counts.append(n);means.append(y)
    return np.asarray(out),np.asarray(counts),np.asarray(means)

def contrast(m,p,name,q,baseline,label,identity,dest,done):
    seqs=sorted(set(m['sequence'].astype(str)));draw=np.random.Generator(np.random.PCG64(20261103)).integers(0,len(seqs),(2000,len(seqs)))
    weights=np.asarray([np.bincount(row,minlength=len(seqs)) for row in draw],np.int16)
    for th in (50,70):
        a=pack(m,p,name,th);b=pack(m,q,baseline,th);diff=[];point=[];ps=[]
        for w in (0,100,200,400):
            x,y=stats.panel(a,w,seqs),stats.panel(b,w,seqs);assert np.array_equal(x[3],y[3])
            diff.append(100*(stats.empirical(*x,weights)-stats.empirical(*y,weights)))
            one=np.ones((1,len(seqs)),np.int16);point.append(float(100*(stats.empirical(*x,one)[0]-stats.empirical(*y,one)[0])))
            eye=np.eye(len(seqs),dtype=np.int16);ps.append(100*(stats.empirical(*x,eye)-stats.empirical(*y,eye)))
        np.savez_compressed(dest/(label+'_AP'+str(th)+'.npz'),**{'wait'+str(w):d for w,d in zip((0,100,200,400),diff)},four_wait_mean=np.mean(diff,axis=0))
        for w,pt,d,ss in zip((0,100,200,400,'four_wait_mean'),point+[float(np.mean(point))],diff+[np.mean(diff,axis=0)],ps+[np.mean(ps,axis=0)]):
            ci=np.percentile(d,[2.5,97.5]);done.append(dict(comparison=label,threshold=th,wait_ms=w,delta_pp=pt,CI95_low=float(ci[0]),CI95_high=float(ci[1]),
                positive_sequences=int((ss>1e-12).sum()),negative_sequences=int((ss< -1e-12).sum()),zero_sequences=int((abs(ss)<=1e-12).sum()),identity=identity))

def prepare(domain):
    register('recipe_independent_'+domain,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        replicates=2000,disk_MiB='<60 contrast draws/metrics, shared predictions/GT',wall_h='bounded30h'),
        producer=ev(BASE/('recipe_'+domain)/'contract.json'),kernel=ev(stats.__file__),domain=domain)

def run(domain):
    name='recipe_independent_'+domain;src=prod.OUT/domain;dest=OUT/domain;dest.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    record(name,dict(status='waiting_immutable_recipe_domain',resource_owner=False,owner=None))
    try:
        wait(src/'terminal.json',30);record(name,dict(status='running_independent_geometry_AP_statistics',resource_owner=True))
        data,_,oldq=prod.materials(domain);cal=load(src/'calibration.json');allchecks=[];boots=[]
        def review_file(m,delta,path,label,gold,**tags):
            boxes=geometry(m,delta)
            with np.load(path) as p:rr=check(m,boxes,p,label,gold)
            assert all(abs(r['error'+str(t)])<1e-8 and r['TP_mismatch'+str(t)]==0 for r in rr for t in (50,70)),(path,rr)
            allchecks.extend(dict(r,**tags) for r in rr);atomic(dest/'AP.json',allchecks)
        for role in ('fit24','calibration8','val21','mechanism_holdout14'):
            m,_=data[role];ind,n,z=primary_state(m,cal['q'])
            with np.load(src/(role+'_state.npz')) as state:
                error=float(np.max(abs(ind-state['delta'])));assert error<1e-10 and np.array_equal(n,state['anchors']) and np.array_equal(z,state['observation']),(domain,role,error)
            atomic(dest/(role+'_recurrence.json'),dict(status='passed',max_state_error_m=error,independent_H1_matching_and_KF=True))
        m,_=data['calibration8'];qgold=load(src/'q_calibration_panels.json')
        for q in prod.Q:
            dd=primary_state(m,q)[0]
            for s in prod.SEEDS:
                label='q'+str(q)+'_seed'+str(s)
                review_file(m,dd,src/(label+'_cal_predictions.npz'),label,qgold,phase='q_calibration',q=q,seed=s)
        independent_q=max(prod.Q,key=lambda q:(float(np.mean([r['AP70'] for r in allchecks if r.get('phase')=='q_calibration' and r.get('q')==q])),-q))
        assert independent_q==cal['q']
        delta=primary_state(m,cal['q'])[0];gold=load(src/'calibration_panels.json')
        for kind in ('frozen','refit'):
            for s in prod.SEEDS:
                path=src/(kind+'_cal_seed'+str(s)+'_predictions.npz')
                for tau in prod.prior.GRID:
                    label=kind+'_tau'+str(round(tau*100))+'_seed'+str(s)
                    review_file(m,delta,path,label,gold,phase='tau_calibration',kind=kind,seed=s,tau=tau)
            chosen=max(prod.prior.GRID,key=lambda tau:(float(np.mean([r['AP70'] for r in allchecks if r.get('phase')=='tau_calibration' and r.get('kind')==kind and r.get('tau')==tau])),-tau))
            assert chosen==cal['choices'][kind]['tau']
        for role in ('val21','mechanism_holdout14'):
            m,_=data[role];delta=primary_state(m,cal['q'])[0];gold=[r for r in load(src/'panels.json') if r['role']==role]
            for kind in ('frozen','refit'):
                for s in prod.SEEDS:
                    label='H1_'+kind+'_seed'+str(s);path=src/(role+'_'+kind+'_seed'+str(s)+'_predictions.npz')
                    review_file(m,delta,path,label,gold,phase='primary',role=role,kind=kind,seed=s)
                    with np.load(path) as p,np.load(prod.prior.OUT/domain/(role+'_seed'+str(s)+'_predictions.npz')) as q:
                        contrast(m,p,label,q,'CoMET_seed'+str(s),role+'_'+label+'_minus_N0',dict(domain=domain,role=role,kind=kind,seed=s),dest,boots)
                    atomic(dest/'bootstrap.json',boots)
        agold=load(src/'ablation_panels.json')
        for role in prod.ROLES:
            m,_=data[role];gold=[r for r in agold if r['role']==role]
            for mode in ('N0',*prod.MODES):
                for s in prod.SEEDS:
                    state=src/(role+'_'+mode+('_seed'+str(s) if mode=='N2' else '')+'_diag_state.npz')
                    with np.load(state) as z:delta=z['delta'].copy()
                    label=mode+'_seed'+str(s);path=src/(role+'_'+mode+'_seed'+str(s)+'_diag_predictions.npz')
                    review_file(m,delta,path,label,gold,phase='declared_ablation',role=role,variant=mode,seed=s)
                    if mode!='N0':
                        with np.load(path) as p,np.load(src/(role+'_N0_seed'+str(s)+'_diag_predictions.npz')) as q:
                            contrast(m,p,label,q,'N0_seed'+str(s),role+'_'+label+'_minus_N0',dict(domain=domain,role=role,variant=mode,seed=s),dest,boots)
                        atomic(dest/'bootstrap.json',boots)
                    atomic(dest/'progress.json',dict(created_at=now(),phase='ablation_independent',role=role,variant=mode,seed=s,AP_panels=len(allchecks),CI_panels=len(boots)))
        m,_=data['val21'];gold=load(src/'sensitivity_panels.json')
        for spath in sorted(src.glob('*_sensitivity_state.npz')):
            label=spath.name.removesuffix('_sensitivity_state.npz')
            with np.load(spath) as z:delta=z['delta'].copy()
            for kind in ('frozen','refit'):
                for s in prod.SEEDS:
                    tag=label+'_'+kind+'_seed'+str(s)
                    review_file(m,delta,src/(tag+'_sensitivity_predictions.npz'),tag,gold,phase='sensitivity',kind=kind,seed=s)
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',AP=ev(dest/'AP.json'),bootstrap=ev(dest/'bootstrap.json'),
            AP_panels=len(allchecks),CI_panels=len(boots),independent_H1_recurrence=True,shared_scores_NMS=True,independent_training=False,wall_s=time.monotonic()-begin))
        record(name,dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--domain',choices=prod.DOMAINS,required=True)
    a=p.parse_args();globals()[a.mode](a.domain)
