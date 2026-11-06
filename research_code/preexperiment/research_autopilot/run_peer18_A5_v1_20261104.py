"""Actual temporal observability statistics and explicitly approximate Riccati check.

MA5 is label-assisted, ICP selected proxy, neither physical ground truth.
q/r are configured filter parameters, not independently estimated true process
and measurement noise. An apparent30% fit cannot alone establish a theorem.
"""
import argparse,time,traceback
import numpy as np
from peer18_v1_20261104 import *
from run_peer16_cached_methods_v1_20261103 import materials,observation
from run_peer18_analyses_v1_20261104 import references
import run_peer17_final_recipe_v1_20261104 as final
OUT=BASE/'A5'
def prepare():
    register('A5',__file__,__doc__,dict(detector_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<5 sequence sufficient stats/metrics'),
        baseline=ev(final.OUT/'author/identity.json'),
        model='Mean-field intermittent-observation scalar Riccati: priorX solves pX²=Q(X+R), P=X-Q; Q=q²/message,R=r²/mean_anchors_per_informative_message. Radial prediction sqrt(2P).',
        ACF='Sequence-demeaned vector ACF at message lags1..100; actual mean time separation; first1/e crossing (right-censored if absent), no future inference input.')
def point(g):
    s=np.sum(g,axis=0);duration=max(s[1],1e-12);rate=s[2]/duration;arrival=s[3]/duration;p=s[3]/max(s[2],1)
    mbar=s[4]/max(s[3],1);R=.36/max(mbar,1e-12);Q=.15**2
    prior=(Q+np.sqrt(Q*Q+4*p*Q*R))/(2*max(p,1e-12));P=prior-Q
    a=s[9:].reshape(100,5);rho=a[:,0]/np.maximum(np.sqrt(a[:,1]*a[:,2]),1e-15);lag=a[:,3]/np.maximum(a[:,4],1)
    valid=a[:,4]>0;cross=np.flatnonzero(valid&(rho<=np.exp(-1)));tau=float(lag[cross[0]]) if len(cross) else float(lag[valid][-1]) if valid.any() else 0.
    return dict(message_rate_Hz=float(rate),informative_arrival_lambda_Hz=float(arrival),
        mean_anchors_given_informative=float(mbar),informative_fraction=float(p),tau_c_seconds=tau,
        tau_c_right_censored=not len(cross),N_eff_tau_times_lambda=float(tau*arrival),
        mean_field_P_per_axis_m2=float(P),predicted_radial_RMSE_m=float(np.sqrt(max(2*P,0))),
        measured_CoMET_to_MA5_RMSE_m=float(np.sqrt(s[5]/max(s[6],1))),
        measured_CoMET_to_ICP_quality_RMSE_m=float(np.sqrt(s[7]/max(s[8],1))),
        ACF=rho.tolist(),lag_seconds=lag.tolist())
def run():
    starttime=time.monotonic();record('A5',dict(status='running_actual_temporal_statistics',resource_owner=True))
    try:
        m,rows=materials('author')[0]['val21'];obs=observation(m);n=np.array([len(z) for _,z in obs]);z=np.array([a.mean(0) if len(a) else np.zeros(2) for _,a in obs])
        d,variance,_,_=final.old.author.kf(m,n,z,.15,.6);ma,icp,mask,identity=references(rows);arrays={};panels=[]
        for wait in (0,100,200,400):
            groups=[];seqs=[]
            for seq in sorted(set(m['sequence'].astype(str))):
                ids=np.flatnonzero((m['sequence']==seq)&(m['wait_ms']==wait));ids=ids[np.argsort(m['receiver_timestamp_us'][ids],kind='stable')]
                t=m['receiver_timestamp_us'][ids].astype(float)/1e6;v=ma[ids]-ma[ids].mean(0);acf=[]
                for lag in range(1,101):
                    x,y=v[:-lag],v[lag:]
                    acf.extend([np.sum(x*y),np.sum(x*x),np.sum(y*y),np.sum(t[lag:]-t[:-lag]),len(x)])
                err=d[ids]-ma[ids];use=mask[ids];pe=d[ids][use]-icp[ids][use]
                groups.append([len(ids),max(float(t[-1]-t[0]),.1),len(ids),int((n[ids]>0).sum()),int(n[ids].sum()),
                    float(np.sum(err*err)),len(err),float(np.sum(pe*pe)),len(pe),*acf]);seqs.append(seq)
            g=np.asarray(groups);result=point(g);rng=np.random.default_rng(2026110405)
            keys=[k for k,v in result.items() if not isinstance(v,(list,bool))];boot=[]
            for _ in range(2000):
                r=point(g[rng.integers(len(g),size=len(g))]);boot.append([r[k] for k in keys])
            ci=np.quantile(boot,[.025,.975],axis=0)
            result.update(wait_ms=wait,CI95={k:ci[:,j].tolist() for j,k in enumerate(keys)},
                relative_RMSE_prediction_error=abs(result['predicted_radial_RMSE_m']-result['measured_CoMET_to_MA5_RMSE_m'])/max(result['measured_CoMET_to_MA5_RMSE_m'],1e-12),
                interpretation='Descriptive mean-field approximation, not validated physical error bound; censored tau reported, no main-text theorem authorized by this numerical check alone.')
            panels.append(result);arrays['wait'+str(wait)+'_sufficient']=g;arrays['wait'+str(wait)+'_sequence']=np.asarray(seqs)
        np.savez_compressed(OUT/'sequence_statistics.npz',**arrays);atomic(OUT/'panels.json',panels);atomic(OUT/'identity.json',identity)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',wall_s=time.monotonic()-starttime,
            scope=__doc__,independent_pending=True,GPU_forwards=0))
        record('A5',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
