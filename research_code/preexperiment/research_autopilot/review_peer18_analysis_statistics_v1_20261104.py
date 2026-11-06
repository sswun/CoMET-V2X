"""Independent A2 source serialization and A3/A4 point/cluster statistics.

A3/A4 share saved localization errors, selection/reference construction and
estimator outputs; this is NOT independent registration or pair assignment.
"""
import argparse,csv,time,traceback
import numpy as np
from peer18_v1_20261104 import *
OUT=BASE/'analysis_statistics_independent'
def prepare():
    register('analysis_statistics_independent',__file__,__doc__,dict(detector_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<4'),
        sources={n:ev(BASE/n/'terminal.json') for n in ('A2','A3','A4')},
        remaining='A7 actual inference call-path interpretation separate. A2 lacks CIs for some legacy cells; never fabricate uncertainty.')
def a3_vector(seq,t,error):
    g=[]
    for name in np.unique(seq):
        indices=np.flatnonzero(seq==name);indices=indices[np.argsort(t[indices],kind='stable')]
        e=error[indices];center=e-np.average(e,axis=0);use=(np.diff(t[indices])>0)&(np.diff(t[indices])<=350000)
        x,y=center[:-1][use],center[1:][use]
        g.append([len(e),float(np.einsum('ij,ij',e,e)),len(e)*float(np.dot(e.mean(0),e.mean(0))),
            float(np.einsum('ij,ij',x,y)),float(np.einsum('ij,ij',x,x)),float(np.einsum('ij,ij',y,y)),
            int((np.einsum('ij,ij',e,e)>1).sum()),float(np.einsum('ij,ij',y-x,y-x)),len(x)])
    g=np.asarray(g)
    def measure(v):
        return np.array([np.sqrt(v[1]/max(v[0],1)),v[2]/max(v[1],1e-15),
            v[3]/max(np.sqrt(v[4]*v[5]),1e-15),v[6]/max(v[0],1),v[7]/max(2*v[8],1)])
    rng=np.random.default_rng(2026110403);point=measure(g.sum(0));samples=[]
    for _ in range(2000):samples.append(measure(np.sum(g[rng.integers(0,len(g),len(g))],axis=0)))
    return point,np.quantile(samples,[.025,.975],axis=0)
def run():
    try:
        checks=[]
        for row in load(BASE/'A2/existing_factors.json')['tables']:
            p=ROOT/row['source']['path'];assert ev(p)==row['source']
            original=list(csv.DictReader(p.open())) if p.suffix=='.csv' else load(p)
            assert original==row['rows'];checks.append(dict(source=row['source'],rows=len(original),exact=True))
        atomic(OUT/'A2.json',dict(checks=checks,scope='Source SHA and full-row serialization only; legacy producer CI coverage not invented.'))
        names=('radial_RMSE_m','sequence_bias_fraction','demeaned_lag1_correlation','failure_gt1m_fraction','difference_white_variance_m2');checks=[]
        with np.load(BASE/'A3/errors.npz') as z:
            for row in load(BASE/'A3/panels.json'):
                key=row['method']+'__'+row['reference']+'__'+str(row['wait_ms']);e=z[key+'_error']
                p,ci=a3_vector(z[key+'_sequence'],z[key+'_timestamp'],e);assert len(e)==row['available']
                point_error=max(abs(p[j]-row['metrics'][n]['value']) for j,n in enumerate(names))
                CI_error=max(np.max(np.abs(ci[:,j]-row['metrics'][n]['CI95'])) for j,n in enumerate(names))
                assert point_error<1e-10 and CI_error<1e-9,(key,point_error,CI_error)
                checks.append(dict(key=key,point_max_error=float(point_error),CI_max_error=float(CI_error)))
        atomic(OUT/'A3.json',checks);checks=[];rng=np.random.default_rng(2026110403)
        names=('fraction_gt0p5m','fraction_gt1m','RMSE_m')
        with np.load(BASE/'A4/errors.npz') as z:
            for row in load(BASE/'A4/panels.json'):
                key=str(row['wait_ms'])+'__'+row['subset']+'__'+row['method'];e=z[key+'_error'];seq=z[key+'_sequence'];groups=[e[seq==s] for s in np.unique(seq)]
                def stat(a):return np.array([np.count_nonzero(a>.5)/len(a),np.count_nonzero(a>1)/len(a),np.sqrt(np.dot(a,a)/len(a))])
                p=stat(e);samples=[]
                for _ in range(2000):samples.append(stat(np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])))
                ci=np.quantile(samples,[.025,.975],axis=0)
                point_error=max(abs(p[j]-row['metrics'][n]['value']) for j,n in enumerate(names));CI_error=max(np.max(np.abs(ci[:,j]-row['metrics'][n]['CI95'])) for j,n in enumerate(names))
                assert point_error<1e-10 and CI_error<1e-9,(key,point_error,CI_error)
                for c in row['CDF']:assert abs(np.count_nonzero(e<=c['meters'])/len(e)-c['fraction'])<1e-12
                assert row['matched']==len(e) and row['GT_population']>=len(e)
                checks.append(dict(key=key,point_max_error=float(point_error),CI_max_error=float(CI_error),CDF_exact=True))
        atomic(OUT/'A4.json',checks)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',scope=__doc__,
            original_error_selection_shared=True,independent_registration_or_matching_not_claimed=True,
            remaining='A2 some legacy CI cells absent; A3/A4 input error/reference/pair construction is shared and not independently reconstructed here.'))
        record('analysis_statistics_independent',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
