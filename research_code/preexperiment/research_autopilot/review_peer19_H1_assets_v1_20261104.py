"""Per-asset independent H1 review; no whole-paper/independent-training claim."""
import argparse,inspect,time,traceback
import numpy as np
from peer19_v1_20261104 import *
import run_peer19_H1_assets_v1_20261104 as prod
from review_peer16_E4_E7_AP_v1_20261103 import check,geometry
from review_peer19_recipe_v3_20261104 import contrast
OUT=BASE/'H1_assets_independent'

def run():
    try:
        kind,q,tau,heads=prod.context()
        for job in prod.JOBS:wait(prod.OUT/job/'terminal.json',33)
        record('H1_assets_independent',dict(status='running_scoped_independent_H1_assets',resource_owner=True))
        data,_,_=prod.recipe.materials('author');ap=[];boots=[];src=prod.OUT/'temporal';gold=load(src/'panels.json')
        for role in ('val21','mechanism_holdout14'):
            m,_=data[role]
            for method in ('single_mean','single_median','hold_last','EMA','window','KF'):
                with np.load(src/(role+'_'+method+'_state.npz')) as z:b=geometry(m,z['delta'])
                for seed in prod.recipe.SEEDS:
                    label=method+'_seed'+str(seed)
                    with np.load(src/(role+'_'+label+'_predictions.npz')) as p:
                        rr=check(m,b,p,label,[r for r in gold if r['role']==role]);assert all(abs(r['error'+str(t)])<1e-8 and r['TP_mismatch'+str(t)]==0 for r in rr for t in (50,70));ap.extend(dict(r,role=role,seed=seed) for r in rr)
                        if method!='KF':
                            with np.load(src/(role+'_KF_seed'+str(seed)+'_predictions.npz')) as k:
                                contrast(m,k,'KF_seed'+str(seed),p,label,role+'_KF_minus_'+label,dict(role=role,seed=seed),OUT,boots)
                    atomic(OUT/'temporal_AP.json',ap);atomic(OUT/'temporal_bootstrap.json',boots)
        import review_peer18_analysis_statistics_v1_20261104 as stats
        ns=dict(stats.__dict__,BASE=prod.OUT,OUT=OUT/'A3_A4',record=lambda *a:None,P18=P18)
        ns['OUT'].mkdir(parents=True,exist_ok=True)
        code=inspect.getsource(stats.a3_vector).replace("int((np.einsum('ij,ij',e,e)>1).sum())","int((np.sum(e*e,axis=1)>1).sum())")
        exec(compile(code,__file__+'::independent_A3_failure_counts','exec'),ns)
        code=inspect.getsource(stats.run).replace("BASE/'A2/existing_factors.json'","P18/'A2/existing_factors.json'")
        exec(compile(code,__file__+'::independent_A3_A4_statistics','exec'),ns);ns['run']();assert load(ns['OUT']/'terminal.json')['status']=='complete_independently_reviewed'
        import review_peer18_A1_A5_A7_v1_20261104 as prior
        for job in ('A1','A5'):
            dest=OUT/job;dest.mkdir(parents=True,exist_ok=False)
            code=inspect.getsource(getattr(prior,job))
            if job=='A1':
                code=code.replace('materials,observation','materials\n    observation=H1_OBS').replace('kf(m,n,z,.15,.6)','kf(m,n,z,Q,.6)')
                code=code.replace("final.OUT/'author/panels.json'","REC/'author/panels.json'")
                code=code.replace("r['seed']==seed]","r['seed']==seed and r['kind']==KIND]") if False else code.replace("if r['role']=='val21' and r['seed']==seed]","if r['role']=='val21' and r['seed']==seed and r['kind']==KIND]")
            else:code=code.replace('Q=.15**2','Q=SELECTED_Q**2')
            ns=dict(prior.__dict__,BASE=prod.OUT,H1_OBS=prod.obs,Q=q,SELECTED_Q=q,REC=prod.REC,KIND=kind)
            exec(compile(code,__file__+'::independent_'+job,'exec'),ns);scope=ns[job](dest)
            atomic(dest/'terminal.json',dict(status='complete_independently_reviewed',created_at=now(),**scope))
        import review_peer16_E8_source_v1_20261103 as source
        code=inspect.getsource(source.run);lines=code.splitlines();hit=[i for i,l in enumerate(lines) if l.strip().startswith('cases=')];assert len(hit)==1
        lines[hit[0]]='        cases=H1_CASES(m)';code='\n'.join(lines)
        ns=dict(source.__dict__,OUT=OUT/'sources',PROD=prod.OUT/'sources',H1_CASES=prod.cases,record=lambda *a:None)
        ns['OUT'].mkdir(parents=True,exist_ok=False);exec(compile(code,__file__+'::independent_sources','exec'),ns);ns['run']()
        assert load(ns['OUT']/'terminal.json')['status']=='complete_independently_reviewed'
        atomic(OUT/'terminal.json',dict(status='complete_independently_reviewed',created_at=now(),scope=__doc__,
            temporal_independent_polygon_AP_statistics=True,A3_A4_independent_cluster_CI=True,A1_CI_shared_kernel=True,A5_independent_points_only=True,
            sources_independent=True,reference_and_pair_construction_shared=True))
        record('H1_assets_independent',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('H1_assets_independent',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':run()
