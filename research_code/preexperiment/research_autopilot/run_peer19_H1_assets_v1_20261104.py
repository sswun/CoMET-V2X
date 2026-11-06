"""R2 H1 assets; new roots, frozen cal choice, reused independent kernels.

Temporal controls/A1/A3/A4/A5/source AP use the ACTUAL selected H1 geometry,
heads, q and prefix, not relabeled N0. All selection stays on fit/cal.
"""
import argparse,inspect,time,traceback
from types import SimpleNamespace
import numpy as np
from peer19_v1_20261104 import *
import run_peer19_recipe_v3_20261104 as recipe
import run_peer18_analyses_v1_20261104 as analysis
REC=BASE/'recipe_v3_interface_recovery_v2'
OUT=BASE/'H1_assets'
JOBS=('temporal','A1','A3','A4','A5','sources')

def prepare():
    for job in JOBS:
        register('H1_asset_'+job,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<120 compact outputs shared inputs',wall_h='bounded32h'),
            job=job,recipe=ev(BASE/'recipe_global_choice_interface_recovery_v2/contract.json'),
            old_analysis=ev(analysis.__file__),role='val21 for diagnostics; val21+hold14 temporal',
            interpretation='MA5 annotation-assisted and ICP selected proxy, not physical GT; A5 no numerical main-text bound. Temporal cal-only grids reused; negative/empty strata retained.')
    register('H1_assets_independent',__file__,'Independent polygons/TP/AP for H1 temporal/source/A1; independently reimplemented A3/A4 cluster statistics and A5 point equations. Scope per-asset, references/score training shared.',dict(GPU_forwards=0,CPU_threads=1,replicates=2000,disk_MiB='<50 checks/draws',wall_h='bounded34h'),producer_jobs=list(JOBS))

def context():
    wait(BASE/'recipe_global_choice_interface_recovery_v2/terminal.json',30)
    registry=load(REC/'method_registry.json');kind=registry['selected_kind'];cal=load(REC/'author/calibration.json')
    recipe.OUT=REC
    return kind,cal['q'],cal['choices'][kind]['tau'],{s:recipe.head('author',kind,s) for s in recipe.SEEDS}

def obs(m):
    from run_peer16_E1_timing_audit_v1_20261103 import match
    result=[]
    for lo,hi in zip(m['condition_offsets'][:-1],m['condition_offsets'][1:]):
        lo,hi=int(lo),int(hi);ri=lo+np.flatnonzero(m['source'][lo:hi]==1);ei=lo+np.flatnonzero(m['source'][lo:hi]==0)
        ii,jj=match(m['cv_boxes'][ri],m['cv_boxes'][ei],2.5)
        result.append((ri[ii],m['cv_boxes'][ei[jj],:2].astype(float)-m['cv_boxes'][ri[ii],:2].astype(float)))
    return result

def cases(m):
    kind,q,tau,heads=context();d=recipe.hard_state(m,q)['delta'];b=recipe.geometry(m,d)[0];result={}
    for s in recipe.SEEDS:
        with np.load(REC/'author'/('val21_'+kind+'_seed'+str(s)+'_predictions.npz')) as z:p={k:z[k] for k in z.files}
        result['H1_'+kind+'_seed'+str(s)]=(b,p)
    return result

def temporal(dest,kind,q,tau,heads):
    import run_peer16_cached_methods_v1_20261103 as baseline
    data,_,_=recipe.materials('author');m,_=data['calibration8'];oo=obs(m);cal=[];chosen={};panels=[]
    for method,grid in (('EMA',(.1,.25,.5,.75,1.)),('window',(3,5,10,20))):
        for p in grid:
            d=baseline.temporal(m,oo,method,q,p);b,f=recipe.geometry(m,d)
            for s,h in heads.items():
                pp,_=recipe.evaluate(m,b,f,h,method+'_seed'+str(s),tau);cal.extend(dict(r,filter=method,param=p,seed=s) for r in pp)
        chosen[method]=max(grid,key=lambda p:(float(np.mean([r['ap70'] for r in cal if r['filter']==method and r['param']==p])),-p))
    atomic(dest/'calibration_panels.json',cal);atomic(dest/'calibration.json',dict(selected=chosen,q=q,tau=tau,kind=kind,cal_only=True))
    for role in ('val21','mechanism_holdout14'):
        m,_=data[role];oo=obs(m)
        for method in ('single_mean','single_median','hold_last','EMA','window','KF'):
            d=baseline.temporal(m,oo,method,q,chosen.get(method));b,f=recipe.geometry(m,d);np.savez_compressed(dest/(role+'_'+method+'_state.npz'),delta=d)
            for s,h in heads.items():
                label=method+'_seed'+str(s);pp,p=recipe.evaluate(m,b,f,h,label,tau);panels.extend(dict(r,role=role,seed=s,filter=method) for r in pp)
                np.savez_compressed(dest/(role+'_'+label+'_predictions.npz'),**p)
            atomic(dest/'panels.json',panels)

def A1(dest,kind,q,tau,heads):
    import run_peer18_A1_v1_20261104 as engine
    code=inspect.getsource(engine.run)
    code=code.replace("tau=load(final.OUT/'author/calibration.json')['selected_tau']",'tau=TAU')
    code=code.replace('co=final.old.author.kf(m,n,z,.15,.6)[0]','co=final.old.author.kf(m,n,z,Q,.6)[0]')
    code=code.replace("final.head('author','no_position',seed)",'HEADS[seed]')
    code=code.replace("final.OUT/'author'/('val21_seed'+str(seed)+'_predictions.npz')","REC/'author'/('val21_'+KIND+'_seed'+str(seed)+'_predictions.npz')")
    code=code.replace("'CoMET_seed'+str(seed)","'H1_'+KIND+'_seed'+str(seed)")
    ns=dict(engine.__dict__,OUT=dest,Q=q,TAU=tau,HEADS=heads,REC=REC,KIND=kind,observation=obs,record=lambda *a:None)
    exec(compile(code,__file__+'::A1_H1_actual_predictions','exec'),ns);ns['run']()

def A3(dest,kind,q,tau,heads):
    m,rows=recipe.materials('author')[0]['val21'];state=recipe.hard_state(m,q);ma,icp,mask,identity=analysis.references(rows);panels=[];arrays={}
    # Same informative-message population for both single and tracked rows.
    # Full population explicitly includes zero fallback in single-message arm.
    for population,available in [('anchor_messages',state['anchors']>0),('all_messages',np.ones(len(rows),bool))]:
        for method,value in [('single_message_H1_anchor_mean',state['observation']),('CoMET_H1',state['delta'])]:
            name=method+'_'+population
            for ref,target,select in [('MA5',ma,np.ones(len(rows),bool)),('ICP_quality_subset',icp,mask)]:
                for w in (0,100,200,400):
                    pop=(m['wait_ms']==w)&select;use=pop&available
                    if not use.any():continue
                    seq=m['sequence'][use].astype(str);ts=m['receiver_timestamp_us'][use];err=value[use]-target[use];key=name+'__'+ref+'__'+str(w)
                    arrays[key+'_sequence']=seq;arrays[key+'_timestamp']=ts;arrays[key+'_error']=err
                    panels.append(dict(method=name,reference=ref,wait_ms=w,available=int(use.sum()),reference_population=int(pop.sum()),availability=float(use.sum()/pop.sum()),metrics=analysis.stats(seq,ts,err),same_population_single_and_H1=True))
    atomic(dest/'panels.json',panels);np.savez_compressed(dest/'errors.npz',**arrays);atomic(dest/'identity.json',identity)

def A4(dest,kind,q,tau,heads):
    code=inspect.getsource(analysis.A4).replace('from run_peer16_cached_methods_v1_20261103 import materials,observation','from run_peer16_cached_methods_v1_20261103 import materials\n    observation=H1_OBS')
    code=code.replace('kf(m,n,z,.15,.6)','kf(m,n,z,Q,.6)')
    ns=dict(analysis.__dict__,H1_OBS=obs,Q=q);exec(compile(code,__file__+'::A4_H1_all_ego','exec'),ns);ns['A4'](dest)

def A5(dest,kind,q,tau,heads):
    import run_peer18_A5_v1_20261104 as engine
    ns=dict(engine.__dict__,OUT=dest,observation=obs,Q=q,record=lambda *a:None)
    code=inspect.getsource(engine.point).replace('Q=.15**2','Q=SELECTED_Q**2');ns['SELECTED_Q']=q
    exec(compile(code,__file__+'::selected_q_point','exec'),ns)
    code=inspect.getsource(engine.run).replace('kf(m,n,z,.15,.6)','kf(m,n,z,Q,.6)')
    exec(compile(code,__file__+'::A5_H1_actual_state','exec'),ns);ns['run']()

def sources(dest,kind,q,tau,heads):
    import run_peer16_E8_source_contribution_v1_20261103 as engine
    code=inspect.getsource(engine.run);lines=code.splitlines();ids=[i for i,l in enumerate(lines) if l.strip().startswith('cases=')];assert len(ids)==1
    lines[ids[0]]='        cases=H1_CASES(m)';code='\n'.join(lines).replace("AP=100*value['ap']","AP=value['ap']")
    ns=dict(engine.__dict__,OUT=dest,H1_CASES=cases,record=lambda *a:None);exec(compile(code,__file__+'::sources_actual_H1','exec'),ns);ns['run']()

def run(job):
    name='H1_asset_'+job;dest=OUT/job;dest.mkdir(parents=True,exist_ok=False)
    try:
        kind,q,tau,heads=context();wait(REC/'author/primary_terminal.json',30);record(name,dict(status='running_H1_actual_paper_asset',resource_owner=True))
        globals()[job](dest,kind,q,tau,heads)
        # Preserve producer-specific partial receipts (e.g. A1 native coverage)
        # instead of silently changing their meaning to a global completion.
        if (dest/'terminal.json').exists():(dest/'producer_scoped_terminal.json').symlink_to(dest/'terminal.json') if False else atomic(dest/'producer_scoped_terminal.json',load(dest/'terminal.json'))
        atomic(dest/'terminal.json',dict(status='complete_light_checked',created_at=now(),job=job,kind=kind,q=q,tau=tau,independent_pending=True,original_release_closed=False))
        record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(dest/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--job',choices=JOBS);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.job)
