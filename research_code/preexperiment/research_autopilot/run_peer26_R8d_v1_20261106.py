"""CoMET-A preregistered recovery, with real-data H1 bitwise degeneration.

First reconstruct all fifteen real-data state streams and report triggers;
then cal8 injected 3/5m production and independent geometry/AP/2000CI. The
remaining gate cases and descriptive stress follow without parameter tuning.
"""
import argparse,csv,time,traceback
import numpy as np
from peer26_v1_20261106 import *
import peer26_R8_estimators_v1_20261106 as model
from run_peer26_R8b_v1_20261106 import DOMAINS,ROLES,SEEDS,context
from peer20_population_core_v1_20261105 import data,packed,statistic
from review_peer16_E4_E7_AP_v1_20261103 import check,geometry
import run_peer19_recipe_v3_20261104 as recipe
OUT=BASE/'R8d';REVIEW=BASE/'R8d_independent';TABLE=ROOT/'docs/research_v2/R8d'

def prepare():
    for name in ('R8d','R8d_independent','R8d_chain'):
        register(name,__file__,__doc__,dict(CPU_threads=1,GPU_forwards=0,updates=0,replicates=2000,
            wall_h='estimated 12-60 including independent review; before Nov14',disk_MiB='<850 compact state/score/TP/draws; no raw copies'),
            algorithm=ev(model.__file__),parameters=dict(K=5,W=5,radius=10.,bin_width=.5,peak_radius=1.,min_pairs=6,
                min_messages=3,min_magnitude=2.,min_peak_ratio=1.5,reset_variance=.25,gate=2.5),
            engineering_adjustments_used=0,real_data='five domains x cal/val/hold; never-triggered stream H1 bitwise equality',
            acquisition_ties='numpy argsort histogram counts reverse; cells with Chebyshev distance>1 are non-adjacent',
            injection='five remote geometry arrays jointly shifted, GT and ego unchanged; perturbation never enters estimator',
            gate='peer26 two gates exactly; T3 recovery descriptive; original H1 heads/q/tau frozen',
            independent='Separate matching/KF/histogram reconstruction + heading/polygon TP/AP/bootstrap; frozen scores upstream shared')

def real_states():
    rows_out=[]
    for domain in DOMAINS:
        dataset,_,_=data(domain);q,_,_=context(domain);dest=OUT/domain;dest.mkdir(parents=True,exist_ok=True)
        for role in ROLES:
            m,rows=dataset[role];a,info=model.acquisition_state(m,q);b,other=model.acquisition_state(m,q,review=True)
            assert np.array_equal(a,b),'independent acquisition state not exact'
            assert info==other
            with np.load(P20/('R6b_'+domain)/(role+'_H1_state.npz')) as z:base=z['delta'].copy()
            untriggered={(r['sequence'],r['wait_ms']) for r in info['trigger_by_stream'] if r['count']==0}
            mask=np.asarray([(str(s),int(w)) in untriggered for s,w in zip(m['sequence'],m['wait_ms'])])
            assert np.array_equal(a[mask],base[mask]),'untriggered stream must be bitwise H1'
            np.savez_compressed(dest/(role+'_state.npz'),delta=a)
            atomic(dest/(role+'_state_identity.json'),dict(created_at=now(),events=info['events'],trigger_by_stream=info['trigger_by_stream'],
                triggers=info['triggers'],streams=info['streams'],never_locked_streams=info['never_locked_streams'],
                H1=ev(P20/('R6b_'+domain)/(role+'_H1_state.npz')),untriggered_bitwise=True,
                independent_state_exact=True,conditions=len(rows)))
            row=dict(domain=domain,role=role,triggers=info['triggers'],streams=info['streams'],untriggered_bitwise=True)
            rows_out.append(row);atomic(OUT/'real_state_progress.json',rows_out)
    atomic(OUT/'real_state_terminal.json',dict(created_at=now(),status='complete_independently_reviewed',panels=rows_out))

def case(domain,role,kind,amplitude=0.,real=False):
    """One complete suite, producer then independent; fixed methods and heads."""
    name=domain+'_'+role+'_'+('real' if real else kind+'_'+str(amplitude))
    dest=OUT/'cases'/name;rev=REVIEW/'cases'/name;dest.mkdir(parents=True,exist_ok=False);rev.mkdir(parents=True,exist_ok=False)
    dataset,_,_=data(domain);original,rows=dataset[role];m,shift=(original,np.zeros((len(rows),2))) if real else model.inject(original,kind,amplitude)
    q,tau,heads=context(domain);n,z=model.observations(m);dh=model.filter_state(m,n,z,q)[0]
    da,info=model.acquisition_state(m,q);single=z
    states={'PUBLIC':np.zeros_like(dh),'single_mean':single,'H1':dh,'CoMET_A':da}
    np.savez_compressed(dest/'states.npz',**states)
    # Experimental truth stays in the harness artifact, never passed to a method.
    np.savez_compressed(dest/'injection_harness.npz',sequence=m['sequence'],wait_ms=m['wait_ms'],timestamp=m['receiver_timestamp_us'],shift=shift)
    info_small={k:v for k,v in info.items() if k not in ('anchors','ever_locked')};atomic(dest/'acquisition.json',info_small)
    summary=[];results={};source_refs=[]
    for review in (False,True):
        nn,zz=model.observations(m,review);hh=model.filter_state(m,nn,zz,q,review=review)[0]
        aa,ii=model.acquisition_state(m,q,review=review)
        assert np.array_equal(hh,dh) and np.array_equal(aa,da) and ii==info
        for method,d in states.items():
            if review:b=geometry(m,d);f=None
            else:b,f=recipe.geometry(m,d)
            for seed,head in heads.items():
                label=method+'_seed'+str(seed);path=dest/(label+'_predictions.npz')
                if review:
                    with np.load(path) as v:p={k:v[k] for k in v.files}
                    gold=load(dest/(label+'_AP.json'));cc=check(m,b,p,label,gold)
                    assert all(r['TP_mismatch'+str(th)]==0 and abs(r['error'+str(th)])<1e-8 for r in cc for th in (50,70))
                else:
                    gold,p=recipe.evaluate(m,b,f,head,label,tau);atomic(dest/(label+'_AP.json'),gold);np.savez_compressed(path,**p)
                    source_refs.append(dict(method=method,seed=seed,prediction=ev(path),head=ev(head/'direct_u.pth')))
                stats=[];draws={}
                for th in (50,70):
                    rr,dd=statistic(packed(m,p,label,th),np.ones(len(rows),bool),review)
                    stats.extend(dict(r,threshold=th) for r in rr);draws['AP'+str(th)]=dd
                if review:
                    gg=load(dest/(label+'_statistics.json'));assert len(gg)==len(stats)
                    for a,b0 in zip(stats,gg):assert abs(a['AP']-b0['AP'])<1e-8 and np.max(np.abs(np.asarray(a['CI95'])-b0['CI95']))<1e-8
                    with np.load(dest/(label+'_draws.npz')) as v:assert all(np.max(np.abs(v[k]-x))<1e-8 for k,x in draws.items())
                else:
                    atomic(dest/(label+'_statistics.json'),stats);np.savez_compressed(dest/(label+'_draws.npz'),**draws)
                    for th in (50,70):
                        point=next(r['AP'] for r in stats if r['threshold']==th and r['wait_ms']=='four_wait_mean')
                        results[method,seed,th]=(point,draws['AP'+str(th)].mean(0))
                atomic((rev if review else dest)/'progress.json',dict(created_at=now(),method=method,seed=seed,review=review))
    for method in states:
        for th in (50,70):
            point=float(np.mean([results[method,s,th][0] for s in SEEDS]));draw=np.mean([results[method,s,th][1] for s in SEEDS],axis=0)
            base=float(np.mean([results['H1',s,th][0] for s in SEEDS]));bd=np.mean([results['H1',s,th][1] for s in SEEDS],axis=0)
            pub=float(np.mean([results['PUBLIC',s,th][0] for s in SEEDS]));pd=np.mean([results['PUBLIC',s,th][1] for s in SEEDS],axis=0)
            h1_reference=[]
            for seed in SEEDS:
                rr=load(P20/('R6b_'+domain)/(role+'_H1_seed'+str(seed)+'_statistics.json'))
                h1_reference.append(next(r['AP'] for r in rr if r['threshold']==th and r['population']=='all' and r['wait_ms']=='four_wait_mean'))
            summary.append(dict(domain=domain,role=role,kind='real' if real else kind,amplitude=amplitude,method=method,threshold=th,
                AP=point,CI95=np.quantile(draw,[.025,.975]).tolist(),delta_H1=point-base,delta_H1_CI95=np.quantile(draw-bd,[.025,.975]).tolist(),
                recovery_PUBLIC=point-pub,recovery_PUBLIC_CI95=np.quantile(draw-pd,[.025,.975]).tolist(),
                ratio_clean_H1=point/float(np.mean(h1_reference)),replicates=2000,heads=3,waits=4,
                never_locked_stream_fraction=info['never_locked_streams']/info['streams'] if method=='CoMET_A' else None))
    atomic(dest/'summary.json',summary)
    atomic(dest/'identity.json',dict(source_predictions=source_refs,q=q,tau=tau,original_core=ev(P20/('R6b_'+domain)/'identity.json'),
        simultaneous_injection_keys=list(model.BOX_KEYS),GT_unchanged=True,injection_hidden_from_methods=True,
        T4='not performed; no same-projection roadside-origin audit yet',R4_external='not included; reusable injection-aware observation interface not verified'))
    atomic(rev/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',summary=ev(dest/'summary.json'),
        exact_independent_state=True,independent_geometry_TP_AP_CI=True))
    atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',independent=ev(rev/'terminal.json')))
    return summary

def gate():
    rows=[]
    for p in (OUT/'cases').glob('*/summary.json'):rows.extend(load(p))
    checks={}
    for role in ('val21','mechanism_holdout14'):
        dd=[r['delta_H1'] for r in rows if r['kind']=='real' and r['role']==role and r['method']=='CoMET_A' and r['threshold']==70]
        assert len(dd)==5;checks[role+'_clean_mean_abs_le_point05']=abs(float(np.mean(dd)))<=.05
        checks[role+'_no_domain_below_minus_point10']=min(dd)>=-.10
    for domain in ('author','strict60'):
        for amp in (3.,5.):
            r=next(r for r in rows if r['domain']==domain and r['role']=='val21' and r['kind']=='T1' and r['amplitude']==amp and r['method']=='CoMET_A' and r['threshold']==70)
            checks[domain+'_'+str(amp)+'_ratio_ge_point90']=r['ratio_clean_H1']>=.90
            checks[domain+'_'+str(amp)+'_paired_CI_positive']=r['delta_H1_CI95'][0]>0
    return dict(created_at=now(),status='complete_independently_reviewed',accepted_extension=all(checks.values()),checks=checks,panels=rows)

def chain():
    start_time=time.monotonic();dest=BASE/'R8d_chain';completed=[]
    try:
        assert now()<DEADLINE
        real_states();completed.append('real_state_bitwise_independent')
        # Cal injection must precede any formal injection outcome; no adjustment
        # is made automatically and negative outcomes retain the same contract.
        for domain in ('author','strict60'):
            for amp in (3.,5.):case(domain,'calibration8','T1',amp);completed.append([domain,'calibration8',amp])
        atomic(OUT/'cal_injection_terminal.json',dict(created_at=now(),status='complete_independently_reviewed',engineering_adjustments_used=0))
        for role in ('val21','mechanism_holdout14'):
            for domain in DOMAINS:
                info=load(OUT/domain/(role+'_state_identity.json'))
                if info['triggers']==0:
                    # Bitwise identity makes all selected heads/prefix/TP exact;
                    # no repeated forward/matching/bootstrap to prove equality.
                    summaries=[]
                    for th in (50,70):
                        vals=[]
                        for seed in SEEDS:
                            rr=load(P20/('R6b_'+domain)/(role+'_H1_seed'+str(seed)+'_statistics.json'))
                            vals.append(next(r['AP'] for r in rr if r['population']=='all' and r['threshold']==th and r['wait_ms']=='four_wait_mean'))
                        summaries.append(dict(domain=domain,role=role,kind='real',amplitude=0.,method='CoMET_A',threshold=th,
                            AP=float(np.mean(vals)),delta_H1=0.,delta_H1_CI95=[0.,0.],ratio_clean_H1=1.,
                            exact_shared_H1=True,state_identity=ev(OUT/domain/(role+'_state_identity.json'))))
                    path=OUT/'cases'/(domain+'_'+role+'_real');path.mkdir(parents=True,exist_ok=False);atomic(path/'summary.json',summaries)
                else:case(domain,role,'T1',0.,True)
        for domain in ('author','strict60'):
            for amp in (3.,5.):case(domain,'val21','T1',amp);completed.append([domain,'val21',amp])
        decision=gate();atomic(OUT/'adoption_gate.json',decision)
        TABLE.mkdir(parents=True,exist_ok=True)
        with (TABLE/'results.csv').open('x',newline='') as f:
            # Heterogeneous exact-zero reference rows deliberately stay explicit.
            cols=sorted(set().union(*(r.keys() for r in decision['panels'])));writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(decision['panels'])
        with (TABLE/'README.md').open('x') as f:f.write('# R8d CoMET-A\n\n'+('两项预注册门通过，接受为研究扩展。' if decision['accepted_extension'] else '预注册门未通过；保留描述性结果。')+
            '\n\n真实数据退化逐位核验、注入生产与独立复核见 '+str(OUT.relative_to(ROOT))+'。T3恢复时间随R8c压力测试补齐；未据val调整。论文数字与根表不变。\n')
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',gate=ev(OUT/'adoption_gate.json'),
            accepted_extension=decision['accepted_extension'],T3_descriptive_pending_R8c=True,wall_s=time.monotonic()-start_time))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',result=ev(OUT/'terminal.json'),completed=completed))
        record('R8d_chain',dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(dest/'terminal.json')))
        record('R8d',dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed=completed))
        record('R8d_chain',dict(status='failed_retained',owner=None,resource_owner=False,terminal=ev(dest/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','chain'));a=p.parse_args();prepare() if a.mode=='prepare' else chain()
