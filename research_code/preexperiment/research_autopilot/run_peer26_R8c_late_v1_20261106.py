"""Descriptive R8c late pressure, sharded by detector; reuse R8d gate cases.

No selection. CoMET-M formal pressure is included only if its cal gate passed;
a failed cal gate is not bypassed by reading its val AP under another name.
T4 is explicitly unperformed, not approximated; cached R4 module translations
are not valid re-estimates after injection and therefore cannot be reused.
"""
import argparse,csv,time,traceback
import numpy as np
from peer26_v1_20261106 import *
import run_peer26_R8d_v1_20261106 as d
import run_peer26_R8b_v1_20261106 as b
import peer26_R8_estimators_v1_20261106 as model
from peer20_population_core_v1_20261105 import data,packed,statistic
from review_peer16_E4_E7_AP_v1_20261103 import geometry,check
OUT=BASE/'R8c_late';REVIEW=BASE/'R8c_late_independent'

def scenarios(domain):
    if domain in ('author','strict60'):
        return [(role,kind,amp) for role in b.ROLES for kind,amp in [*[('T1',a) for a in (.5,1.,2.,3.,5.)],('T2',2.),('T3',2.)]]
    return [('val21','T1',a) for a in (1.,2.,3.)]

def prepare():
    for domain in b.DOMAINS:
        register('R8c_late_'+domain,__file__,__doc__,dict(CPU_threads=1,GPU_forwards=0,updates=0,replicates=2000,
            wall_h='estimated 12-84 including independent review per domain; before Nov14',
            disk_MiB='<1400 author/strict or <250 other, compact selected scores/TP/state/draws; shared raw cache references'),
            domain=domain,cases=scenarios(domain),seed=2026111401,
            perturbation=ev(model.__file__),methods=['PUBLIC','single_mean','H1','CoMET_A','CoMET_M if cal gate passed'],
            source_case_engine=ev(d.__file__),R4_reuse='not directly possible: robust module consumes original estimated boxes/validity, not injected inputs; no new external runs',
            T4='not performed unless same-projection sensor origin is independently audited; no approximate rotation',
            pairing='2000 same whole-sequence bootstrap draws, all methods share per-sequence injected direction',
            T3_recovery='first post-step message where norm(injected-state+shift-clean-H1-state)<=.5m; no crossing=right censored; incremental perturbation error, not physical GT')

def add_motion(path,domain,role,kind,amp):
    """Add the fitted M state, independently recomputed TP/AP/paired draws."""
    import run_peer19_recipe_v3_20261104 as recipe
    m0,rows=data(domain)[0][role];m,shift=model.inject(m0,kind,amp);q,tau,heads=b.context(domain)
    fit=load(b.OUT/domain/'fit24.json')['CoMET_M'];state=model.motion_state(m,rows,q,fit)
    independent=model.motion_state(m,rows,q,fit,True);assert np.max(abs(state-independent))<1e-11
    np.savez_compressed(path/'CoMET_M_state.npz',delta=state)
    results={}
    for review in (False,True):
        boxes=geometry(m,independent) if review else recipe.geometry(m,state)[0]
        features=None if review else recipe.geometry(m,state)[1]
        for seed,head in heads.items():
            name='CoMET_M_seed'+str(seed);pred=path/(name+'_predictions.npz')
            if review:
                with np.load(pred) as z:p={k:z[k] for k in z.files}
                gold=load(path/(name+'_AP.json'));checks=check(m,boxes,p,name,gold)
                assert all(r['TP_mismatch'+str(th)]==0 and abs(r['error'+str(th)])<1e-8 for r in checks for th in (50,70))
            else:
                gold,p=recipe.evaluate(m,boxes,features,head,name,tau);atomic(path/(name+'_AP.json'),gold);np.savez_compressed(pred,**p)
            rows_stat=[];draws={}
            for th in (50,70):
                rr,dd=statistic(packed(m,p,name,th),np.ones(len(rows),bool),review)
                rows_stat.extend(dict(r,threshold=th) for r in rr);draws['AP'+str(th)]=dd
                if not review:results[seed,th]=(next(r['AP'] for r in rr if r['wait_ms']=='four_wait_mean'),dd.mean(0))
            if review:
                for a,z in zip(rows_stat,load(path/(name+'_statistics.json'))):assert abs(a['AP']-z['AP'])<1e-8 and np.max(abs(np.asarray(a['CI95'])-z['CI95']))<1e-8
                with np.load(path/(name+'_draws.npz')) as z:assert all(np.max(abs(x-z[k]))<1e-8 for k,x in draws.items())
            else:atomic(path/(name+'_statistics.json'),rows_stat);np.savez_compressed(path/(name+'_draws.npz'),**draws)
    summary=[]
    for th in (50,70):
        point=float(np.mean([results[s,th][0] for s in b.SEEDS]));dd=np.mean([results[s,th][1] for s in b.SEEDS],axis=0)
        base={}
        for method in ('H1','PUBLIC'):
            pp=[];samples=[]
            source=Path(load(path/'base_case.json')['directory'])
            for seed in b.SEEDS:
                stat=load(source/(method+'_seed'+str(seed)+'_statistics.json'))
                pp.append(next(r['AP'] for r in stat if r['threshold']==th and r['wait_ms']=='four_wait_mean'))
                with np.load(source/(method+'_seed'+str(seed)+'_draws.npz')) as z:samples.append(z['AP'+str(th)].mean(0))
            base[method]=(float(np.mean(pp)),np.mean(samples,axis=0))
        clean=[]
        for seed in b.SEEDS:
            stat=load(P20/('R6b_'+domain)/(role+'_H1_seed'+str(seed)+'_statistics.json'))
            clean.append(next(r['AP'] for r in stat if r['population']=='all' and r['threshold']==th and r['wait_ms']=='four_wait_mean'))
        summary.append(dict(domain=domain,role=role,kind=kind,amplitude=amp,method='CoMET_M',threshold=th,AP=point,
            CI95=np.quantile(dd,[.025,.975]).tolist(),delta_H1=point-base['H1'][0],delta_H1_CI95=np.quantile(dd-base['H1'][1],[.025,.975]).tolist(),
            recovery_PUBLIC=point-base['PUBLIC'][0],recovery_PUBLIC_CI95=np.quantile(dd-base['PUBLIC'][1],[.025,.975]).tolist(),
            ratio_clean_H1=point/float(np.mean(clean)),heads=3,waits=4,replicates=2000))
    atomic(path/'CoMET_M_summary.json',summary);return summary

def recovery(domain,role,path):
    m,rows=data(domain)[0][role];_,shift=model.inject(m,'T3',2.)
    with np.load(P20/('R6b_'+domain)/(role+'_H1_state.npz')) as z:clean=z['delta'].copy()
    with np.load(path/'states.npz') as z:states={k:z[k] for k in z.files}
    reports=[]
    for method,state in states.items():
        for seq in sorted(set(map(str,m['sequence']))):
            for wait in (0,100,200,400):
                ids=np.flatnonzero((m['sequence'].astype(str)==seq)&(m['wait_ms']==wait));ids=ids[np.linalg.norm(shift[ids],axis=1)>0]
                if not len(ids):continue
                error=np.linalg.norm(state[ids]+shift[ids]-clean[ids],axis=1);hit=np.flatnonzero(error<=.5)
                reports.append(dict(method=method,sequence=seq,wait_ms=wait,messages_to_first_half_meter=int(hit[0])+1 if len(hit) else None,
                    post_step_messages=len(ids),right_censored=not bool(len(hit))))
    return reports

def run(domain):
    dest=BASE/('R8c_late_'+domain);table=ROOT/'docs/research_v2/R8c'/domain;begin=time.monotonic();done=[]
    try:
        assert now()<DEADLINE
        assert load(d.OUT/'terminal.json')['status']=='complete_independently_reviewed'
        cg=load(b.OUT/'cal_gate.json');include_M=cg['cal_gate_passed']
        results=[];identities=[];recovery_rows=[]
        for role,kind,amp in scenarios(domain):
            assert now()<DEADLINE
            name=domain+'_'+role+'_'+kind+'_'+str(amp);shared=d.OUT/'cases'/name
            if (shared/'terminal.json').exists():
                assert load(shared/'terminal.json')['status']=='complete_independently_reviewed';base=shared
                rr=load(base/'summary.json')
            else:
                original_out,original_review=d.OUT,d.REVIEW;d.OUT=OUT;d.REVIEW=REVIEW
                try:rr=d.case(domain,role,kind,amp)
                finally:d.OUT=original_out;d.REVIEW=original_review
                base=OUT/'cases'/name
            resultdir=OUT/'motion'/name;resultdir.mkdir(parents=True,exist_ok=False)
            atomic(resultdir/'base_case.json',dict(directory=str(base),terminal=ev(base/'terminal.json'),no_raw_copy=True))
            results.extend(rr)
            if include_M:results.extend(add_motion(resultdir,domain,role,kind,amp))
            else:atomic(resultdir/'M_not_evaluated.json',dict(reason='R8b calibration gate failed; do not bypass no-val rule',cal_gate=ev(b.OUT/'cal_gate.json')))
            if kind=='T3':recovery_rows.extend(dict(r,domain=domain,role=role) for r in recovery(domain,role,base))
            done.append(name);identities.append(dict(case=name,base_summary=ev(base/'summary.json'),motion=ev(resultdir/'CoMET_M_summary.json') if include_M else None))
            atomic(dest/'progress.json',dict(created_at=now(),completed=done,last=name))
        table.mkdir(parents=True,exist_ok=True)
        cols=sorted(set().union(*(r.keys() for r in results)))
        with (table/'results.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(results)
        atomic(table/'T3_recovery.json',recovery_rows);atomic(table/'sources.json',identities)
        with (table/'README.md').open('x') as f:f.write('# R8c '+domain+'\n\n描述性压力测试，不选方法。三冻结头、四等待、2000整序列配对CI，生产与独立复核。'
            '\n\n五组远端框同步平移，GT不动，逐序列方向跨等待与检测器固定。T4未做；R4缓存不可当成注入后重新估计。'
            '\n\nCoMET-M '+('已纳入。' if include_M else '因R8b cal门失败未纳入，以免绕过不读val规则。')+'论文数字与根表不改。\n')
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',completed=done,
            table=ev(table/'results.csv'),sources=ev(table/'sources.json'),descriptive_only=True,wall_s=time.monotonic()-begin))
        record('R8c_late_'+domain,dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(dest/'terminal.json')))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed=done))
        record('R8c_late_'+domain,dict(status='failed_retained',owner=None,resource_owner=False,terminal=ev(dest/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('--domain',choices=b.DOMAINS);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.domain)
