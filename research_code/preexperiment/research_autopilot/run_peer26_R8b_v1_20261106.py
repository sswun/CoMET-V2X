"""Formal CoMET-M: fit-only regression, independent cal gate, then fixed outcomes.

No detector training/forward; three frozen readout heads are evaluated on the
actual new geometry. Calibration is always finished and independently checked
before any val/hold AP is read. Failed calibration never opens formal AP.
"""
import argparse,csv,time,traceback
from pathlib import Path
import numpy as np
from peer26_v1_20261106 import *
import peer26_R8_estimators_v1_20261106 as model
from peer20_population_core_v1_20261105 import data,annotations,packed,statistic,REC
import run_peer19_recipe_v3_20261104 as recipe
from review_peer16_E4_E7_AP_v1_20261103 import check,geometry
from run_peer20_auxiliary_GT_v1_20261105 import changed_truth,rematch
from run_peer21_label_median_GT_v1_20261105 import changed_truth as median_truth
DOMAINS=('author','strict60','v2xvit_seed2024','v2xvit_seed2025','where2comm')
ROLES=('calibration8','val21','mechanism_holdout14');SEEDS=(2023,2024,2025)
OUT=BASE/'R8b';REVIEW=BASE/'R8b_independent';TABLE=ROOT/'docs/research_v2/R8b'

def prepare():
    for name in ('R8b','R8b_independent','R8b_chain'):
        register(name,__file__,__doc__,dict(CPU_threads=1,GPU_forwards=0,detector_updates=0,replicates=2000,
            wall_h='estimated 12-72 including full auxiliary independent review; before Nov14',
            disk_MiB='<750 compact state/selected candidate score+TP and cluster draws; shared pools/GT/head refs'),
            models=['CoMET_M','CoMET_M_speed_report_only'],domains=list(DOMAINS),roles=list(ROLES),
            source_calibrations={d:ev(P20/('R6b_'+d)/'calibration.json') for d in DOMAINS},
            source_acceptance={d:ev(P20/('R6b_'+d+'_independent')/'terminal.json') for d in DOMAINS},
            strict_cal4=ev(P19/'strict_cal4_H1_independent/terminal.json'),
            algorithm=ev(model.__file__),gate='peer25 exact five gates; cal five-domain AP70 mean>0 before val; no new tuning',
            shared_geometry_GT='original domain material; state deltas and selected candidate IDs stored, no raw cache copies',
            speed_ablation='OLS x on intercept+speed, y intercept=fit target y mean; p0 from the actually constrained residual',
            independent_scope='Rebuild motion/matching/KF; independent heading/polygons/TP/AP/bootstrap; frozen neural scores shared')

def context(domain):
    cal=load(P20/('R6b_'+domain)/'calibration.json');recipe.OUT=REC
    heads={s:recipe.head(domain,cal['kind'],s) for s in SEEDS}
    for s,h in heads.items():assert ev(h/'direct_u.pth')==cal['heads'][str(s)]
    return cal['q'],cal['tau'],heads

def tag(role,method,seed):return role+'_'+method+'_seed'+str(seed)

def baseline(domain,role,seed):
    path=P20/('R6b_'+domain)/(role+'_H1_seed'+str(seed)+'_predictions.npz')
    with np.load(path) as z:p={k:z[k] for k in z.files}
    return p,'H1_seed'+str(seed),path

def stats(domain,role,method,seed,m,rows,p,label,b,review,auxiliary=False):
    directory=(REVIEW if review else OUT)/domain;source=OUT/domain;name=tag(role,method,seed)
    pops,_=annotations(rows);targets=[('official',None)]
    if auxiliary:
        targets += [('all_gaps',changed_truth(m,rows,role,float('inf'))[0]),('label_median',median_truth(m,rows,role,None)[0])]
    result=[];draws={}
    for ref,gt in targets:
        prediction=p if gt is None else dict(p,**rematch(m,p,label,b,gt,review))
        if gt is not None:
            tp={k:v for k,v in prediction.items() if k.endswith('__tp50') or k.endswith('__tp70')}
            tp_path=source/(name+'_'+ref+'_TP.npz')
            if review:
                with np.load(tp_path) as z:assert all(np.array_equal(v,z[k]) for k,v in tp.items())
            else:np.savez_compressed(tp_path,**tp)
        for pop in ('all','colabelled') if ref=='official' else ('all',):
            for th in (50,70):
                rr,dd=statistic(packed(m,prediction,label,th),pops[pop],review)
                result.extend(dict(r,reference=ref,population=pop,threshold=th) for r in rr)
                draws[ref+'_'+pop+'_AP'+str(th)]=dd
    path=source/(name+'_statistics.json');drawpath=source/(name+'_draws.npz')
    if review:
        gold=load(path);assert len(gold)==len(result)
        for a,z in zip(result,gold):
            assert (a['reference'],a['population'],a['threshold'],a['wait_ms'])==(z['reference'],z['population'],z['threshold'],z['wait_ms'])
            assert abs(a['AP']-z['AP'])<1e-8 and np.max(np.abs(np.asarray(a['CI95'])-z['CI95']))<1e-8
        with np.load(drawpath) as z:assert all(np.max(np.abs(v-z[k]))<1e-8 for k,v in draws.items())
        atomic(directory/(name+'_checked.json'),dict(status='passed',statistics=ev(path),draws=ev(drawpath),independent_AP_CI=True))
    else:atomic(path,result);np.savez_compressed(drawpath,**draws)

def role_run(domain,role,review=False,auxiliary=False):
    dest=(REVIEW if review else OUT)/domain;dest.mkdir(parents=True,exist_ok=True);src=OUT/domain
    q,tau,heads=context(domain);dataset,_,_=data(domain);m,rows=dataset[role]
    n,z=model.observations(m,review);dh=model.filter_state(m,n,z,q,review=review)[0]
    with np.load(P20/('R6b_'+domain)/(role+'_H1_state.npz')) as v:assert np.array_equal(v['delta'],dh),'H1 replay not bitwise'
    fits=load(src/'fit24.json');deltas={'H1':dh}
    for method in ('CoMET_M','CoMET_M_speed'):
        fit=fits[method];d=model.motion_state(m,rows,q,fit,review)
        statepath=src/(role+'_'+method+'_state.npz')
        if review:
            with np.load(statepath) as v:assert np.max(np.abs(v['delta']-d))<1e-11
        else:np.savez_compressed(statepath,delta=d)
        deltas[method]=d
    done=[]
    for method,d in deltas.items():
        b=geometry(m,d) if review else recipe.geometry(m,d)[0]
        f=None if review or method=='H1' else recipe.geometry(m,d)[1]
        for seed,head in heads.items():
            name=tag(role,method,seed);label=method+'_seed'+str(seed);predpath=src/(name+'_predictions.npz')
            if method=='H1':
                p,label,original=baseline(domain,role,seed)
                if not review:atomic(src/(name+'_shared_prediction.json'),dict(prediction=ev(original),no_copy=True))
                gold=load(P20/('R6b_'+domain)/(role+'_H1_seed'+str(seed)+'_AP.json'))
            elif review:
                with np.load(predpath) as v:p={k:v[k] for k in v.files}
                gold=load(src/(name+'_AP.json'))
            else:
                gold,p=recipe.evaluate(m,b,f,head,label,tau)
                atomic(src/(name+'_AP.json'),gold);np.savez_compressed(predpath,**p)
            if review:
                checks=check(m,b,p,label,gold)
                assert all(r['TP_mismatch'+str(th)]==0 and abs(r['error'+str(th)])<1e-8 for r in checks for th in (50,70))
            stats(domain,role,method,seed,m,rows,p,label,b,review,auxiliary)
            done.append(name);atomic((REVIEW if review else OUT)/'progress.json',dict(created_at=now(),domain=domain,role=role,completed=done))
    atomic(dest/(role+'_terminal.json'),dict(created_at=now(),status='complete_independently_reviewed' if review else 'complete_light_checked',
        completed=done,q=q,tau=tau,head_weights={str(s):ev(h/'direct_u.pth') for s,h in heads.items()},
        auxiliary=auxiliary,three_head_seeds_not_detector_seeds=True))

def fitcal(review=False):
    start_time=time.monotonic();dest=REVIEW if review else OUT
    assert load(dest/'contract.json')['script']==ev(__file__)
    for domain in DOMAINS:
        src=OUT/domain;src.mkdir(exist_ok=True);q,_,_=context(domain);dataset,_,_=data(domain);m,rows=dataset['fit24']
        fits={method:model.fit_motion(m,rows,q,method.endswith('_speed'),review) for method in ('CoMET_M','CoMET_M_speed')}
        if review:
            gold=load(src/'fit24.json')
            for method,fit in fits.items():
                assert np.max(np.abs(np.asarray(fit['coef'])-gold[method]['coef']))<1e-11
                assert abs(fit['p0']-gold[method]['p0'])<1e-11
        else:atomic(src/'fit24.json',fits)
        role_run(domain,'calibration8',review)
    atomic(dest/'cal_terminal.json',dict(created_at=now(),status='complete_independently_reviewed' if review else 'complete_light_checked',
        wall_s=time.monotonic()-start_time,formal_AP_read=False))

def summaries(roles,auxiliary=False):
    result=[]
    for domain in DOMAINS:
        for role in roles:
            for method in ('CoMET_M','CoMET_M_speed'):
                for ref in ('official','all_gaps','label_median') if auxiliary else ('official',):
                    for pop in ('all','colabelled') if ref=='official' else ('all',):
                        for th in (50,70):
                            points={};samples={}
                            for variant in (method,'H1'):
                                pp=[];dd=[]
                                for seed in SEEDS:
                                    name=tag(role,variant,seed)
                                    row=next(x for x in load(OUT/domain/(name+'_statistics.json')) if x['reference']==ref and x['population']==pop and x['threshold']==th and x['wait_ms']=='four_wait_mean')
                                    pp.append(row['AP'])
                                    with np.load(OUT/domain/(name+'_draws.npz')) as z:dd.append(z[ref+'_'+pop+'_AP'+str(th)].mean(0))
                                points[variant]=float(np.mean(pp));samples[variant]=np.mean(dd,axis=0)
                            delta=points[method]-points['H1'];diff=samples[method]-samples['H1']
                            result.append(dict(domain=domain,role=role,method=method,reference=ref,population=pop,threshold=th,
                                AP=points[method],H1_AP=points['H1'],delta=delta,CI95=np.quantile(diff,[.025,.975]).tolist(),
                                replicates=2000,heads=3,waits=4))
    return result

def decision(cal_only=False):
    rows=summaries(('calibration8',) if cal_only else ('val21','mechanism_holdout14'),not cal_only)
    if cal_only:
        chosen=[r for r in rows if r['method']=='CoMET_M' and r['population']=='all' and r['threshold']==70]
        mean=float(np.mean([r['delta'] for r in chosen]));passed=mean>0
        value=dict(created_at=now(),status='complete_independently_reviewed',cal_gate_passed=passed,cal_five_domain_delta_AP70=mean,
            input_roles=['calibration8'],formal_AP_read=False,panels=rows,independent=ev(REVIEW/'cal_terminal.json'))
        atomic(OUT/'cal_gate.json',value);return passed
    primary=[r for r in rows if r['method']=='CoMET_M' and r['reference']=='official' and r['population']=='all']
    val=[r for r in primary if r['role']=='val21' and r['threshold']==70]
    hold=[r for r in primary if r['role']=='mechanism_holdout14' and r['threshold']==70]
    ap50=[r for r in primary if r['role']=='val21' and r['threshold']==50]
    checks=dict(calibration=load(OUT/'cal_gate.json')['cal_gate_passed'],
        author_CI_positive=next(r for r in val if r['domain']=='author')['CI95'][0]>0,
        val_mean_at_least_point10=float(np.mean([r['delta'] for r in val]))>=.10,
        at_least_four_positive=sum(r['delta']>0 for r in val)>=4,no_domain_below_minus_point20=min(r['delta'] for r in val)>=-.20,
        hold_mean_nonnegative=float(np.mean([r['delta'] for r in hold]))>=0,
        val_AP50_mean_at_least_minus_point10=float(np.mean([r['delta'] for r in ap50]))>=-.10)
    atomic(OUT/'adoption_gate.json',dict(created_at=now(),status='complete_independently_reviewed',accepted=all(checks.values()),checks=checks,panels=rows))
    return all(checks.values())

def finish(accepted,cal_failed=False):
    TABLE.mkdir(parents=True,exist_ok=True);roles=('calibration8',) if cal_failed else ROLES
    rows=summaries(('calibration8',))+([] if cal_failed else summaries(ROLES[1:],True))
    path=TABLE/'results.csv';assert not path.exists()
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    gate=OUT/('cal_gate.json' if cal_failed else 'adoption_gate.json')
    value=dict(created_at=now(),status='complete_independently_reviewed',accepted_extension=accepted,
        cal_gate_failed=cal_failed,formal_AP_not_read=cal_failed,gate=ev(gate),table=ev(path),
        frozen_paper_unchanged=True,estimator_line_closed_if_failed=True,peer26_CoMET_A_explicit_exception=True)
    atomic(OUT/'terminal.json',value);record('R8b',dict(status=value['status'],owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')))
    with (TABLE/'README.md').open('x') as f:f.write('# R8b CoMET-M\n\n'+('预注册门通过，接受为研究扩展。' if accepted else '预注册门未通过，保留负结果，不递归调整。')+
        ('cal8 前置门失败，未读取 val/hold AP。' if cal_failed else '五域、三冻结头、四等待，2000 次整序列配对 CI，独立复核一次。')+
        '\n\n论文数字与根表未改；来源与门见 '+str(gate.relative_to(ROOT))+'。\n')

def chain():
    import subprocess,sys
    dest=BASE/'R8b_chain';done=[];begin=time.monotonic()
    try:
        assert now()<DEADLINE
        for mode in ('fitcal','review_cal'):
            with (dest/(mode+'.log')).open('x') as log:rc=subprocess.run([sys.executable,'-u',__file__,mode],stdout=log,stderr=subprocess.STDOUT).returncode
            assert rc==0,(mode,rc);done.append(mode)
        passed=decision(True)
        if passed:
            for domain in DOMAINS:
                for role in ROLES[1:]:
                    for review in (False,True):role_run(domain,role,review,True)
            accepted=decision(False);finish(accepted)
        else:finish(False,True)
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',result=ev(OUT/'terminal.json'),wall_s=time.monotonic()-begin))
        record('R8b_chain',dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(dest/'terminal.json')))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed=done))
        record('R8b_chain',dict(status='failed_retained',owner=None,resource_owner=False,terminal=ev(dest/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','chain','fitcal','review_cal'));a=p.parse_args()
    prepare() if a.mode=='prepare' else chain() if a.mode=='chain' else fitcal(a.mode=='review_cal')
