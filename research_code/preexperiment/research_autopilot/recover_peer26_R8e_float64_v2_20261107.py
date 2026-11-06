"""Restore independent R8e polygon precision, not the frozen cal-only method.

The production evaluator converts boxes to float64 before trigonometry. The
first independent polygon code applied np.cos to float32 angles. Keep every
production input, score, threshold, arm, and completed case unchanged. Promote
only independent polygon inputs to float64; independently match/AP/CI again.
"""
import argparse,csv,time,traceback
import numpy as np
from peer26_v1_20261106 import *
import run_peer26_R8e_v1_20261107 as e
ORIGINAL=e.OUT;OUT=BASE/'R8e_recovery_v2';original_matrix=e.matrix

def matrix(a,b,review=False):
    if review:return original_matrix(np.asarray(a,np.float64),np.asarray(b,np.float64),True)
    return original_matrix(a,b,False)

def prepare():
    register('R8e_recovery_v2',__file__,__doc__,dict(CPU_threads=1,GPU_forwards=0,updates=0,wall_h='estimated1-4',disk_MiB='<100 only missing selected hold cases; reuse15 accepted cases and first failed production'),
        failure=ev(BASE/'R8e_chain/terminal.json'),replay=ev(BASE/'R8e_identity_diagnostic/diagnostic.json'),crossed=ev(BASE/'R8e_boundary_diagnostic/diagnostic.json'),
        frozen_choice=ev(ORIGINAL/'cal_choice.json'),cal_mapping=ev(ORIGINAL/'score_calibration.json'),no_new_cal_arm=True,no_production_change=True)

def case(role,seed,arm):
    old=ORIGINAL/(role+'_'+str(seed)+'_'+e.tag(arm))
    return old if (old/'terminal.json').exists() else OUT/old.name

def evaluate(role,seed,arm,models):
    name=role+'_'+str(seed)+'_'+e.tag(arm);old=ORIGINAL/name;dest=OUT/name;dest.mkdir(parents=True,exist_ok=False)
    shared=(old/'predictions.npz').exists();source=old if shared else dest
    atomic(dest/'identity.json',dict(frozen_arm=arm,cal_mapping=ev(ORIGINAL/'score_calibration.json'),
        shared_production=ev(old/'identity.json') if shared else None,tier2_weight=ev(P21/'R7b/A'/('seed'+str(seed))/'training/final.pth'),
        precision='Independent polygon float64 before trig, same accepted float32 IoU; no threshold/tie change'))
    gold=e.npz(old/'predictions.npz') if shared else None;goldstats=load(old/'statistics.json') if shared else None;golddraw=e.npz(old/'draws.npz') if shared else None
    for review in ((True,) if shared else (False,True)):
        m,rows,frames,_,_=e.sources(role,seed,review);scores=[];tps={50:[],70:[]};offset=[0];origins=[];ids=[]
        if shared:
            a,sa,b,sb,*_=frames[172]
            before=e.combine(a,sa,b,sb,arm,models,True)
            e.matrix=matrix
            after=e.combine(a,sa,b,sb,arm,models,True);canonical=e.combine(a,sa,b,sb,arm,models,False)
            assert all(np.array_equal(x,y) for x,y in zip(after,canonical))
            atomic(OUT/'precision_qualification.json',dict(created_at=now(),late_dtype=str(a.dtype),native_dtype=str(b.dtype),
                original_independent_differs=not all(np.array_equal(x,y) for x,y in zip(before,canonical)),
                float64_independent_exact=True,matrix_max_abs=float(np.max(abs(matrix(a,b,True)-matrix(a,b,False)))),
                no_threshold_or_tie_rule_change=True,no_fit_or_AP_choice=True))
        else:e.matrix=matrix
        for a,sa,b,sb,gt,ai,bi in frames:
            boxes,ss,origin=e.combine(a,sa,b,sb,arm,models,review);scores.extend(ss);offset.append(len(scores));origins.extend(origin[:,0]);ids.extend(ai[j] if k==0 else bi[j] for k,j in origin)
            if review:
                for th in (50,70):
                    _,tp,_,n=e.independent.match_frame(boxes,-np.arange(len(boxes),dtype=float),np.zeros(len(boxes)),gt,np.zeros(len(gt)),th/100);assert n==len(gt);tps[th].extend(tp)
            else:
                overlap=e.iou32(boxes,gt)
                for th in (50,70):tps[th].extend((e.greedy(overlap,np.arange(len(boxes)),th/100)>=0).astype(np.int8))
        p={'combo__score':np.asarray(scores),'combo__offset':np.asarray(offset),'combo__origin':np.asarray(origins),'combo__candidate':np.asarray(ids),
            **{'combo__tp'+str(th):np.asarray(v,np.int8) for th,v in tps.items()}}
        if review:assert all(np.array_equal(p[k],gold[k]) for k in p),'independent precision recovery mismatch'
        else:gold=p;np.savez_compressed(dest/'predictions.npz',**p)
        stats=[];draws={}
        for th in (50,70):
            rr,dd=e.statistic(e.packed(m,p,'combo',th),np.ones(len(rows),bool),review);stats.extend(dict(r,threshold=th) for r in rr);draws['AP'+str(th)]=dd
        if review:
            for x,y in zip(stats,goldstats):assert abs(x['AP']-y['AP'])<1e-8 and np.max(abs(np.asarray(x['CI95'])-y['CI95']))<1e-8
            assert all(np.max(abs(draws[k]-golddraw[k]))<1e-8 for k in draws)
        else:goldstats=stats;golddraw=draws;atomic(dest/'statistics.json',stats);np.savez_compressed(dest/'draws.npz',**draws)
    if shared:
        atomic(dest/'shared_production.json',dict(statistics=ev(old/'statistics.json'),draws=ev(old/'draws.npz'),predictions=ev(old/'predictions.npz')))
    atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',statistics=ev(source/'statistics.json'),draws=ev(source/'draws.npz'),predictions=ev(source/'predictions.npz'),
        canonical_GT_identity_checked=True,independent_combination_matching_AP_CI=True))

def summaries(role,arm):
    result=[]
    for th in (50,70):
        values={};samples={}
        for method in ('combo','tier1','tier2'):
            pts=[];draws=[]
            for i,seed in enumerate(e.SEEDS):
                if method=='combo':
                    t=load(case(role,seed,arm)/'terminal.json');stat=ROOT/t['statistics']['path'];draw=ROOT/t['draws']['path'];prefix=''
                elif method=='tier1':
                    src=P20/'R6b_author';stat=src/(role+'_H1_seed'+str(2023+i)+'_statistics.json');draw=src/(role+'_H1_seed'+str(2023+i)+'_draws.npz');prefix='all_'
                else:
                    src=P21/'R7b_analysis'/('A_'+str(seed)+'_'+role);stat=src/'result_statistics.json';draw=src/'result_draws.npz';prefix='all_'
                rr=load(stat);pts.append(next(r['AP'] for r in rr if r['threshold']==th and r['wait_ms']=='four_wait_mean' and r.get('population','all')=='all'))
                with np.load(draw) as z:draws.append(z[prefix+'AP'+str(th)].mean(0))
            values[method]=float(np.mean(pts));samples[method]=np.mean(draws,axis=0)
        for baseline in ('tier1','tier2'):
            result.append(dict(role=role,threshold=th,reference=baseline,AP=values['combo'],baseline_AP=values[baseline],delta=values['combo']-values[baseline],
                paired_CI95=np.quantile(samples['combo']-samples[baseline],[.025,.975]).tolist(),replicates=2000,seeds=3,waits=4))
    return result

def run():
    begin=time.monotonic()
    try:
        arm=tuple(load(ORIGINAL/'cal_choice.json')['selected']);assert arm==(.3,False)
        models=load(ORIGINAL/'score_calibration.json')['models'];shared=[]
        for path in sorted(ORIGINAL.glob('*/terminal.json')):
            t=load(path);assert t['status']=='complete_independently_reviewed';shared.append(ev(path))
        assert len(shared)==15;atomic(OUT/'accepted_shared_cases.json',shared)
        for seed in e.SEEDS:evaluate('mechanism_holdout14',seed,arm,models[str(seed)])
        rows=summaries('val21',arm)+summaries('mechanism_holdout14',arm);checks={}
        for baseline in ('tier1','tier2'):
            r=next(r for r in rows if r['role']=='val21' and r['threshold']==70 and r['reference']==baseline)
            checks[baseline+'_val_gain_ge_point50']=r['delta']>=.50;checks[baseline+'_val_CI_positive']=r['paired_CI95'][0]>0
            checks[baseline+'_hold_same_sign']=next(r['delta'] for r in rows if r['role']=='mechanism_holdout14' and r['threshold']==70 and r['reference']==baseline)>0
        checks['val_AP50_ge_tier2_minus_point20']=next(r['delta'] for r in rows if r['role']=='val21' and r['threshold']==50 and r['reference']=='tier2')>=-.20
        atomic(OUT/'adoption_gate.json',dict(created_at=now(),accepted_extension=all(checks.values()),checks=checks,panels=rows,chosen=arm,original_failure_retained=True))
        e.TABLE.mkdir(parents=True,exist_ok=True)
        with (e.TABLE/'results.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with (e.TABLE/'README.md').open('x') as f:f.write('# R8e 两档框级组合\n\n'+('预注册门通过。' if all(checks.values()) else '预注册门未通过，不递归调参。')+'系统组合，cal8四组选择冻结。原工程失败保留，独立多边形输入升float64与生产语义一致；共享15个接受格和首个hold生产产物，仅补缺失独立核验及剩余hold。来源 '+str(OUT.relative_to(ROOT))+'。论文表不改。\n')
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',gate=ev(OUT/'adoption_gate.json'),table=ev(e.TABLE/'results.csv'),wall_s=time.monotonic()-begin))
        record('R8e_recovery_v2',dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
        record('R8e_recovery_v2',dict(status='failed_retained',owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));a=p.parse_args();globals()[a.mode]()
