"""Finite light R7b postcheck: cal-only choice, both tiers, original gate.

Reuses independently polygon-matched AP/2000 whole-sequence draws; it is not
another training, forward, bootstrap sample or adoption criterion. No dataset
imports while waiting. Both A/B are reported; only the cal-selected tier is
judged by the existing val gate. Holdout signs never enter selection/adoption.
"""
import argparse
import fcntl
import time
import traceback
from peer21_v1_20261105 import *

NAME='R7b_gate_postcheck'
OUT=BASE/NAME
SCIENCE=BASE/'R7b_analysis'
SEEDS=(20261101,20261102,20261103)


def prepare():
    register(NAME,__file__,__doc__,dict(CPU_threads=1,updates=0,GPU_forwards=0,
        disk_MiB='<3 references, both-tier contrasts and gate verification; shared draws only'),
        source=ev(ROOT/'preexperiment/research_autopilot/consume_peer21_R7b_v1_20261105.py'),
        no_new_selection=True,no_new_adoption_gate=True,upstream_independent_matching_required=True,
        waiting='R7b original consumer canonical terminal; stdlib only',R7c_no_launch=True)


def verify_ref(value):
    assert ev(ROOT/value['path'])==value,value


def verify():
    import numpy as np
    terminal=load(SCIENCE/'terminal.json');assert terminal['status']=='complete_independently_reviewed'
    verify_ref(terminal['decision'])
    choice=load(SCIENCE/'calibration_choice.json');means={};refs=[]
    # Finish the independent calibration decision before opening formal files.
    for tier in ('A','B'):
        scores=[]
        for seed in SEEDS:
            src=SCIENCE/f'{tier}_{seed}_calibration8'
            assert load(src.with_name(src.name+'_independent')/'terminal.json')['status']=='complete_independently_reviewed'
            scores.append(next(r['AP'] for r in load(src/'result_statistics.json')
                if (r['population'],r['threshold'],r['wait_ms'])==('all',70,'four_wait_mean')))
            refs.append(ev(src/'result_statistics.json'))
        means[tier]=sum(scores)/3
    selected='B' if means['B']>means['A'] else 'A'
    assert selected==choice['selected'] and choice['roles_used']==['calibration8']
    assert all(abs(means[t]-choice['means'][t])<1e-12 for t in means)
    assert refs==choice['sources']
    atomic(OUT/'calibration_verified.json',dict(created_at=now(),selected=selected,means=means,
        roles_used=['calibration8'],sources=refs,verified_before_reading_formal=True))
    independent_rows=0;source_refs=[];cache={}
    for tier in ('A','B'):
        for seed in SEEDS:
            for role in ('calibration8','val21','mechanism_holdout14'):
                src=SCIENCE/f'{tier}_{seed}_{role}';review=src.with_name(src.name+'_independent')
                a,b=load(src/'terminal.json'),load(review/'terminal.json')
                assert a['canonical_GT_identity_checked'] and b['canonical_GT_identity_checked']
                assert b['polygon_matching_independent'] and b['AP_bootstrap_independent']
                verify_ref(a['statistics']);verify_ref(b['statistics'])
                ident=load(src/'identity.json')
                for k in ('raw_terminal','records','contract'):verify_ref(ident[k])
                assert ident['canonical_GT_same_as_H1'] and ident['all_temporal_keys'] and ident['input']=='CoMET_H1'
                prod=load(src/'result_statistics.json');gold=load(review/'result_statistics.json')
                assert len(prod)==len(gold)==40
                baseline=load(P20/'R6b_author'/(role+'_H1_seed2023_statistics.json'))
                for x,y,z in zip(prod,gold,baseline):
                    assert (x['population'],x['threshold'],x['wait_ms'])==(y['population'],y['threshold'],y['wait_ms'])==(z['population'],z['threshold'],z['wait_ms'])
                    assert x['replicates']==y['replicates']==2000
                    assert abs(x['AP']-y['AP'])<1e-8 and np.max(abs(np.array(x['CI95'])-y['CI95']))<1e-8
                    if x['wait_ms']!='four_wait_mean':assert x['GT']==z['GT'] and x['conditions']==z['conditions']
                independent_rows+=40
                source_refs.append(dict(production=ev(src/'result_statistics.json'),independent=ev(review/'result_statistics.json')))

    def result(tier,role,method,index):
        key=tier,role,method,index
        if key in cache:return cache[key]
        seed=SEEDS[index]
        if method=='H1_clean_native':src=SCIENCE/f'{tier}_{seed}_{role}';stem='result'
        elif method=='H1_zero_training':
            src=P20/'R6b_author';stem=role+'_H1_seed'+str(2023+index)
            assert load(P20/'R6b_author_independent/terminal.json')['status']=='complete_independently_reviewed'
        else:
            src=P20/f'R7_analysis_seed{seed}'
            stem=role+'_'+('clean' if method=='clean_K1' else method)
            assert load(src.with_name(src.name+'_independent')/'terminal.json')['status']=='complete_independently_reviewed'
        with np.load(src/(stem+'_draws.npz')) as z:draws={k:z[k] for k in z.files}
        values={(r['population'],r['threshold'],r['wait_ms']):r for r in load(src/(stem+'_statistics.json'))}
        cache[key]=(values,draws);return cache[key]

    panels=[]
    for tier in ('A','B'):
        for role in ('val21','mechanism_holdout14'):
            for ref in ('clean_K1','original_K4','PUBLIC_finetuned','H1_zero_training'):
                for population in ('all','zero','colabelled','nonzero'):
                    for th in (50,70):
                        values=[];draws=[];key=population+'_AP'+str(th)
                        for i in range(3):
                            a,ad=result(tier,role,'H1_clean_native',i);b,bd=result(tier,role,ref,i)
                            for w in (0,100,200,400):assert a[population,th,w]['GT']==b[population,th,w]['GT']
                            values.append([a[population,th,w]['AP']-b[population,th,w]['AP'] for w in (0,100,200,400)])
                            draws.append(ad[key]-bd[key])
                        # Independent subtraction before seed/wait reduction.
                        dd=sum(draws)/3;point=sum(np.asarray(x) for x in values)/3
                        for iw,w in enumerate((0,100,200,400,'four_wait_mean')):
                            v=dd[iw] if iw<4 else dd.sum(0)/4
                            pt=point[iw] if iw<4 else point.sum()/4
                            panels.append(dict(tier=tier,role=role,population=population,threshold=th,reference=ref,
                                wait_ms=w,delta_pp=float(pt),CI95=np.quantile(v,[.025,.975]).tolist(),replicates=2000))
    assert len(panels)==640
    by={(r['tier'],r['role'],r['population'],r['threshold'],r['reference'],r['wait_ms']):r for r in panels}
    original=load(SCIENCE/'contrasts.json');assert len(original)==320
    for r in original:
        x=by[selected,r['role'],r['population'],r['threshold'],r['reference'],r['wait_ms']]
        assert abs(x['delta_pp']-r['delta_pp'])<1e-10 and np.max(abs(np.array(x['CI95'])-r['CI95']))<1e-10
    def row(ref,th):return by[selected,'val21','all',th,ref,'four_wait_mean']
    p50,p70,h50,h70=row('PUBLIC_finetuned',50),row('PUBLIC_finetuned',70),row('H1_zero_training',50),row('H1_zero_training',70)
    checks=dict(PUBLIC_AP50_CI_low_positive=p50['CI95'][0]>0,PUBLIC_AP70_CI_low_positive=p70['CI95'][0]>0,
        H1_AP50_at_least1pp=h50['delta_pp']>=1.,H1_AP50_CI_low_positive=h50['CI95'][0]>0,H1_AP70_at_least_minus030=h70['delta_pp']>=-.30)
    decision=load(SCIENCE/'decision.json')
    assert decision['selected_tier']==selected and checks==decision['checks'] and all(checks.values())==decision['qualifies']
    for source in load(SCIENCE/'source_identities.json'):
        verify_ref(source['stats']);verify_ref(source['draws'])
    atomic(OUT/'both_tier_contrasts.json',panels)
    atomic(OUT/'decision_verified.json',dict(created_at=now(),selected=selected,checks=checks,qualifies=all(checks.values()),
        original_decision=ev(SCIENCE/'decision.json'),both_tiers_reported=True,holdout_not_gate=True,
        all_seed_same_sign_not_gate=True,no_new_threshold=True,main_method_unchanged='CoMET_H1',
        R7c_no_launch=True,no_automatic_root_table_mutation=True))
    atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',AP_rows_verified=independent_rows,
        original_contrasts_verified=320,both_tier_contrasts=640,decision=ev(OUT/'decision_verified.json'),
        sources=source_refs,matching_independent_upstream=True,not_independent_training=True))
    record(NAME,dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')))


def run():
    lock=(OUT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        paths=[BASE/'R7b_consumer_chain/terminal.json',SCIENCE/'terminal.json']
        while not all(p.exists() for p in paths):
            for p in paths:
                if p.exists():assert load(p)['status']!='failed_retained',p
            atomic(OUT/'progress.json',dict(created_at=now(),waiting=[str(p.relative_to(ROOT)) for p in paths if not p.exists()],
                dataset_waiters=0,numpy_loaded=False,no_model_wakeup=True));time.sleep(45)
        verify()
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(NAME,dict(status='failed_retained',owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));globals()[p.parse_args().mode]()
