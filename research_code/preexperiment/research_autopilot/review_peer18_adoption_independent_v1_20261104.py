"""Independently read the unchanged full/partial preregistered adoption gates.

Wait for separately implemented full2000draw statistics for every variant.
The partial N1/N1b/N2/N3 gate requires five-identity mean>=.25 and CI>0;
do not retroactively attach the full N4 hold gate to it. Report negative hold
and author-zero-shot explicitly. No retraining, per-cell choice or tex edit.
"""
import argparse,time,traceback,statistics
import numpy as np
from peer18_v1_20261104 import *
import run_peer17_final_recipe_v1_20261104 as final
from run_peer16_cached_methods_v1_20261103 import DOMAINS
from recover_peer18_candidate_statistics_launch_v2_20261104 import name_for
OUT=BASE/'adoption_independent_v1'
SEEDS=(2023,2024,2025)
VARIANTS=('N1','N1b','N2','N3','N4','N5','B1','B2')

def prepare():
    register('adoption_independent_v1',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        disk_MiB='<8 references and summary only',wall_h='bounded25h dependencies'),
        original_gate=ev(BASE/'adoption_read/contract.json'),original_read=ev(BASE/'adoption_read/terminal.json'),
        dependencies={str(s):ev(BASE/name_for(s)/'contract.json') for s in (0,*SEEDS)},
        authorization='peer18 full N4 gate then fixed simplicity order N1,N1b,N2,N3 partial gate; no new criteria or variant.')

def calculate(variant,reviewed):
    point={};rep={};absolute={}
    for role in ('val21','mechanism_holdout14'):
        for th in (50,70):
            key=role+'_AP'+str(th);matrix=np.zeros((5,3));draw=np.zeros((5,3,2000));ap=np.zeros((5,3))
            for j,d in enumerate(DOMAINS):
                for k,s in enumerate(SEEDS):
                    lane=0 if variant in ('N1','N1b') else s
                    rows=reviewed[lane];r=next(x for x in rows if x['domain']==d and x['role']==role and x['variant']==variant and x['seed']==s and x['threshold']==th and x['wait_ms']=='four_wait_mean')
                    matrix[j,k]=r['delta_pp']
                    label=role+'_seed'+str(s)+'_'+variant+'_minus_N0'
                    with np.load(BASE/name_for(lane)/d/(label+'_AP'+str(th)+'.npz')) as z:
                        draw[j,k]=z['four_wait_mean']
                    base=[x['ap'+str(th)] for x in load(final.OUT/d/'panels.json') if x['role']==role and x['seed']==s]
                    assert len(base)==4;ap[j,k]=statistics.fmean(base)+matrix[j,k]
            point[key]=matrix;rep[key]=draw;absolute[key]=ap
    d=point['val21_AP70'];r=rep['val21_AP70'];idx=list(DOMAINS).index('author')
    author=statistics.fmean(d[idx]);five=statistics.fmean(d.flat)
    ca=np.percentile(np.sum(r[idx],axis=0)/3,[2.5,97.5]);cf=np.percentile(np.sum(r,axis=(0,1))/15,[2.5,97.5])
    hold=point['mechanism_holdout14_AP70'];p50=point['val21_AP50']
    sa=statistics.stdev(absolute['val21_AP70'][idx]);sd=statistics.stdev(d[idx])
    gates=dict(author_ge_p40_CI_positive=author>=.4 and ca[0]>0,
        five_mean_ge_p25_no_domain_below_m20=five>=.25 and all(statistics.fmean(row)>=-.2 for row in d),
        hold_same_positive_sign=statistics.fmean(hold[idx])>0 and statistics.fmean(hold.flat)>0,
        AP50_noninferior=statistics.fmean(p50[idx])>=-.2 and statistics.fmean(p50.flat)>=-.2,
        three_seed_SD_le_p25=max(sa,sd)<=.25)
    return dict(variant=variant,author_val_AP70_delta=author,author_CI95=ca.tolist(),five_mean_val_AP70_delta=five,five_mean_CI95=cf.tolist(),
        by_domain_val_AP70_delta={d:statistics.fmean(row) for d,row in zip(DOMAINS,point['val21_AP70'])},
        by_domain_hold_AP70_delta={d:statistics.fmean(row) for d,row in zip(DOMAINS,hold)},
        author_hold_AP70_delta=statistics.fmean(hold[idx]),five_hold_AP70_delta=statistics.fmean(hold.flat),
        author_val_AP50_delta=statistics.fmean(p50[idx]),five_val_AP50_delta=statistics.fmean(p50.flat),
        author_AP70_seed_SD=sa,author_delta_seed_SD=sd,gates=gates,all_gates=all(gates.values()),
        partial_component_gate=five>=.25 and cf[0]>0,raw_seed_deltas={k:v.tolist() for k,v in point.items()})

def run():
    begin=time.monotonic();record('adoption_independent_v1',dict(status='waiting_independent_statistics',resource_owner=False,owner=None))
    try:
        deps=[BASE/name_for(s)/'terminal.json' for s in (0,*SEEDS)]
        while not all(p.exists() for p in deps):
            assert time.monotonic()-begin<25*3600;time.sleep(30)
        for p in deps:assert load(p)['status']=='complete_independently_reviewed',(p,load(p))
        record('adoption_independent_v1',dict(status='running_independent_gate_arithmetic',resource_owner=True))
        reviewed={s:load(BASE/name_for(s)/'panels.json') for s in (0,*SEEDS)}
        rows=[calculate(v,reviewed) for v in VARIANTS];old=load(BASE/'adoption_read/preregistered_reads.json')
        for r in rows:
            expected=next(x for x in old['rows'] if x['variant']==r['variant'])
            for key,value in r.items():
                if key in ('gates','all_gates','partial_component_gate','variant'):assert value==expected[key],(r['variant'],key)
                elif key.startswith('by_domain_hold'):continue
                elif isinstance(value,dict):
                    for k,v in value.items():assert np.max(abs(np.asarray(v)-np.asarray(expected[key][k])))<1e-8,(r['variant'],key,k)
                else:assert np.max(abs(np.asarray(value)-np.asarray(expected[key])))<1e-8,(r['variant'],key)
        full=next(r for r in rows if r['variant']=='N4')
        component=next((v for v in ('N1','N1b','N2','N3') if next(r for r in rows if r['variant']==v)['partial_component_gate']),None)
        decision='CoMET_Net_N4' if full['all_gates'] else 'CoMET_plus_'+component if component else 'retain_CoMET'
        assert decision==old['recommendation']
        transfer=[]
        for v in ('N2','N4','N5','B1','B2'):
            for target in DOMAINS:
                for role in ('val21','mechanism_holdout14'):
                    for th in (50,70):
                        delta=[]
                        for s in SEEDS:
                            src=BASE/'CoMET_Net/formal'/('seed'+str(s))/('author__to__'+target+'__'+v)
                            assert load(BASE/'CoMET_Net/independent'/('seed'+str(s))/src.name/'terminal.json')['status']=='complete_independently_reviewed'
                            p=[r['ap'+str(th)] for r in load(src/'panels.json') if r['role']==role]
                            n=[r['ap'+str(th)] for r in load(final.OUT/target/'panels.json') if r['role']==role and r['seed']==s]
                            assert len(p)==len(n)==4;delta.append(statistics.fmean(p)-statistics.fmean(n))
                        transfer.append(dict(variant=v,target=target,role=role,threshold=th,mean_delta_pp=statistics.fmean(delta),seed_SD=statistics.stdev(delta),
                            training_detector='author',is_zero_shot=target!='author',CI_not_computed=True))
        atomic(OUT/'preregistered_reads.json',dict(created_at=now(),rows=rows,recommendation=decision,
            independent_statistics=True,shared_matching=True,original_gate_unchanged=True,zero_shot_separate=transfer,
            caveats=['Five checkpoint identities are not five independent architectures;3 neural seeds are not detector-training seeds.',
                'Full N4 gate failed; N2 partial gate does not include full-gate hold condition. Negative hold remains reported.',
                'No independent training/full recurrence claim. Main.tex and old N0 table rows unchanged.']))
        if component:
            weights={d:{str(s):load(BASE/'CoMET_Net/training_recovery_v2'/('seed'+str(s))/d/component/'terminal.json')['weights'] for s in SEEDS} for d in DOMAINS}
        else:weights={}
        atomic(OUT/'method_registry.json',dict(created_at=now(),decision=decision,component=component,
            enabled_only_after_this_independent_gate=True,own_domain_weights=weights,
            original_N0_preserved=ev(P17/'final_recipe/contract.json'),no_new_fitting=True,
            name='CoMET with learned soft association' if component=='N2' else decision,
            scope='Own-detector trained component. Author-zero-shot separately reported; no detector feature/backbone retraining.'))
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',recommendation=decision,
            read=ev(OUT/'preregistered_reads.json'),method_registry=ev(OUT/'method_registry.json'),main_tex_unchanged=True,whole_release_closed=False))
        record('adoption_independent_v1',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record('adoption_independent_v1',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
