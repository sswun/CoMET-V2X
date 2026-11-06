"""Close known R6 population coverage gaps, no new method or GPU computation.

CoAlign complete systems (three fixed fusion seeds); frozen stronger-late
columns; R4 cal-selected arm's auxiliary reading. Root official tables unchanged.
"""
import argparse, csv, inspect, io, json, time, traceback
from pathlib import Path
from peer20_v1_20261105 import *

NAME='R6_coverage_completion_v1'
OUT=BASE/NAME

def prepare():
    for key in ('chain','coalign','coalign_independent','stronger_late','R4_auxiliary','tables'):
        register(NAME+'_'+key,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
            replicates=2000,disk_MiB='<120 compact outcomes/draws per CoAlign seed; <20 derived tables'),
            systems='Three existing CoAlign H1/K1/GRAPH full systems, val21 only, official and three auxiliary GT tiers',
            stronger_rule='Inherit P19 direct-contrast per-role/per-wait/per-IoU all-population columns; no new population/tier choices',
            R4='Keep original official -.3pp gate/cal-selected arm; auxiliary summaries do not select or adopt',
            finite_predecessor=ev(BASE/'finite_resource_DAG_v1/contract.json'))

def coalign_inputs(family,review):
    import numpy as np
    import run_peer19_coalign_H1_v1_20261104 as sources
    import peer20_saved_native_v1_20261105 as native
    for seed in sources.SEEDS:
        k1,graph=sources.old_outputs(seed)
        for method,path,arm in [('H1',sources.OUT/f'seed{seed}/evaluation','H1_INPUT'),('K1',k1,'K1_INPUT'),('GRAPH',graph,'COALIGN_FULL')]:
            def cases():
                recs=load(path/'records.json')
                for w in (0,100,200,400):
                    rs=[r for r in recs if r['wait_ms']==w]
                    with np.load(path/f'wait{w:03d}.npz') as z:
                        gtshards={r['GT_shard'] for r in rs};assert len(gtshards)==1
                        with np.load(ROOT/rs[0]['GT_shard']) as g:
                            for r in rs:
                                j=r['shard_case'];gt=g[f"c{r['GT_case']}_gt"]
                                yield r,z[f'c{j}_{arm}_box3d_lidar'],z[f'c{j}_{arm}_scores'],z[f'c{j}_{arm}_label_preds'],gt[:,:7],gt[:,7]
            label=f'CoAlign_{method}_fusionseed{seed}'
            m,rows,p,b=native.assemble('author','val21',cases(),label,review)
            yield 'val21',m,rows,p,label,b,dict(system_not_module=True,fusion_seed=seed,input=arm,
                evaluation=ev(path/'terminal.json'),records=ev(path/'records.json'),weight=ev(sources.weight(seed)),
                prior_full_system_review=ev(P19/'coalign_H1_independent/terminal.json'),
                calibration_and_head_not_fitted_here=True)

def coalign(review=False):
    import run_peer20_external_population_v1_20261105 as ext
    key=NAME+'_coalign'+('_independent' if review else '')
    code=inspect.getsource(ext.run).replace("'R6bd_external_'+family", "'R6_coverage_completion_v1_coalign'")
    # The frozen runner appends _independent itself; no change to numeric kernels.
    ns=dict(ext.__dict__,inputs=coalign_inputs,record=lambda unused,v:record(key,v))
    exec(compile(code,__file__+'::existing_R6_engine_CoAlign','exec'),ns);ns['run']('systems',review)

def stronger_late():
    import numpy as np
    from peer20_population_core_v1_20261105 import POPS,WAITS
    dest=BASE/(NAME+'_stronger_late');late=BASE/'R6bd_external_late'
    direct=load(P19/'H1_direct_contrasts/panels.json');done=[];choices={}
    for role in ('val21','mechanism_holdout14'):
        for tier in ('','_gap1s','_gap3s','_all_gaps'):
            tag=role+'_stronger_late'+tier;result=[];draws={}
            for th in (50,70):
                cols=[r['identity']['baseline_columns'] for r in direct if r['threshold']==th and r['identity']['role']==role and 'stronger_late' in r['comparison']]
                assert cols and all(x==cols[0] for x in cols)
                columns=cols[0];choices[role+'_AP'+str(th)]=columns
                rows={};dd={}
                for label in set(columns.values()):
                    base=role+'_'+label+tier;path=late/(base+'_statistics.json');rows[label]=load(path)
                    with np.load(late/(base+'_draws.npz')) as z:dd[label]={k:z[k].copy() for k in z.files}
                for pop in POPS:
                    x=np.vstack([dd[columns[str(w)]][pop+'_AP'+str(th)][i] for i,w in enumerate(WAITS)])
                    draws[pop+'_AP'+str(th)]=x
                    rr=[]
                    for w in WAITS:
                        label=columns[str(w)];r=next(a for a in rows[label] if a['population']==pop and a['threshold']==th and a['wait_ms']==w)
                        rr.append(dict(r,method=tag,frozen_column=label))
                    result.extend(rr);mean=x.mean(0)
                    result.append(dict(method=tag,population=pop,threshold=th,wait_ms='four_wait_mean',AP=float(np.mean([r['AP'] for r in rr])),CI95=np.quantile(mean,[.025,.975]).tolist(),replicates=2000,sequences=rr[0]['sequences']))
            atomic(dest/(tag+'_statistics.json'),result);np.savez_compressed(dest/(tag+'_draws.npz'),**draws)
            # Separate reader validates each copied cell and independently forms
            # the mean/percentiles from the written selected distributions.
            stored=load(dest/(tag+'_statistics.json'))
            with np.load(dest/(tag+'_draws.npz')) as z:
                for th in (50,70):
                    for pop in POPS:
                        rr=[r for r in stored if r['population']==pop and r['threshold']==th]
                        assert len(rr)==5
                        for r in rr[:4]:
                            orig=load(late/(role+'_'+r['frozen_column']+tier+'_statistics.json'))
                            src=next(a for a in orig if a['population']==pop and a['threshold']==th and a['wait_ms']==r['wait_ms'])
                            assert r['AP']==src['AP'] and r['CI95']==src['CI95']
                        assert abs(rr[-1]['AP']-sum(r['AP'] for r in rr[:4])/4)<1e-10
                        assert np.allclose(rr[-1]['CI95'],np.percentile(z[pop+'_AP'+str(th)].mean(0),[2.5,97.5]),atol=1e-10,rtol=0)
            done.append(ev(dest/(tag+'_statistics.json')))
    atomic(dest/'identity.json',dict(columns=choices,selection_source=ev(P19/'H1_direct_contrasts/panels.json'),
        source_review=ev(BASE/'R6bd_external_late_independent/terminal.json'),all_original_late_columns_retained=True,no_population_or_tier_selection=True))
    atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',statistics=done,
        scope='Independent numeric composite reader; upstream matching/AP/bootstrap separately accepted. No new matching or fits.'))
    record(NAME+'_stronger_late',dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))

def R4_auxiliary():
    import numpy as np
    from peer20_population_core_v1_20261105 import POPS,WAITS
    dest=BASE/(NAME+'_R4_auxiliary');cal=load(P19/'robust_external/calibration.json');selected=cal['selected_global'];rows=[]
    src=BASE/'R6bd_external_modules';gold=load(P19/'R4_independent_read/accepted_decision.json')
    for role in ('val21','mechanism_holdout14'):
        for th in (50,70):
            for pop in POPS:
                hh=[];ee=[];hd=[];ed=[]
                for seed in (2023,2024,2025):
                    htag=role+'_H1_seed'+str(seed);etag=role+'_'+selected+'_seed'+str(seed)
                    for path,pts,tag,draw in [(BASE/'R6b_author',hh,htag,hd),(src,ee,etag,ed)]:
                        points=load(path/(tag+'_statistics.json'))
                        pts.append(next(r['AP'] for r in points if r['population']==pop and r['threshold']==th and r['wait_ms']=='four_wait_mean'))
                        with np.load(path/(tag+'_draws.npz')) as z:draw.append(z[pop+'_AP'+str(th)].mean(0))
                delta=np.mean(ed,axis=0)-np.mean(hd,axis=0)
                row=dict(role=role,threshold=th,population=pop,external_cal_selected=selected,external_AP=float(np.mean(ee)),H1_AP=float(np.mean(hh)),external_minus_H1=float(np.mean(ee)-np.mean(hh)),CI95=np.quantile(delta,[.025,.975]).tolist(),replicates=2000,head_average=True,posthoc_auxiliary=(pop!='all'))
                if role=='val21' and pop=='all' and th==70:
                    assert abs(row['external_minus_H1']-gold['external_minus_H1'])<1e-8
                    assert (row['external_minus_H1']>=-.3)==gold['weaken_claim']
                rows.append(row)
    atomic(dest/'panels.json',rows);atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',
        official_gate_unchanged=ev(P19/'R4_independent_read/accepted_decision.json'),cal_choice_unchanged=True,
        core_review=ev(BASE/'R6b_author_independent/terminal.json'),external_review=ev(BASE/'R6bd_external_modules_independent/terminal.json'),
        scope='Paired shared draws with independently accepted upstream AP/CI; auxiliary columns do not select or adopt.'))
    record(NAME+'_R4_auxiliary',dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))

def tables():
    from publish_peer20_aux_tables_v1_20261105 import table
    import publish_peer20_aux_tables_v1_20261105 as pub
    target=ROOT/'docs/paper_draft_v0_20261103/final_tables_v3/aux_colabelled/coverage_completion_v1'
    pub.OUT=target;target.mkdir(parents=True,exist_ok=False)
    # table() uses its defining module's globals; exact source-row round trip.
    manifests=[]
    for name,paths in [('CoAlign_systems',sorted((BASE/(NAME+'_coalign')).glob('*_statistics.json'))),
        ('frozen_stronger_late',sorted((BASE/(NAME+'_stronger_late')).glob('*_statistics.json'))),
        ('R4_auxiliary',[BASE/(NAME+'_R4_auxiliary/panels.json')])]:manifests.append(table(name,paths))
    atomic(target/'manifest.json',dict(created_at=now(),tables=manifests,official_main_unchanged=True,
        source_reviews=[ev(BASE/(NAME+'_'+n)/'terminal.json') for n in ('coalign_independent','stronger_late','R4_auxiliary')],
        no_physical_truth_claim=True,original_release_closed=False))
    dest=BASE/(NAME+'_tables');atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',manifest=ev(target/'manifest.json')))
    record(NAME+'_tables',dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))

def run():
    # This runner is launched only after the finite CPU DAG drains, never as
    # another dataset-holding waiter alongside its two active heavy workers.
    assert load(BASE/'finite_resource_DAG_v1/terminal.json')['status']=='complete_light_checked'
    for name in ('R6bd_external_late_independent','R6bd_external_modules_independent','R6b_author_independent'):
        assert load(BASE/name/'terminal.json')['status']=='complete_independently_reviewed'
    coalign(False);coalign(True);stronger_late();R4_auxiliary();tables()

def chain():
    name=NAME+'_chain'
    try:
        predecessor=BASE/'finite_resource_DAG_v1/terminal.json'
        record(name,dict(status='waiting_finite_CPU_DAG_no_datasets_loaded',resource_owner=False))
        while not predecessor.exists():time.sleep(30)
        record(name,dict(status='running_known_population_coverage_successors',resource_owner=True))
        run()
        atomic(OUT/'chain_terminal.json',dict(created_at=now(),status='complete_light_checked',
            coverage_tables=ev(BASE/(NAME+'_tables')/'terminal.json'),no_model_wakeup=True))
        record(name,dict(status='complete_light_checked',terminal=ev(OUT/'chain_terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(OUT/'chain_terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(name,dict(status='failed_retained',terminal=ev(OUT/'chain_terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','chain'));globals()[p.parse_args().mode]()
