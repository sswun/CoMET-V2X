"""Actual R6 auxiliary tables, all registered independent dependencies required.

CSV long tables retain per-row JSON source path/row/SHA and exact field values.
Do not touch root main tables/v2/tex. This is not original-release closure.
Also compute paired external-minus-H1 from common bootstrap draws, never by
subtracting confidence interval endpoints or selecting by population outcomes.
"""
import argparse,csv,hashlib,json,traceback
import numpy as np
from peer20_population_core_v1_20261105 import *
import run_peer20_external_population_v1_20261105 as ext
import recover_peer20_external_sources_v2_20261105 as recovery
OUT=ROOT/'docs/paper_draft_v0_20261103/final_tables_v3/aux_colabelled'
WORK=BASE/'R6_aux_tables'

def external(f):return BASE/(('R6bd_external_v2_' if f in recovery.FAMILIES else 'R6bd_external_')+f)
def prepare():
    register('R6_aux_tables',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<80 exact source-linked CSV/ledger'),
        output=str(OUT),core_domains=list(h1.recipe.DOMAINS),external_families=list(ext.FAMILIES),
        no_new_selection=True,independent_numeric_audit='Every CSV source path/row and numeric/list value reloaded and exact-compared; matching/statistics independently required upstream.')

def table(name,paths):
    target=OUT/(name+'.csv');assert not target.exists(),target
    manifest=[];count=0
    with target.open('x',newline='') as stream:
        fields=('source_path','source_sha256','source_row','values_json');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for path in paths:
            ref=ev(path);data=load(path);assert isinstance(data,list),path
            for i,row in enumerate(data):
                writer.writerow(dict(source_path=ref['path'],source_sha256=ref['sha256'],source_row=i,values_json=json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)));count+=1
            manifest.append(dict(source=ref,rows=len(data)))
    # Separate round-trip reader; no trust in the number of emitted rows.
    cached={};checked=0
    with target.open(newline='') as stream:
        for row in csv.DictReader(stream):
            path=ROOT/row['source_path']
            if row['source_path'] not in cached:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==row['source_sha256'];cached[row['source_path']]=load(path)
            assert json.loads(row['values_json'])==cached[row['source_path']][int(row['source_row'])];checked+=1
    assert checked==count
    return dict(table=ev(target),rows=count,sources=manifest,independent_numeric_roundtrip=True,scientific_AP_matching_bootstrap_independently_accepted_upstream=True)

def paired_external():
    done=[]
    for family in ext.FAMILIES:
        src=external(family);domain={'native_strict60':'strict60','native_vit2024':'v2xvit_seed2024','native_vit2025':'v2xvit_seed2025','native_where':'where2comm'}.get(family,'author');core=BASE/('R6b_'+domain)
        for path in sorted(src.glob('*_statistics.json')):
            tag=path.name.removesuffix('_statistics.json')
            if tag.endswith(('_gap1s','_gap3s','_all_gaps')):continue
            rows=load(path);role='mechanism_holdout14' if tag.startswith('mechanism_holdout14_') else 'val21'
            label=tag.removeprefix(role+'_');native_family=family.startswith('native_');seed_in_label=next((s for s in h1.recipe.SEEDS if label.endswith('_seed'+str(s))),None)
            with np.load(src/(tag+'_draws.npz')) as z:external_draws={k:z[k] for k in z.files}
            values={(r['population'],r['threshold'],r['wait_ms']):r for r in rows}
            for seed in (h1.recipe.SEEDS if seed_in_label is None else (seed_in_label,)):
                ct=role+'_H1_seed'+str(seed);gold=load(core/(ct+'_statistics.json'));pts={(r['population'],r['threshold'],r['wait_ms']):r for r in gold}
                with np.load(core/(ct+'_draws.npz')) as z:cd={k:z[k] for k in z.files}
                for pop in POPS:
                    for th in (50,70):
                        key=pop+'_AP'+str(th);draw=cd[key]-external_draws[key]
                        for iw,w in enumerate((*WAITS,'four_wait_mean')):
                            a,b=pts[pop,th,w],values[pop,th,w]
                            if iw<4:assert a['GT']==b['GT'],(family,tag,pop,th,w,a['GT'],b['GT'])
                            dd=draw[iw] if iw<4 else draw.mean(0)
                            done.append(dict(domain=domain,role=role,head_seed=seed,external_family=family,external_label=label,population=pop,threshold=th,wait_ms=w,
                                H1_minus_external_pp=a['AP']-b['AP'],CI95=np.quantile(dd,[.025,.975]).tolist(),replicates=2000,paired_draws=True,
                                system_comparison=native_family,module_comparison=family=='modules',different_detector_system=family in ('native_vit','native_fcooper'),
                                core_statistics=ev(core/(ct+'_statistics.json')),external_statistics=ev(path)))
    atomic(WORK/'external_contrasts.json',done)

def run():
    name='R6_aux_tables'
    try:
        for d in h1.recipe.DOMAINS:
            for prefix,suffix in [('R6b_','_independent'),('R6d_','_independent'),('R6bc_paired_','')]:wait(BASE/(prefix+d+suffix)/'terminal.json')
        for f in ext.FAMILIES:wait(external(f).with_name(external(f).name+'_independent')/'terminal.json')
        for n in ('R6c_independent','R6c_A5_cluster_independent','R6e_choice_independent'):wait(BASE/n/'terminal.json')
        record(name,dict(status='running_actual_auxiliary_table_generation',resource_owner=True))
        OUT.mkdir(parents=True,exist_ok=True);manifest=[]
        corepaths=[p for d in h1.recipe.DOMAINS for p in sorted((BASE/('R6b_'+d)).glob('*_statistics.json'))]
        auxiliary=[p for d in h1.recipe.DOMAINS for p in sorted((BASE/('R6d_'+d)).glob('*_statistics.json'))]
        externalpaths=[p for f in ext.FAMILIES for p in sorted(external(f).glob('*_statistics.json'))]
        contrasts=[BASE/('R6bc_paired_'+d)/'contrasts.json' for d in h1.recipe.DOMAINS]
        ratios=[BASE/('R6bc_paired_'+d)/'A1_retention.json' for d in h1.recipe.DOMAINS]
        paired_external()
        for label,paths in [('core_official_populations',corepaths),('core_auxiliary_GT',auxiliary),('external_all_protocols',externalpaths),('core_paired_contrasts',contrasts),('A1_retention',ratios),('external_paired_contrasts',[WORK/'external_contrasts.json']),('A5_ACF_cluster',[BASE/'R6c_A5_cluster/panels.json'])]:manifest.append(table(label,paths))
        atomic(OUT/'peer20_manifest.json',dict(created_at=now(),tables=manifest,main_tables_unchanged=True,
            population_definitions='all/exact-zero/actual-co-labelled/nonzero; cal8 zero and co-labelled overlap35 frames',
            posthoc=True,official_GT_primary=True,filled_GT_label_assisted_not_physical_truth=True,clean_choice=ev(BASE/'R6e_choice_independent/decision.json'),
            primary_H1_head_identity='Original H1 for R6b/c/d interpretation; if clean wins these remain explicitly original-head diagnostics, never renamed new main',
            R7_separate=ev(BASE/'R7_preregistered_read/contract.json'),original_release_closed=False))
        atomic(WORK/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',manifest=ev(OUT/'peer20_manifest.json'),scope='R6 tables with upstream scientific acceptance + numeric roundtrip only, not original release or whole paper.'))
        record(name,dict(status='complete_independently_reviewed',terminal=ev(WORK/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(WORK/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(WORK/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));globals()[p.parse_args().mode]()
