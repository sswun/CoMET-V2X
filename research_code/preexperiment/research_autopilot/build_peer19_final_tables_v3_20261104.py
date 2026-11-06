"""R2 new v3 paper tables; immutable source operands, per-asset review scope.

Never overwrite v2, tex, legacy ledger or old scientific results. Final H1,
old N0/N2 controls and system/module/diagnostic identities remain separate.
"""
import argparse,csv,io,math,os,time,traceback
import numpy as np
from peer19_v1_20261104 import *
import run_peer19_H1_assets_v1_20261104 as assets
OUT=ROOT/'docs/paper_draft_v0_20261103/final_tables_v3'
JOB=BASE/'final_tables_v3'
OLD=ROOT/'docs/paper_draft_v0_20261103/final_tables_v2'

def prepare():
    register('final_tables_v3',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='bounded36h dependencies',disk_MiB='<40 numeric ledger/CSV, no duplicated prediction/GT'),
        deadline='2026-11-05T18:00:00+08:00',previous=ev(OLD/'manifest.json'),
        scientific_jobs=['R1 five domains/AP/CI','H1 direct390','H1 assets','OPV full2170/V2 H1','R4 same-H1-head robust external','R5 three source numbers'],
        CoAlign_R3='Optional if independently complete at build, otherwise separate open11-06 06:00; never block P0 v3 awaiting P1',
        output=str(OUT.relative_to(ROOT)),main_method='CoMET_all_ego_H1',N2_in_main_method=False)

def run():
    begin=time.monotonic();record('final_tables_v3',dict(status='waiting_actual_H1_scientific_inputs',resource_owner=False,owner=None))
    try:
        for d in assets.recipe.DOMAINS:
            root=BASE/('recipe_independent_interface_recovery_v2' if d=='strict60' else 'recipe_independent')/d
            wait(root/'terminal.json',36,accepted=('complete_independently_reviewed',))
        for root in (BASE/'H1_direct_contrasts',BASE/'H1_assets_independent',BASE/'H1_boundaries/opv_independent',BASE/'H1_boundaries/v2_independent',BASE/'robust_external_independent',BASE/'legacy_three_sources'):
            wait(root/'terminal.json',36,accepted=('complete_independently_reviewed',))
        assert not (OUT/'numeric_ledger.json').exists();OUT.mkdir(exist_ok=True);cells=[];cache={};refs={};sources={};reviews={}
        kind=load(assets.REC/'method_registry.json')['selected_kind']
        def addressed(path,ptr):
            path=path.resolve()
            if path not in cache:cache[path]=load(path);refs[path]=ev(path)
            x=cache[path]
            for k in ptr:x=x[k]
            return float(x),dict(source=refs[path],pointer=list(ptr),format='json')
        def add(table,row,metric,path,ptr,review,scope,op='identity',pointers=None):
            pointers=[ptr] if pointers is None else pointers;vs=[];ops=[]
            for pointer in pointers:
                v,o=addressed(path,pointer);vs.append(v);ops.append(o)
            value={'identity':lambda:vs[0],'mean':lambda:float(np.mean(vs)),'sample_SD':lambda:float(np.std(vs,ddof=1))}[op]()
            assert math.isfinite(value)
            if review not in reviews:reviews[review]=ev(review)
            cells.append(dict(table=table,row=row,metric=metric,value=value,operation=op,factor=1.,operands=ops,
                review_status='independent_scoped;see_source_terminal',review_evidence=reviews[review],scope=scope))
        def scalar_leaves(x,ptr=()):
            if isinstance(x,bool):return
            if isinstance(x,(float,int)):
                yield ptr,x
            elif isinstance(x,list):
                for i,y in enumerate(x):yield from scalar_leaves(y,ptr+(i,))
            elif isinstance(x,dict):
                for k,y in x.items():
                    if k in ('identity','source','predictions','operands','weights','head','script','input','scope','geometry','reference_population_source') or k.endswith(('_error','_mismatches')):continue
                    yield from scalar_leaves(y,ptr+(k,))
        def panel(table,path,review,scope,select=None):
            data=load(path);sources[table]=ev(path)
            if isinstance(data,dict):data=[data];rootdict=True
            else:rootdict=False
            for i,r in enumerate(data):
                if select and not select(r):continue
                tag='|'.join(str(r.get(k,'')) for k in ('domain','role','method','variant','noise','module','filter','kind','source','reference','seed','head_seed','wait_ms','threshold')) if isinstance(r,dict) else ''
                row=str(i)+'|'+tag
                for ptr,v in scalar_leaves(r):
                    if not ptr or ptr[0] in ('seed','head_seed','wait_ms','threshold'):continue
                    add(table,row,'value__'+'__'.join(map(str,ptr)),path,ptr if rootdict else (i,)+ptr,review,scope)
        # Only unchanged/non-H1 control tables are reused; never stale main rows.
        preserve={'A6_rigid_dose_response','E1_speed_coupling','readout_held_site','readout_held_site_CI','FFNet_adaptation_auxiliary',
                  'bounded_neural_all_formal_AP','bounded_preregistered_gate','A2_training_test_factors','A7_inference_code_paths'}
        oldcells=load(OLD/'numeric_ledger.json')['cells']
        for c in oldcells:
            if c['table'] in preserve or (c['table']=='table1' and not c['row'].startswith('CoMET_')):
                cells.append(dict(c,table='legacy_unchanged_'+c['table'],scope='Explicit unchanged control, never current H1. '+c['scope']))
        for domain in assets.recipe.DOMAINS:
            src=assets.REC/domain;review=BASE/('recipe_independent_interface_recovery_v2' if domain=='strict60' else 'recipe_independent')/domain/'terminal.json'
            panel('H1_cross_detector',src/'panels.json',review,'Actual H1 own-input/current-q/selected '+kind+' no-position head; all val+hold signs retained.',select=lambda r:r['kind']==kind)
            pp=load(src/'panels.json')
            for role in ('val21','mechanism_holdout14'):
                for w in (0,100,200,400):
                    ids=[i for i,r in enumerate(pp) if r['kind']==kind and r['role']==role and r['wait_ms']==w];assert len(ids)==3
                    for th in (50,70):
                        for op in ('mean','sample_SD'):add('table1' if domain=='author' and role=='val21' else 'H1_cross_detector_summary',domain+'|'+role+'|'+str(w),'AP'+str(th)+'_'+op,src/'panels.json',(),review,'SD over head seeds, not detector seeds; global kind selected only on five-domain cal8.',op,[(i,'ap'+str(th)) for i in ids])
            panel('learning_untrained_H1_ablation',src/'ablation_panels.json',review,'Original q/tau/frozen3heads controls, not reselected final H1; no claim learning benefit.',None)
            panel('H1_sensitivity',src/'sensitivity_panels.json',review,'All predefined score and association gates, analysis only; not val selection.',select=lambda r:r['kind']==kind)
            panel('H1_readout_fitcal_comparison',src/'calibration_panels.json',review,'cal-only frozen vs geometry-refit/tau grid; no formal-role choice.')
        panel('H1_direct_CI',BASE/'H1_direct_contrasts/panels.json',BASE/'H1_direct_contrasts/terminal.json','390 direct paired2000CI, actual native/stronger-late/method-specific matched module readouts; not full modules as systems.')
        rev=BASE/'H1_assets_independent/terminal.json'
        panel('H1_temporal_aggregation',assets.OUT/'temporal/panels.json',rev,'All-ego observations and exact current H1 three heads; EMA/window cal-only.')
        panel('H1_temporal_CI',BASE/'H1_assets_independent/temporal_bootstrap.json',rev,'Independent AP/statistic kernel,2000 whole-sequence draws.')
        panel('H1_source_AP',assets.OUT/'sources/source_AP.json',rev,'Other-source ignore AP not additive attribution; original labels, official allGT unchanged.')
        panel('H1_fixed_FP_source_TP',assets.OUT/'sources/fixed_FP_TP.json',rev,'Prespecified GT-normalized FP budgets; diagnostic, not deployable GT budget selection.')
        panel('H1_A1_retention',assets.OUT/'A1/retention.json',rev,'Same detector/head and matched ego/reference; poles/negative intervals retained. Reference/CI kernel shared, point AP/ratio independent.')
        for a in ('A3','A4','A5'):panel('H1_'+a,assets.OUT/a/'panels.json',rev,'Actual H1 diagnostic. MA5 annotation-assisted/ICP selected proxy; A5 point equations independently checked, mean-field not physical bound.')
        panel('H1_robust_external',BASE/'robust_external/panels.json',BASE/'robust_external_independent/terminal.json','Exact same H1 weights/prefix on cached external module geometry; robust parameters cal8 only; not full external systems.')
        panel('H1_robust_external_CI',BASE/'robust_external_independent/bootstrap.json',BASE/'robust_external_independent/terminal.json','Paired2000 sequenceCI vs H1, all methods/filters retained.')
        for boundary in ('opv','v2'):
            panel('H1_'+boundary+'_boundary',BASE/'H1_boundaries'/boundary/'panels.json',BASE/'H1_boundaries'/(boundary+'_independent/terminal.json'),'Exact author H1 q/tau/no-position3heads transfer, no targetfit; OPV full2170/allCAV P1; V2 original BEV population/no postROI. Not old ownhead relabeling.')
        panel('legacy_strict_statistics_resolved',BASE/'legacy_three_sources/strict_panels.json',BASE/'legacy_three_sources/terminal.json','Source timestamp arithmetic; 45/446ms are rounded45.391/445.5025ms; no historical ledger overwrite.')
        # Anchor/KF trajectory arrays are referenced, not duplicated into tables.
        refs_only={d:dict(calibration=ev(assets.REC/d/'calibration.json'),states={r:ev(assets.REC/d/(r+'_state.npz')) for r in ('fit24','calibration8','val21','mechanism_holdout14')}) for d in assets.recipe.DOMAINS}
        for table in sorted({c['table'] for c in cells}):
            s=io.StringIO();w=csv.DictWriter(s,fieldnames=['row','metric','value','review_status','scope','source_operands']);w.writeheader()
            for c in cells:
                if c['table']==table:w.writerow({k:c[k] for k in ('row','metric','value','review_status','scope')}|dict(source_operands=__import__('json').dumps(c['operands'],ensure_ascii=False)))
            path=OUT/(table+'.csv');tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()));tmp.write_text(s.getvalue());tmp.replace(path)
        atomic(OUT/'numeric_ledger.json',dict(created_at=now(),cells=cells))
        atomic(OUT/'manifest.json',dict(created_at=now(),status='generated_pending_numeric_review',generator=ev(__file__),method=ev(assets.REC/'method_registry.json'),previous_v2_unchanged=True,
            H1_anchor_KF_trajectory_references=refs_only,scientific_sources=sources,robust_claim_read=ev(BASE/'robust_external/preregistered_read.json'),legacy_decisions=ev(BASE/'legacy_three_sources/decisions.json'),
            P1_CoAlign_open_if_not_reviewed=not (BASE/'coalign_H1_independent/terminal.json').exists(),FFNet_table1_included=False,
            original_release_closed=False,remaining=['Original four legacy discrepancies retained','A5 no physical error bound','K2/K3/K4/K5 evidence preserved, no new successor','R3 separate deadline11-06 06:00'],
            numeric_ledger=ev(OUT/'numeric_ledger.json'),no_automatic_main_tex_edits=True))
        import review_peer17_final_tables_v2_20261104 as reviewer
        reviewer.OUT=OUT;audit=reviewer.run();assert audit['status']=='passed',audit
        atomic(JOB/'terminal.json',dict(status='complete_light_checked',created_at=now(),numeric_audit=ev(OUT/'independent_numeric_audit.json'),cells=len(cells),wall_s=time.monotonic()-begin,whole_release_closed=False))
        record('final_tables_v3',dict(status='complete_light_checked',terminal=ev(JOB/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(JOB/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('final_tables_v3',dict(status='failed_retained',terminal=ev(JOB/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
