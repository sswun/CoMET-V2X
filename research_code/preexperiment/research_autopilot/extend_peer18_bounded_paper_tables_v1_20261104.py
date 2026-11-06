"""Versioned P0 extension after independent bounded-round evidence, not tex edit.

Keep N0/main tables immutable in meaning. Add all neural own/zero-shot AP,
full and partial gate arithmetic, actual selected-component direct contrasts,
and A6 fixed-score diagnostic curves. Never relabel old N0 as new method.
"""
import argparse,csv,fcntl,io,math,os,shutil,time,traceback
from peer18_v1_20261104 import *
import build_peer17_final_tables_v2_20261104 as numeric
OUT=ROOT/'docs/paper_draft_v0_20261103/final_tables_v2'
JOB=BASE/'bounded_paper_tables_v1'
DEPS=('adoption_independent_v1','accepted_component_direct_contrasts_v1','A6_rigid_sensitivity_v1')

def prepare():
    register('bounded_paper_tables_v1',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        disk_MiB='<30 new CSV/ledger plus one previous version; shared inputs',wall_h='bounded27h dependencies'),
        previous_manifest=ev(OUT/'manifest.json'),dependencies={n:ev(BASE/n/'contract.json') for n in DEPS},
        deadline_P0='2026-11-05T12:00:00+08:00',input_policy='Every new number source pointer+SHA+review; old scientific results and main.tex untouched.')

def run():
    begin=time.monotonic();record('bounded_paper_tables_v1',dict(status='waiting_accepted_bounded_round_evidence',resource_owner=False,owner=None))
    try:
        for n in DEPS:
            p=BASE/n/'terminal.json'
            while not p.exists():assert time.monotonic()-begin<27*3600;time.sleep(30)
            assert load(p)['status']=='complete_independently_reviewed',(n,load(p))
        record('bounded_paper_tables_v1',dict(status='versioning_accepted_bounded_paper_tables',resource_owner=True))
        with (AUTO/'.paper_table_update.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            version=OUT/'versions/cycle12_before_bounded_round';version.mkdir(exist_ok=False)
            for n in ('numeric_ledger.json','manifest.json','independent_numeric_audit.json'):shutil.copyfile(OUT/n,version/n)
            cells=load(OUT/'numeric_ledger.json')['cells'];before=len(cells);newtables=set()
            def add(table,row,metric,path,pointer,review,scope,status='independent_AP_and_statistics'):
                op=numeric.operand(path,pointer);v=float(numeric.value(op));assert math.isfinite(v),(table,row,metric)
                cells.append(dict(table=table,row=row,metric=metric,value=v,operation='identity',factor=1.,operands=[op],
                    review_status=status,review_evidence=ev(review),scope=scope));newtables.add(table)
            for seed in (2023,2024,2025):
                root=BASE/'CoMET_Net/formal'/('seed'+str(seed))
                for p in sorted(root.glob('*/panels.json')):
                    review=BASE/'CoMET_Net/independent'/root.name/p.parent.name/'terminal.json'
                    assert load(review)['status']=='complete_independently_reviewed'
                    for i,r in enumerate(load(p)):
                        row='|'.join(map(str,(seed,r['training_detector'],r['target'],r['variant'],r['role'],r['wait_ms'])))
                        for th in (50,70):add('bounded_neural_all_formal_AP',row,'AP'+str(th),p,(i,'ap'+str(th)),review,
                            'Frozen single600-attempt-window round/cal-loss checkpoint selection. Own-training and author-zero-shot separate. No new detector training;3neural seeds not3detector seeds. Saved-state independent geometry/AP, not independent recurrence.',status='independent_saved_state_geometry_AP')
            p=BASE/'adoption_independent_v1/preregistered_reads.json';review=BASE/'adoption_independent_v1/terminal.json'
            for i,r in enumerate(load(p)['rows']):
                for metric in ('author_val_AP70_delta','five_mean_val_AP70_delta','author_hold_AP70_delta','five_hold_AP70_delta',
                    'author_val_AP50_delta','five_val_AP50_delta','author_AP70_seed_SD','author_delta_seed_SD'):
                    add('bounded_preregistered_gate',r['variant'],metric,p,('rows',i,metric),review,
                        'Independent2000draw statistics. Five checkpoint identities not independent architectures; original full/partial criteria unchanged. Hold negatives retained.')
                for key in ('author_CI95','five_mean_CI95'):
                    for j,suffix in enumerate(('low','high')):add('bounded_preregistered_gate',r['variant'],key+'_'+suffix,p,('rows',i,key,j),review,'CI of paired resample averages, never average CI endpoints.')
                for key in ('by_domain_val_AP70_delta','by_domain_hold_AP70_delta'):
                    for d in r[key]:add('bounded_preregistered_gate',r['variant']+'|'+d,key,p,('rows',i,key,d),review,'Report every checkpoint identity/held role; no cell selection.')
            p=BASE/'accepted_component_direct_contrasts_v1/panels.json';review=BASE/'accepted_component_direct_contrasts_v1/terminal.json'
            for i,r in enumerate(load(p)):
                row=r['comparison']+'|'+str(r['threshold'])+'|'+str(r['wait_ms'])
                for metric in ('delta_pp','CI95_low','CI95_high','positive_sequences','negative_sequences','zero_sequences'):
                    add('accepted_component_direct_CI',row,metric,p,(i,metric),review,
                        'Actual independently matched selected-component predictions vs actual baseline; new direct independent bootstrap, not sum of old effect estimates. Module/system scopes in source identity; N0 rows retained.')
            p=BASE/'A6_rigid_sensitivity_v1/direction_mean_curves.json';review=BASE/'A6_rigid_sensitivity_v1/terminal.json'
            for i,r in enumerate(load(p)):
                row='|'.join(map(str,(r['dose'],r['threshold'],r['wait_ms'])))
                for metric in ('mean_direction_AP','delta_from_dose0_pp'):add('A6_rigid_dose_response',row,metric,p,(i,metric),review,'OFFICIAL-label-assisted fixedscore remote-xy dose response, four predeclared cardinal directions averaged; shared NMS, independent matching/statistics. Diagnostic not deployable.')
                for key in ('AP_CI95','delta_CI95'):
                    for j,suffix in enumerate(('low','high')):add('A6_rigid_dose_response',row,key+'_'+suffix,p,(i,key,j),review,'Paired whole-sequence2000 resampling; direction mean, no best direction/dose selection.')
            for table in sorted(newtables):
                stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=['row','metric','value','review_status','scope','source_operands']);w.writeheader()
                for c in cells:
                    if c['table']==table:w.writerow({k:c[k] for k in ('row','metric','value','review_status','scope')}|dict(source_operands=__import__('json').dumps(c['operands'],ensure_ascii=False)))
                p=OUT/(table+'.csv');tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(stream.getvalue());tmp.replace(p)
            atomic(OUT/'numeric_ledger.json',dict(created_at=now(),cells=cells))
            manifest=load(OUT/'manifest.json');manifest.update(created_at=now(),generator=ev(__file__),numeric_ledger=ev(OUT/'numeric_ledger.json'),
                previous_version=ev(version/'manifest.json'),bounded_round_extension={n:ev(BASE/n/'terminal.json') for n in DEPS},
                current_method_registry=ev(BASE/'adoption_independent_v1/method_registry.json'),old_N0_rows_not_relabelled=True,
                scientific_pending=['Original numberledger discrepancies/unresolved retain original audit; no legacy blanketclosure',
                    'FFNet paper-protocol qualification not established','Additional A1 actual OFFICIAL-trained architectures absent; no broad median',
                    'Legacy physical/source/release coverage and independent recurrence remain separately scoped'],pending=[],whole_release_closed=False)
            atomic(OUT/'manifest.json',manifest)
            import review_peer17_final_tables_v2_20261104 as reviewer
            result=reviewer.run();assert result['status']=='passed',result
            atomic(JOB/'terminal.json',dict(created_at=now(),status='complete_light_checked',previous_cells=before,cells=len(cells),
                added_cells=len(cells)-before,numeric_audit=ev(OUT/'independent_numeric_audit.json'),method_registry=manifest['current_method_registry'],
                main_tex_unchanged=True,whole_release_closed=False,wall_s=time.monotonic()-begin))
        record('bounded_paper_tables_v1',dict(status='complete_light_checked',terminal=ev(JOB/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(JOB/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record('bounded_paper_tables_v1',dict(status='failed_retained',terminal=ev(JOB/'terminal.json'),resource_owner=False,owner=None));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
