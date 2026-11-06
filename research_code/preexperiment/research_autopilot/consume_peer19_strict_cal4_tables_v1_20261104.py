"""Append actual H1 cal4 evidence only after unchanged recipe is independently confirmed.

Frozen generator/integration remain intact. Preserve their small root files in
a version directory, append source-addressed cal4 AP/selection numbers, and
rerun independent numeric serialization/arithmetic audit. A changed choice
does NOT relabel old formal predictions: leave table untouched and flag the
required corrected recipe for the next scheduled judgment.
"""
import argparse,csv,io,json,shutil,time,traceback
from peer19_v1_20261104 import *
NAME='strict_cal4_H1_table_completion'
OUT=BASE/NAME
TABLE=ROOT/'docs/paper_draft_v0_20261103/final_tables_v3'

def prepare():
    register(NAME,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        wall_h='bounded36h dependencies',disk_MiB='<20 small table versions/ledger, no candidate/GT copy'),
        dependencies=['strict_cal4_H1_independent','final_tables_v3_integration'],
        correction_contract=ev(BASE/'strict_cal4_H1/contract.json'),
        selection_rule='Only integrate unchanged q, both taus and global chosen kind. Otherwise corrected scientific successor, never relabel.',
        deadline='2026-11-05T18:00:00+08:00')

def run():
    start=time.monotonic();record(NAME,dict(status='waiting_independent_H1_cal4_and_v3',resource_owner=False,owner=None))
    try:
        wait(BASE/'strict_cal4_H1_independent/terminal.json',12,accepted=('complete_independently_reviewed',))
        decision=load(BASE/'strict_cal4_H1_independent/decision.json')
        if not decision['recipe_identity_unchanged']:
            atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',table_updated=False,
                final_v3_accepted=False,requires_new_scientific_recipe=True,decision=ev(BASE/'strict_cal4_H1_independent/decision.json'),
                next_judgment='Next regular3h calendar, no event wakeup',scope='Coverage diagnostic only; old predictions not relabeled.'))
            record(NAME,dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));return
        wait(BASE/'final_tables_v3_integration/terminal.json',36,accepted=('complete_light_checked',))
        record(NAME,dict(status='running_versioned_H1_cal4_table_completion',resource_owner=True))
        archive=TABLE/'versions/before_strict_H1_cal4_v1';archive.mkdir(parents=True,exist_ok=False)
        for p in TABLE.iterdir():
            if p.is_file():shutil.copy2(p,archive/p.name)
        cells=load(TABLE/'numeric_ledger.json')['cells'];manifest=load(TABLE/'manifest.json')
        source=BASE/'strict_cal4_H1_independent/AP.json';ref=ev(source);review=ev(BASE/'strict_cal4_H1_independent/terminal.json')
        scope='Actual4wait calibration8 H1 strict60, same selected q/tau/heads independently confirmed. Prior strict H1_readout_fitcal_comparison0/400 remains explicitly legacy; use this supplement for full calibration evidence. Not formal-role selection.'
        def add(table,row,metric,value,source,pointer):
            cells.append(dict(table=table,row=row,metric=metric,value=float(value),operation='identity',factor=1.,
                operands=[dict(source=source,pointer=pointer,format='json')],review_status='independent_H1_cal4_AP_and_choice',review_evidence=review,scope=scope))
        for i,r in enumerate(load(source)):
            row='|'.join(map(str,(r['phase'],r['method'],r['wait_ms'])))
            for metric in ('AP50','AP70'):add('H1_strict_cal4',row,metric,r[metric],ref,[i,metric])
        src=BASE/'strict_cal4_H1_independent/decision.json';ref=ev(src)
        add('H1_cal4_selection','strict60','q',decision['selected_q'],ref,['selected_q'])
        for kind in ('frozen','refit'):
            add('H1_cal4_selection','strict60|'+kind,'tau',decision['tau'][kind],ref,['tau',kind])
            add('H1_cal4_selection','all5|'+kind,'cal8_AP70',decision['global_cal8_means'][kind],ref,['global_cal8_means',kind])
        for table in ('H1_strict_cal4','H1_cal4_selection'):
            stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['row','metric','value','review_status','scope','source_operands']);writer.writeheader()
            for c in cells:
                if c['table']==table:writer.writerow({k:c[k] for k in ('row','metric','value','review_status','scope')}|dict(source_operands=json.dumps(c['operands'])))
            (TABLE/(table+'.csv')).write_text(stream.getvalue())
        atomic(TABLE/'numeric_ledger.json',dict(created_at=now(),cells=cells))
        manifest.update(created_at=now(),H1_actual_cal4=ev(src),H1_actual_cal4_review=review,
            prior_strict_cal2_explicitly_legacy=True,actual_main_recipe_identity_unchanged=True,
            cal4_consumer=ev(__file__),before_cal4=ev(archive/'manifest.json'),numeric_ledger=ev(TABLE/'numeric_ledger.json'))
        atomic(TABLE/'manifest.json',manifest)
        import review_peer17_final_tables_v2_20261104 as auditor
        auditor.OUT=TABLE;result=auditor.run();assert result['status']=='passed',result
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',table_updated=True,
            recipe_identity_unchanged=True,numeric_audit=ev(TABLE/'independent_numeric_audit.json'),cells=len(cells),
            scope='Independent cal4 scientific scope and independent root numeric audit, not whole-paper acceptance.',wall_s=time.monotonic()-start))
        record(NAME,dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(NAME,dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
